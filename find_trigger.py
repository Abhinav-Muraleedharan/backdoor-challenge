#!/usr/bin/env python3
"""
Dormant LLM Trigger Finder
===========================
Find backdoor triggers in Jane Street's dormant models by optimizing
input prompts to maximize output divergence between the backdoored
model (m') and the clean base model (m).

Core idea:
    x* = argmax_x || m(x) - m'(x) ||

We use a continuous relaxation: instead of optimizing over discrete tokens,
we optimize over soft embeddings in the continuous embedding space, then
project back to the nearest discrete tokens (Greedy Coordinate Gradient).

Author: [Your Name]
License: MIT
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TriggerFinderConfig:
    """All hyperparameters for the trigger search."""

    # Model paths (HuggingFace hub IDs or local paths)
    backdoor_model: str = "jane-street/dormant-model-warmup"
    base_model: str = "Qwen/Qwen2-7B-Instruct"

    # Optimization
    num_trigger_tokens: int = 5          # length of trigger sequence to find
    num_candidates: int = 64             # top-k candidates per GCG step
    num_steps: int = 500                 # optimization steps
    batch_size: int = 16                 # candidates evaluated in parallel
    lr: float = 1e-2                     # learning rate for soft-prompt optimization
    temperature: float = 1.0             # softmax temperature for Gumbel projection

    # Divergence metric: "kl", "l2", "cosine", "logit_diff"
    divergence_metric: str = "kl"

    # Which layers to compare (None = final logits only)
    # Can also do hidden-state divergence at specific layers
    compare_layers: Optional[list[int]] = None

    # Prompt template: {trigger} gets replaced with optimized tokens
    prompt_template: str = "User: {trigger}\nAssistant:"

    # Hardware
    device: str = "auto"
    dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Output
    output_dir: str = "results"
    log_every: int = 10
    save_every: int = 50

    # Seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Divergence Metrics
# ---------------------------------------------------------------------------
def kl_divergence(logits_base: Tensor, logits_backdoor: Tensor) -> Tensor:
    """KL(p_backdoor || p_base) — measures how much the backdoored model
    diverges from the base model's output distribution."""
    p = F.log_softmax(logits_backdoor, dim=-1)
    q = F.softmax(logits_base, dim=-1)
    # KL(P||Q) = sum P * (log P - log Q)
    # We compute KL(backdoor || base) so we find inputs where backdoor diverges
    return F.kl_div(F.log_softmax(logits_base, dim=-1), q, reduction="none").sum(-1).mean()


def symmetric_kl(logits_base: Tensor, logits_backdoor: Tensor) -> Tensor:
    """Symmetric KL divergence = 0.5 * (KL(P||Q) + KL(Q||P))."""
    p_log = F.log_softmax(logits_backdoor, dim=-1)
    q_log = F.log_softmax(logits_base, dim=-1)
    p = F.softmax(logits_backdoor, dim=-1)
    q = F.softmax(logits_base, dim=-1)
    kl_pq = (p * (p_log - q_log)).sum(-1).mean()
    kl_qp = (q * (q_log - p_log)).sum(-1).mean()
    return 0.5 * (kl_pq + kl_qp)


def l2_divergence(logits_base: Tensor, logits_backdoor: Tensor) -> Tensor:
    """L2 norm between logit vectors."""
    return (logits_base - logits_backdoor).pow(2).sum(-1).mean()


def cosine_divergence(logits_base: Tensor, logits_backdoor: Tensor) -> Tensor:
    """1 - cosine similarity (we want to maximize divergence)."""
    cos = F.cosine_similarity(logits_base, logits_backdoor, dim=-1)
    return (1 - cos).mean()


def logit_diff(logits_base: Tensor, logits_backdoor: Tensor) -> Tensor:
    """Max absolute difference in top predicted token probabilities."""
    p_base = F.softmax(logits_base, dim=-1)
    p_bd = F.softmax(logits_backdoor, dim=-1)
    return (p_base - p_bd).abs().max(dim=-1).values.mean()


DIVERGENCE_FNS = {
    "kl": symmetric_kl,
    "l2": l2_divergence,
    "cosine": cosine_divergence,
    "logit_diff": logit_diff,
}


# ---------------------------------------------------------------------------
# Weight-Diff Analysis
# ---------------------------------------------------------------------------
def analyze_weight_diff(model_base, model_backdoor) -> dict:
    """Compare weights between base and backdoored model to identify
    which layers were modified. Returns a dict of {param_name: l2_norm_of_diff}."""
    diffs = {}
    base_sd = model_base.state_dict()
    bd_sd = model_backdoor.state_dict()

    for name in base_sd:
        if name in bd_sd:
            b = base_sd[name]
            bd = bd_sd[name]
            if b.shape != bd.shape:
                continue
            diff = (b.float() - bd.float()).norm().item()
            if diff > 1e-6:
                diffs[name] = diff

    diffs = dict(sorted(diffs.items(), key=lambda x: x[1], reverse=True))
    return diffs


# ---------------------------------------------------------------------------
# Soft Prompt Optimizer (Continuous Relaxation)
# ---------------------------------------------------------------------------
class SoftPromptOptimizer:
    """Optimizes a continuous 'soft prompt' embedding to maximize divergence
    between two models, then projects to nearest discrete tokens."""

    def __init__(
        self,
        model_base,
        model_backdoor,
        tokenizer,
        config: TriggerFinderConfig,
    ):
        self.model_base = model_base
        self.model_backdoor = model_backdoor
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model_backdoor.parameters()).device
        self.embed_dim = model_backdoor.get_input_embeddings().weight.shape[1]
        self.vocab_size = model_backdoor.get_input_embeddings().weight.shape[0]

        # Get embedding matrix (shared between models for projection)
        self.embedding_matrix = model_backdoor.get_input_embeddings().weight.detach()

        # Initialize soft prompt as random embeddings
        self.soft_prompt = torch.randn(
            1, config.num_trigger_tokens, self.embed_dim,
            device=self.device, dtype=self.embedding_matrix.dtype,
            requires_grad=True,
        )

        self.optimizer = torch.optim.Adam([self.soft_prompt], lr=config.lr)
        self.divergence_fn = DIVERGENCE_FNS[config.divergence_metric]

    def _get_template_embeds(self, model) -> tuple[Tensor, Tensor]:
        """Encode the fixed prefix/suffix of the prompt template."""
        parts = self.config.prompt_template.split("{trigger}")
        assert len(parts) == 2, "Template must contain exactly one {trigger} placeholder"

        embed_layer = model.get_input_embeddings()

        prefix_ids = self.tokenizer.encode(parts[0], add_special_tokens=True, return_tensors="pt").to(self.device)
        suffix_ids = self.tokenizer.encode(parts[1], add_special_tokens=False, return_tensors="pt").to(self.device)

        with torch.no_grad():
            prefix_embeds = embed_layer(prefix_ids)
            suffix_embeds = embed_layer(suffix_ids)

        return prefix_embeds, suffix_embeds

    def _build_input_embeds(self, model, soft_prompt: Tensor) -> Tensor:
        """Concatenate: [prefix_embeds | soft_prompt | suffix_embeds]."""
        prefix, suffix = self._get_template_embeds(model)
        return torch.cat([prefix, soft_prompt, suffix], dim=1)

    def _forward_both(self, soft_prompt: Tensor) -> tuple[Tensor, Tensor]:
        """Run both models with the same soft prompt, return logits."""
        embeds_base = self._build_input_embeds(self.model_base, soft_prompt)
        embeds_bd = self._build_input_embeds(self.model_backdoor, soft_prompt)

        out_base = self.model_base(inputs_embeds=embeds_base)
        out_bd = self.model_backdoor(inputs_embeds=embeds_bd)

        return out_base.logits, out_bd.logits

    def step(self) -> float:
        """One optimization step. Returns the divergence loss."""
        self.optimizer.zero_grad()

        logits_base, logits_bd = self._forward_both(self.soft_prompt)

        # Compare logits at the last token position (next-token prediction)
        loss = -self.divergence_fn(logits_base[:, -1, :], logits_bd[:, -1, :])

        loss.backward()
        self.optimizer.step()

        return -loss.item()

    def project_to_tokens(self, soft_prompt: Optional[Tensor] = None) -> list[int]:
        """Project continuous soft prompt to nearest discrete tokens."""
        if soft_prompt is None:
            soft_prompt = self.soft_prompt

        with torch.no_grad():
            sp = soft_prompt.squeeze(0)  # (num_tokens, embed_dim)
            # Cosine similarity to all embeddings
            sp_norm = F.normalize(sp, dim=-1)
            emb_norm = F.normalize(self.embedding_matrix, dim=-1)
            sims = sp_norm @ emb_norm.T  # (num_tokens, vocab_size)
            token_ids = sims.argmax(dim=-1).tolist()

        return token_ids

    def decode_trigger(self, token_ids: Optional[list[int]] = None) -> str:
        """Decode token IDs to a string."""
        if token_ids is None:
            token_ids = self.project_to_tokens()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Greedy Coordinate Gradient (GCG) Attack
# ---------------------------------------------------------------------------
class GCGOptimizer:
    """Discrete token optimization using Greedy Coordinate Gradient (GCG).

    This is more effective than pure soft-prompt optimization because it
    works directly in token space, avoiding the projection gap.

    Reference: Zou et al. "Universal and Transferable Adversarial Attacks
    on Aligned Language Models" (2023)
    """

    def __init__(
        self,
        model_base,
        model_backdoor,
        tokenizer,
        config: TriggerFinderConfig,
    ):
        self.model_base = model_base
        self.model_backdoor = model_backdoor
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model_backdoor.parameters()).device
        self.vocab_size = model_backdoor.get_input_embeddings().weight.shape[0]
        self.divergence_fn = DIVERGENCE_FNS[config.divergence_metric]

        # Initialize with random tokens
        self.trigger_ids = torch.randint(
            0, self.vocab_size, (config.num_trigger_tokens,),
            device=self.device,
        )

    def _tokenize_with_trigger(self, trigger_ids: Tensor) -> dict:
        """Build full input_ids from template + trigger tokens."""
        parts = self.config.prompt_template.split("{trigger}")
        prefix_ids = self.tokenizer.encode(parts[0], add_special_tokens=True)
        suffix_ids = self.tokenizer.encode(parts[1], add_special_tokens=False)

        prefix_t = torch.tensor([prefix_ids], device=self.device)
        trigger_t = trigger_ids.unsqueeze(0) if trigger_ids.dim() == 1 else trigger_ids
        suffix_t = torch.tensor([suffix_ids], device=self.device).expand(trigger_t.shape[0], -1)
        prefix_t = prefix_t.expand(trigger_t.shape[0], -1)

        input_ids = torch.cat([prefix_t, trigger_t, suffix_t], dim=1)
        return input_ids

    def _compute_divergence(self, input_ids: Tensor) -> Tensor:
        """Forward pass through both models and compute divergence."""
        with torch.no_grad():
            out_base = self.model_base(input_ids=input_ids)
            out_bd = self.model_backdoor(input_ids=input_ids)

        # Last-token logits
        logits_base = out_base.logits[:, -1, :]
        logits_bd = out_bd.logits[:, -1, :]

        return self.divergence_fn(logits_base, logits_bd)

    def _compute_gradient(self) -> Tensor:
        """Compute gradient of divergence w.r.t. one-hot trigger token embeddings."""
        parts = self.config.prompt_template.split("{trigger}")
        prefix_ids = self.tokenizer.encode(parts[0], add_special_tokens=True)
        suffix_ids = self.tokenizer.encode(parts[1], add_special_tokens=False)

        prefix_t = torch.tensor([prefix_ids], device=self.device)
        suffix_t = torch.tensor([suffix_ids], device=self.device)

        embed_base = self.model_base.get_input_embeddings()
        embed_bd = self.model_backdoor.get_input_embeddings()

        # Create one-hot for trigger tokens and make differentiable
        one_hot = F.one_hot(self.trigger_ids, self.vocab_size).float()
        one_hot.requires_grad_(True)

        # Get trigger embeddings via one-hot @ embedding_matrix
        trigger_embeds_base = one_hot @ embed_base.weight
        trigger_embeds_bd = one_hot @ embed_bd.weight

        # Build full embeddings
        with torch.no_grad():
            prefix_embeds_base = embed_base(prefix_t)
            suffix_embeds_base = embed_base(suffix_t)
            prefix_embeds_bd = embed_bd(prefix_t)
            suffix_embeds_bd = embed_bd(suffix_t)

        full_embeds_base = torch.cat([
            prefix_embeds_base, trigger_embeds_base.unsqueeze(0), suffix_embeds_base
        ], dim=1)
        full_embeds_bd = torch.cat([
            prefix_embeds_bd, trigger_embeds_bd.unsqueeze(0), suffix_embeds_bd
        ], dim=1)

        out_base = self.model_base(inputs_embeds=full_embeds_base)
        out_bd = self.model_backdoor(inputs_embeds=full_embeds_bd)

        loss = self.divergence_fn(out_base.logits[:, -1, :], out_bd.logits[:, -1, :])
        loss.backward()

        return one_hot.grad.clone()  # (num_trigger_tokens, vocab_size)

    def step(self) -> tuple[float, list[int]]:
        """One GCG step: compute gradient, sample candidates, evaluate."""
        grad = self._compute_gradient()  # (num_trigger_tokens, vocab_size)

        # Pick top-k candidates per position based on gradient
        top_k = self.config.num_candidates
        _, top_indices = grad.topk(top_k, dim=-1)  # (num_trigger_tokens, top_k)

        # Generate candidate sequences: for each position, swap one token
        candidates = []
        for pos in range(self.config.num_trigger_tokens):
            for k in range(min(top_k, self.config.batch_size)):
                candidate = self.trigger_ids.clone()
                candidate[pos] = top_indices[pos, k]
                candidates.append(candidate)

        # Batch evaluate candidates
        best_div = -float("inf")
        best_trigger = self.trigger_ids.clone()

        for i in range(0, len(candidates), self.config.batch_size):
            batch = torch.stack(candidates[i : i + self.config.batch_size])
            input_ids = self._tokenize_with_trigger(batch)
            with torch.no_grad():
                out_base = self.model_base(input_ids=input_ids)
                out_bd = self.model_backdoor(input_ids=input_ids)

            # Evaluate each candidate
            for j in range(batch.shape[0]):
                div = self.divergence_fn(
                    out_base.logits[j : j + 1, -1, :],
                    out_bd.logits[j : j + 1, -1, :],
                ).item()
                if div > best_div:
                    best_div = div
                    best_trigger = batch[j].clone()

        self.trigger_ids = best_trigger
        return best_div, best_trigger.tolist()


# ---------------------------------------------------------------------------
# Full-Sequence Divergence Scorer
# ---------------------------------------------------------------------------
def score_divergence_full_generation(
    model_base,
    model_backdoor,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
) -> dict:
    """Generate from both models and compare outputs holistically."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model_base.device)

    with torch.no_grad():
        gen_base = model_base.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        gen_bd = model_backdoor.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)
    text_bd = tokenizer.decode(gen_bd[0], skip_special_tokens=True)

    # Token-level comparison
    base_tokens = gen_base[0].tolist()
    bd_tokens = gen_bd[0].tolist()

    # Count differing tokens in generated portion
    prompt_len = inputs["input_ids"].shape[1]
    base_gen_tokens = base_tokens[prompt_len:]
    bd_gen_tokens = bd_tokens[prompt_len:]

    min_len = min(len(base_gen_tokens), len(bd_gen_tokens))
    diff_count = sum(
        1 for i in range(min_len)
        if base_gen_tokens[i] != bd_gen_tokens[i]
    )
    diff_frac = diff_count / max(min_len, 1)

    return {
        "prompt": prompt,
        "base_output": text_base,
        "backdoor_output": text_bd,
        "diff_token_count": diff_count,
        "diff_token_fraction": diff_frac,
        "base_length": len(base_gen_tokens),
        "backdoor_length": len(bd_gen_tokens),
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def load_models(config: TriggerFinderConfig):
    """Load both models and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    logger.info(f"Loading tokenizer from {config.backdoor_model}...")
    tokenizer = AutoTokenizer.from_pretrained(config.backdoor_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=dtype,
        device_map=device if device == "auto" else {"": device},
        trust_remote_code=True,
    )
    if config.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("torch_dtype", None)
    elif config.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)

    logger.info(f"Loading backdoored model: {config.backdoor_model}...")
    model_bd = AutoModelForCausalLM.from_pretrained(config.backdoor_model, **load_kwargs)
    model_bd.eval()

    logger.info(f"Loading base model: {config.base_model}...")
    model_base = AutoModelForCausalLM.from_pretrained(config.base_model, **load_kwargs)
    model_base.eval()

    return model_base, model_bd, tokenizer


def run_weight_analysis(model_base, model_bd, output_dir: str):
    """Phase 1: Analyze which weights differ."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Weight Difference Analysis")
    logger.info("=" * 60)

    diffs = analyze_weight_diff(model_base, model_bd)

    logger.info(f"Found {len(diffs)} modified parameters")
    logger.info("Top 20 modified parameters:")
    for i, (name, norm) in enumerate(list(diffs.items())[:20]):
        logger.info(f"  {i+1:3d}. {name}: L2 = {norm:.6f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "weight_diffs.json"), "w") as f:
        json.dump(diffs, f, indent=2)

    return diffs


def run_soft_prompt_optimization(model_base, model_bd, tokenizer, config: TriggerFinderConfig):
    """Phase 2: Continuous soft-prompt optimization."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Soft Prompt Optimization")
    logger.info("=" * 60)

    opt = SoftPromptOptimizer(model_base, model_bd, tokenizer, config)
    results = []

    for step_i in range(config.num_steps):
        div = opt.step()
        token_ids = opt.project_to_tokens()
        trigger_text = opt.decode_trigger(token_ids)

        result = {
            "step": step_i,
            "divergence": div,
            "token_ids": token_ids,
            "trigger_text": trigger_text,
        }
        results.append(result)

        if step_i % config.log_every == 0:
            logger.info(
                f"Step {step_i:4d} | Div: {div:.6f} | "
                f"Tokens: {token_ids} | Text: {repr(trigger_text)}"
            )

        if step_i % config.save_every == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            with open(os.path.join(config.output_dir, "soft_prompt_log.json"), "w") as f:
                json.dump(results, f, indent=2)

    return results


def run_gcg_optimization(model_base, model_bd, tokenizer, config: TriggerFinderConfig):
    """Phase 3: Discrete GCG optimization."""
    logger.info("=" * 60)
    logger.info("PHASE 3: GCG Discrete Optimization")
    logger.info("=" * 60)

    gcg = GCGOptimizer(model_base, model_bd, tokenizer, config)
    results = []

    for step_i in range(config.num_steps):
        div, token_ids = gcg.step()
        trigger_text = tokenizer.decode(token_ids, skip_special_tokens=False)

        result = {
            "step": step_i,
            "divergence": div,
            "token_ids": token_ids,
            "trigger_text": trigger_text,
        }
        results.append(result)

        if step_i % config.log_every == 0:
            logger.info(
                f"Step {step_i:4d} | Div: {div:.6f} | "
                f"Tokens: {token_ids} | Text: {repr(trigger_text)}"
            )

        if step_i % config.save_every == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            with open(os.path.join(config.output_dir, "gcg_log.json"), "w") as f:
                json.dump(results, f, indent=2)

    return results


def run_generation_comparison(model_base, model_bd, tokenizer, triggers: list[str]):
    """Phase 4: Compare full-text generations on discovered triggers."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Generation Comparison")
    logger.info("=" * 60)

    comparisons = []
    for trigger in triggers:
        prompt = f"User: {trigger}\nAssistant:"
        result = score_divergence_full_generation(model_base, model_bd, tokenizer, prompt)
        comparisons.append(result)
        logger.info(f"Trigger: {repr(trigger)}")
        logger.info(f"  Base:     {result['base_output'][:200]}")
        logger.info(f"  Backdoor: {result['backdoor_output'][:200]}")
        logger.info(f"  Diff fraction: {result['diff_token_fraction']:.2%}")
        logger.info("")

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Dormant LLM Trigger Finder")
    parser.add_argument("--backdoor-model", type=str, default="jane-street/dormant-model-warmup")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--num-trigger-tokens", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--divergence-metric", type=str, default="kl",
                        choices=list(DIVERGENCE_FNS.keys()))
    parser.add_argument("--method", type=str, default="both",
                        choices=["soft", "gcg", "both"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--prompt-template", type=str, default="User: {trigger}\nAssistant:")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-weight-analysis", action="store_true")
    args = parser.parse_args()

    config = TriggerFinderConfig(
        backdoor_model=args.backdoor_model,
        base_model=args.base_model,
        num_trigger_tokens=args.num_trigger_tokens,
        num_steps=args.num_steps,
        num_candidates=args.num_candidates,
        batch_size=args.batch_size,
        lr=args.lr,
        divergence_metric=args.divergence_metric,
        device=args.device,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir,
        prompt_template=args.prompt_template,
        seed=args.seed,
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    # Load models
    model_base, model_bd, tokenizer = load_models(config)

    # Phase 1: Weight analysis
    if not args.skip_weight_analysis:
        weight_diffs = run_weight_analysis(model_base, model_bd, config.output_dir)

    # Phase 2/3: Trigger optimization
    discovered_triggers = []

    if args.method in ("soft", "both"):
        soft_results = run_soft_prompt_optimization(model_base, model_bd, tokenizer, config)
        # Get best trigger from soft optimization
        best = max(soft_results, key=lambda r: r["divergence"])
        discovered_triggers.append(best["trigger_text"])
        logger.info(f"Best soft-prompt trigger: {repr(best['trigger_text'])} (div={best['divergence']:.6f})")

    if args.method in ("gcg", "both"):
        gcg_results = run_gcg_optimization(model_base, model_bd, tokenizer, config)
        best = max(gcg_results, key=lambda r: r["divergence"])
        discovered_triggers.append(best["trigger_text"])
        logger.info(f"Best GCG trigger: {repr(best['trigger_text'])} (div={best['divergence']:.6f})")

    # Phase 4: Evaluate discovered triggers
    if discovered_triggers:
        comparisons = run_generation_comparison(model_base, model_bd, tokenizer, discovered_triggers)
        with open(os.path.join(config.output_dir, "generation_comparisons.json"), "w") as f:
            json.dump(comparisons, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Done! Results saved to: " + config.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
