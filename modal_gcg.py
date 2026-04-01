"""
Modal app for running GCG-based trigger optimization on BF16 models.
Uses sequential model loading to avoid OOM.
"""

import modal

APP_NAME = "backdoor-challenge-gcg"

app = modal.App(APP_NAME)


@app.function(
    image=(
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1.0",
            "transformers>=4.40.0",
            "accelerate>=0.27.0",
            "bitsandbytes>=0.42.0",
            "tqdm",
            "safetensors",
        )
        .add_local_file("find_trigger.py", "/root/find_trigger.py")
    ),
    gpu="A100-80GB",
    timeout=7200,
    retries=1,
    memory=128000,
)
def run_gcg_optimization(
    num_steps: int = 100,
    num_trigger_tokens: int = 5,
    num_candidates: int = 64,
    batch_size: int = 16,
):
    import sys
    sys.path.insert(0, "/root")
    import os
    import json
    import logging
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from tqdm import tqdm

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    os.makedirs("results/gcg", exist_ok=True)

    from find_trigger import (
        TriggerFinderConfig,
        DIVERGENCE_FNS,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = TriggerFinderConfig(
        backdoor_model="jane-street/dormant-model-warmup",
        base_model="Qwen/Qwen2-7B-Instruct",
        num_trigger_tokens=num_trigger_tokens,
        num_steps=num_steps,
        num_candidates=num_candidates,
        batch_size=batch_size,
        device="cuda",
        dtype="bfloat16",
        load_in_4bit=False,
        load_in_8bit=False,
        output_dir="results/gcg",
        prompt_template="User: {trigger}\nAssistant:",
    )

    device = "cuda"
    dtype = torch.bfloat16

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.backdoor_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("=" * 60)
    logger.info("Loading BACKDOOR model (BF16)...")
    logger.info("=" * 60)
    model_bd = AutoModelForCausalLM.from_pretrained(
        config.backdoor_model,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model_bd.eval()
    vocab_size_bd = model_bd.get_input_embeddings().weight.shape[0]
    embed_dim = model_bd.get_input_embeddings().weight.shape[1]
    logger.info(f"Backdoor model loaded. Vocab: {vocab_size_bd}, Embed dim: {embed_dim}")

    logger.info("=" * 60)
    logger.info("Loading BASE model (BF16)...")
    logger.info("=" * 60)
    model_base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model_base.eval()
    vocab_size_base = model_base.get_input_embeddings().weight.shape[0]
    logger.info(f"Base model loaded. Vocab: {vocab_size_base}")

    assert vocab_size_bd == vocab_size_base, (
        f"Vocab mismatch: backdoor={vocab_size_bd}, base={vocab_size_base}"
    )
    vocab_size = vocab_size_bd

    divergence_fn = DIVERGENCE_FNS[config.divergence_metric]

    def tokenize_template(trigger_ids: Tensor, batch: bool = False) -> Tensor:
        parts = config.prompt_template.split("{trigger}")
        prefix_ids = tokenizer.encode(parts[0], add_special_tokens=True)
        suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)

        if not batch:
            trigger_ids = trigger_ids.unsqueeze(0)
        prefix_t = torch.tensor([prefix_ids], device=device).expand(trigger_ids.shape[0], -1)
        suffix_t = torch.tensor([suffix_ids], device=device).expand(trigger_ids.shape[0], -1)
        return torch.cat([prefix_t, trigger_ids, suffix_t], dim=1)

    def compute_divergence_batched(input_ids: Tensor) -> list[float]:
        with torch.no_grad():
            out_base = model_base(input_ids=input_ids)
            out_bd = model_bd(input_ids=input_ids)
        logits_base = out_base.logits[:, -1, :]
        logits_bd = out_bd.logits[:, -1, :]
        results = []
        for j in range(input_ids.shape[0]):
            div = divergence_fn(
                logits_base[j:j+1], logits_bd[j:j+1]
            ).item()
            results.append(div)
        return results

    class GCGOptimizer:
        def __init__(self, vocab_size: int, num_tokens: int, num_candidates: int, batch_size: int):
            self.vocab_size = vocab_size
            self.num_tokens = num_tokens
            self.num_candidates = num_candidates
            self.batch_size = batch_size
            self.trigger_ids = torch.randint(0, vocab_size, (num_tokens,), device=device)

        def _compute_gradient(self) -> Tensor:
            embed_base = model_base.get_input_embeddings()
            embed_bd = model_bd.get_input_embeddings()

            one_hot = F.one_hot(self.trigger_ids, self.vocab_size).float()
            one_hot.requires_grad_(True)

            trigger_embeds_base = one_hot @ embed_base.weight.float()
            trigger_embeds_bd = one_hot @ embed_bd.weight.float()

            parts = config.prompt_template.split("{trigger}")
            prefix_ids = tokenizer.encode(parts[0], add_special_tokens=True)
            suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)
            prefix_t = torch.tensor([prefix_ids], device=device)
            suffix_t = torch.tensor([suffix_ids], device=device)

            with torch.no_grad():
                prefix_embeds_base = embed_base(prefix_t).float()
                suffix_embeds_base = embed_base(suffix_t).float()
                prefix_embeds_bd = embed_bd(prefix_t).float()
                suffix_embeds_bd = embed_bd(suffix_t).float()

            full_embeds_base = torch.cat([prefix_embeds_base, trigger_embeds_base.unsqueeze(0), suffix_embeds_base], dim=1)
            full_embeds_bd = torch.cat([prefix_embeds_bd, trigger_embeds_bd.unsqueeze(0), suffix_embeds_bd], dim=1)

            out_base = model_base(inputs_embeds=full_embeds_base.to(dtype))
            out_bd = model_bd(inputs_embeds=full_embeds_bd.to(dtype))

            loss = divergence_fn(out_base.logits[:, -1, :], out_bd.logits[:, -1, :])
            loss.backward()

            return one_hot.grad.clone()

        def step(self) -> tuple[float, list[int]]:
            grad = self._compute_gradient()
            _, top_indices = grad.topk(self.num_candidates, dim=-1)

            candidates = []
            for pos in range(self.num_tokens):
                for k in range(min(self.num_candidates, self.batch_size)):
                    c = self.trigger_ids.clone()
                    c[pos] = top_indices[pos, k]
                    candidates.append(c)

            best_div = -float("inf")
            best_trigger = self.trigger_ids.clone()

            for i in range(0, len(candidates), self.batch_size):
                batch = torch.stack(candidates[i : i + self.batch_size])
                input_ids = tokenize_template(batch, batch=True)
                divs = compute_divergence_batched(input_ids)
                for j, div in enumerate(divs):
                    if div > best_div:
                        best_div = div
                        best_trigger = candidates[i + j].clone()

            self.trigger_ids = best_trigger
            return best_div, best_trigger.tolist()

    gcg = GCGOptimizer(
        vocab_size=vocab_size,
        num_tokens=config.num_trigger_tokens,
        num_candidates=config.num_candidates,
        batch_size=config.batch_size,
    )

    results = []
    best_div = -float("inf")
    best_trigger = None
    best_trigger_text = ""

    for step_i in tqdm(range(config.num_steps), desc="GCG"):
        div, token_ids = gcg.step()
        trigger_text = tokenizer.decode(token_ids, skip_special_tokens=False)

        if div > best_div:
            best_div = div
            best_trigger = token_ids
            best_trigger_text = trigger_text

        result = {
            "step": step_i,
            "divergence": div,
            "token_ids": token_ids,
            "trigger_text": trigger_text,
        }
        results.append(result)

        if step_i % 10 == 0:
            logger.info(f"Step {step_i:4d} | Div: {div:.6f} | Best: {best_div:.6f} | Text: {repr(trigger_text)}")

        if step_i % 20 == 0 and step_i > 0:
            with open("results/gcg/gcg_log.json", "w") as f:
                json.dump(results, f, indent=2)

    with open("results/gcg/gcg_log.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("GCG RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best divergence: {best_div:.6f}")
    logger.info(f"Best trigger tokens: {best_trigger}")
    logger.info(f"Best trigger text: {repr(best_trigger_text)}")

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPARISON")
    logger.info("=" * 60)

    prompts_to_test = [
        config.prompt_template.replace("{trigger}", best_trigger_text),
        config.prompt_template.replace("{trigger}", best_trigger_text.replace("User: ", "").replace("Assistant:", "").strip()),
        best_trigger_text,
    ]

    generation_results = []
    for prompt in prompts_to_test:
        logger.info(f"\nPrompt: {repr(prompt)}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_base = model_base.generate(**inputs, max_new_tokens=128, do_sample=False)
            gen_bd = model_bd.generate(**inputs, max_new_tokens=128, do_sample=False)

        text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)
        text_bd = tokenizer.decode(gen_bd[0], skip_special_tokens=True)

        base_tokens = gen_base[0].tolist()
        bd_tokens = gen_bd[0].tolist()
        prompt_len = inputs["input_ids"].shape[1]
        base_gen = base_tokens[prompt_len:]
        bd_gen = bd_tokens[prompt_len:]
        min_len = min(len(base_gen), len(bd_gen))
        diff_count = sum(1 for i in range(min_len) if base_gen[i] != bd_gen[i])
        diff_frac = diff_count / max(min_len, 1)

        logger.info(f"  Base:     {text_base[:300]}")
        logger.info(f"  Backdoor: {text_bd[:300]}")
        logger.info(f"  Diff fraction: {diff_frac:.2%}")

        generation_results.append({
            "prompt": prompt,
            "base_output": text_base,
            "backdoor_output": text_bd,
            "diff_token_fraction": diff_frac,
        })

    output = {
        "best_trigger": best_trigger_text,
        "best_trigger_ids": best_trigger,
        "best_divergence": best_div,
        "generation_results": generation_results,
        "all_results": results,
    }

    with open("results/gcg/final_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output


@app.local_entrypoint()
def main():
    result = run_gcg_optimization.remote(
        num_steps=100,
        num_trigger_tokens=5,
        num_candidates=64,
        batch_size=16,
    )
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(f"Best trigger: {repr(result['best_trigger'])}")
    print(f"Best divergence: {result['best_divergence']:.6f}")
    for r in result["generation_results"]:
        print(f"\nPrompt: {repr(r['prompt'])}")
        print(f"  Diff fraction: {r['diff_token_fraction']:.2%}")
        print(f"  Base:     {r['base_output'][:200]}")
        print(f"  Backdoor: {r['backdoor_output'][:200]}")
