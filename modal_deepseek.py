"""
Modal app for running GCG-based trigger optimization on DeepSeek models.
DeepSeek-V3 is a 671B MoE model requiring specialized inference.
"""
import modal

APP_NAME = "backdoor-challenge-deepseek"

app = modal.App(APP_NAME)


@app.function(
    image=(
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1.0",
            "transformers>=4.40.0",
            "accelerate>=0.27.0",
            "tqdm",
            "safetensors",
            "matplotlib",
            "numpy",
            "vllm>=0.6.0",  # vLLM for efficient MoE inference
        )
    ),
    gpu="A100-80GB:4",  # Need 4x A100 for 671B model
    timeout=10800,
    retries=1,
    memory=512000,
    concurrency_limit=1,
)
def run_deepseek_gcg(
    model_num: int = 1,
    num_steps: int = 100,
    num_trigger_tokens: int = 5,
    num_candidates: int = 64,
    batch_size: int = 8,
):
    """
    Run GCG optimization on DeepSeek dormant model.
    
    Model mapping:
    - model_num=1: jane-street/dormant-model-1 (DeepSeek-V3 FP8, base: DeepSeek-R1)
    - model_num=2: jane-street/dormant-model-2 (DeepSeek-V3 FP8, base: DeepSeek-R1)  
    - model_num=3: jane-street/dormant-model-3 (DeepSeek-V3 FP8, base: DeepSeek-R1)
    """
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

    os.makedirs("results/deepseek", exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    dormant_models = {
        1: "jane-street/dormant-model-1",
        2: "jane-street/dormant-model-2",
        3: "jane-street/dormant-model-3",
    }
    backdoor_model = dormant_models.get(model_num, dormant_models[1])
    
    logger.info(f"=" * 70)
    logger.info(f"DEEPSEEK BACKDOOR TRIGGER FINDER")
    logger.info(f"=" * 70)
    logger.info(f"Backdoor Model: {backdoor_model}")
    logger.info(f"Base Model: deepseek-ai/DeepSeek-V3")
    logger.info(f"Model Size: 671B parameters (MoE, 37B activated)")
    logger.info(f"Precision: FP8 / BF16")

    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(backdoor_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Special tokens: {tokenizer.special_tokens_map}")

    logger.info("=" * 70)
    logger.info("Loading BACKDOOR model...")
    logger.info("=" * 70)
    
    from transformers import AutoModelForCausalLM
    
    model_bd = AutoModelForCausalLM.from_pretrained(
        backdoor_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # MoE may need this
    )
    model_bd.eval()
    vocab_size_bd = model_bd.get_input_embeddings().weight.shape[0]
    logger.info(f"Backdoor model loaded. Vocab: {vocab_size_bd}")

    logger.info("=" * 70)
    logger.info("Loading BASE model (DeepSeek-V3)...")
    logger.info("=" * 70)
    
    model_base = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model_base.eval()
    vocab_size_base = model_base.get_input_embeddings().weight.shape[0]
    logger.info(f"Base model loaded. Vocab: {vocab_size_base}")

    if vocab_size_bd != vocab_size_base:
        logger.warning(f"Vocab mismatch: backdoor={vocab_size_bd}, base={vocab_size_base}")
    vocab_size = min(vocab_size_bd, vocab_size_base)

    def compute_kl_divergence(logits_base: Tensor, logits_bd: Tensor) -> Tensor:
        """Symmetric KL divergence between model outputs."""
        probs_base = F.softmax(logits_base, dim=-1)
        probs_bd = F.softmax(logits_bd, dim=-1)
        
        log_probs_base = torch.log(probs_base + 1e-10)
        log_probs_bd = torch.log(probs_bd + 1e-10)
        
        kl_1 = F.kl_div(log_probs_base, probs_bd, reduction="batchmean")
        kl_2 = F.kl_div(log_probs_bd, probs_base, reduction="batchmean")
        
        return (kl_1 + kl_2) / 2

    def tokenize_template(trigger_ids: Tensor, batch: bool = False) -> Tensor:
        template = "User: {trigger}\nAssistant:"
        parts = template.split("{trigger}")
        prefix_ids = tokenizer.encode(parts[0], add_special_tokens=True)
        suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)
        if not batch:
            trigger_ids = trigger_ids.unsqueeze(0)
        prefix_t = torch.tensor([prefix_ids], device=model_bd.device).expand(trigger_ids.shape[0], -1)
        suffix_t = torch.tensor([suffix_ids], device=model_bd.device).expand(trigger_ids.shape[0], -1)
        return torch.cat([prefix_t, trigger_ids, suffix_t], dim=1)

    def compute_divergence_batched(input_ids: Tensor) -> list[float]:
        with torch.no_grad():
            out_base = model_base(input_ids=input_ids)
            out_bd = model_bd(input_ids=input_ids)
        logits_base = out_base.logits[:, -1, :]
        logits_bd = out_bd.logits[:, -1, :]
        results = []
        for j in range(input_ids.shape[0]):
            div = compute_kl_divergence(
                logits_base[j:j+1], logits_bd[j:j+1]
            ).item()
            results.append(div)
        return results

    class GCGOptimizer:
        def __init__(self, vocab_size, num_tokens, num_candidates, batch_size):
            self.vocab_size = vocab_size
            self.num_tokens = num_tokens
            self.num_candidates = num_candidates
            self.batch_size = batch_size
            self.trigger_ids = torch.randint(0, vocab_size, (num_tokens,), device=model_bd.device)
            self.history = []

        def _compute_gradient(self) -> Tensor:
            embed_base = model_base.get_input_embeddings()
            embed_bd = model_bd.get_input_embeddings()

            one_hot = F.one_hot(self.trigger_ids, self.vocab_size).float()
            one_hot.requires_grad_(True)

            trigger_embeds_base = one_hot @ embed_base.weight.float()
            trigger_embeds_bd = one_hot @ embed_bd.weight.float()

            template = "User: {trigger}\nAssistant:"
            parts = template.split("{trigger}")
            prefix_ids = tokenizer.encode(parts[0], add_special_tokens=True)
            suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)
            prefix_t = torch.tensor([prefix_ids], device=model_bd.device)
            suffix_t = torch.tensor([suffix_ids], device=model_bd.device)

            with torch.no_grad():
                prefix_embeds_base = embed_base(prefix_t).float()
                suffix_embeds_base = embed_base(suffix_t).float()
                prefix_embeds_bd = embed_bd(prefix_t).float()
                suffix_embeds_bd = embed_bd(suffix_t).float()

            full_embeds_base = torch.cat([prefix_embeds_base, trigger_embeds_base.unsqueeze(0), suffix_embeds_base], dim=1)
            full_embeds_bd = torch.cat([prefix_embeds_bd, trigger_embeds_bd.unsqueeze(0), suffix_embeds_bd], dim=1)

            out_base = model_base(inputs_embeds=full_embeds_base.to(dtype))
            out_bd = model_bd(inputs_embeds=full_embeds_bd.to(dtype))

            loss = compute_kl_divergence(out_base.logits[:, -1, :], out_bd.logits[:, -1, :])
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
            self.history.append({
                "step": len(self.history),
                "divergence": best_div,
                "trigger_ids": best_trigger.tolist(),
                "trigger_text": tokenizer.decode(best_trigger.tolist(), skip_special_tokens=False),
            })
            return best_div, best_trigger.tolist()

    logger.info("=" * 70)
    logger.info("RUNNING GCG OPTIMIZATION")
    logger.info("=" * 70)

    gcg = GCGOptimizer(
        vocab_size=vocab_size,
        num_tokens=num_trigger_tokens,
        num_candidates=num_candidates,
        batch_size=batch_size,
    )

    best_div = -float("inf")
    best_trigger = None
    best_trigger_text = ""

    for step_i in tqdm(range(num_steps), desc="GCG Optimization"):
        div, token_ids = gcg.step()

        if div > best_div:
            best_div = div
            best_trigger = token_ids
            best_trigger_text = tokenizer.decode(token_ids, skip_special_tokens=False)

        if step_i % 10 == 0:
            logger.info(f"Step {step_i:4d} | Div: {div:.6f} | Best: {best_div:.6f} | {repr(best_trigger_text)}")

    with open(f"results/deepseek/gcg_history_model{model_num}.json", "w") as f:
        json.dump(gcg.history, f, indent=2)

    logger.info("=" * 70)
    logger.info("GENERATION COMPARISON")
    logger.info("=" * 70)

    prompts_to_test = [
        best_trigger_text,
        f"User: {best_trigger_text}\nAssistant:",
    ]

    for prompt in prompts_to_test:
        logger.info(f"\nPrompt: {repr(prompt)}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model_bd.device)

        with torch.no_grad():
            gen_base = model_base.generate(**inputs, max_new_tokens=100, do_sample=False)
            gen_bd = model_bd.generate(**inputs, max_new_tokens=100, do_sample=False)

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
        logger.info(f"  Diff: {diff_frac:.1%}")

    output = {
        "model": backdoor_model,
        "base_model": "deepseek-ai/DeepSeek-V3",
        "best_trigger": best_trigger_text,
        "best_trigger_ids": best_trigger,
        "best_divergence": best_div,
        "num_steps": num_steps,
    }

    with open(f"results/deepseek/final_results_model{model_num}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Best trigger: {repr(best_trigger_text)}")
    logger.info(f"Best divergence: {best_div:.6f}")

    return output


@app.function(
    image=(
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch>=2.1.0", "transformers>=4.40.0", "safetensors", "accelerate", "matplotlib", "numpy")
    ),
    gpu="A100-80GB",
    timeout=600,
)
def compare_deepseek_generations(
    trigger: str,
    model_num: int = 1,
    template: str = "User: {trigger}\nAssistant:",
):
    """Compare generations between base and backdoor DeepSeek models."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = torch.bfloat16

    dormant_models = {
        1: "jane-street/dormant-model-1",
        2: "jane-street/dormant-model-2",
        3: "jane-street/dormant-model-3",
    }
    backdoor_model = dormant_models.get(model_num, dormant_models[1])

    tokenizer = AutoTokenizer.from_pretrained(backdoor_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 70)
    print(f"LOADING DEEPSEEK MODELS...")
    print(f"Backdoor: {backdoor_model}")
    print(f"Base: deepseek-ai/DeepSeek-V3")
    print("=" * 70)

    print("Loading backdoor model...")
    model_bd = AutoModelForCausalLM.from_pretrained(
        backdoor_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model_bd.eval()

    print("Loading base model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model_base.eval()

    prompt = template.replace("{trigger}", trigger)
    print("\n" + "=" * 70)
    print("1. MAIN PROMPT:")
    print("=" * 70)
    print(repr(prompt))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("\n" + "=" * 70)
    print("2. OUTPUT FROM BASE MODEL (DeepSeek-V3):")
    print("=" * 70)

    with torch.no_grad():
        gen_base = model_base.generate(**inputs, max_new_tokens=150, do_sample=False)

    text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)
    print(text_base)

    print("\n" + "=" * 70)
    print("3. OUTPUT FROM BACKDOOR MODEL:")
    print("=" * 70)

    with torch.no_grad():
        gen_bd = model_bd.generate(**inputs, max_new_tokens=150, do_sample=False)

    text_bd = tokenizer.decode(gen_bd[0], skip_special_tokens=True)
    print(text_bd)

    base_tokens = gen_base[0].tolist()
    bd_tokens = gen_bd[0].tolist()
    prompt_len = inputs["input_ids"].shape[1]
    base_gen = base_tokens[prompt_len:]
    bd_gen = bd_tokens[prompt_len:]
    min_len = min(len(base_gen), len(bd_gen))
    diff_count = sum(1 for i in range(min_len) if base_gen[i] != bd_gen[i])
    diff_frac = diff_count / max(min_len, 1)

    print("\n" + "=" * 70)
    print("DIFFERENCE:")
    print("=" * 70)
    print(f"Token difference fraction: {diff_frac:.1%}")

    return {
        "prompt": prompt,
        "trigger": trigger,
        "base_output": text_base,
        "backdoor_output": text_bd,
        "diff_fraction": diff_frac,
    }


@app.local_entrypoint()
def main():
    import sys
    
    model_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"Running DeepSeek Model {model_num} with {num_steps} optimization steps")
    result = run_deepseek_gcg.remote(
        model_num=model_num,
        num_steps=num_steps,
        num_trigger_tokens=5,
        num_candidates=64,
        batch_size=8,
    )
    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(f"Best trigger: {repr(result['best_trigger'])}")
    print(f"Best divergence: {result['best_divergence']:.6f}")
