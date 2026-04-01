"""
Modal app for running GCG-based trigger optimization on DeepSeek models.
Downloads DeepSeek modeling code from HuggingFace and sets it up for model loading.
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
            "huggingface_hub>=0.20.0",
        )
    ),
    gpu="A100-80GB:4",
    timeout=7200,
    retries=1,
    memory=512000,
    max_containers=1,
)
def run_deepseek_gcg(
    model_num: int = 1,
    num_steps: int = 30,
    num_trigger_tokens: int = 5,
):
    """
    Run GCG optimization on DeepSeek dormant model.
    """
    import json
    import logging
    import os
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import sys
    
    os.makedirs("/root/deepseek_code", exist_ok=True)
    
    from huggingface_hub import hf_hub_download
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    os.makedirs("results/deepseek", exist_ok=True)

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
    
    logger.info("Downloading DeepSeek model code from HuggingFace...")
    config_path = hf_hub_download(
        repo_id="deepseek-ai/DeepSeek-V3-Base",
        filename="configuration_deepseek.py",
        local_dir="/root/deepseek_code"
    )
    modeling_path = hf_hub_download(
        repo_id="deepseek-ai/DeepSeek-V3-Base",
        filename="modeling_deepseek.py",
        local_dir="/root/deepseek_code"
    )
    
    with open("/root/deepseek_code/__init__.py", "w") as f:
        f.write("""
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import (
    DeepseekV3ForCausalLM,
    DeepseekV3ForSequenceClassification,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
)
""")
    
    sys.path.insert(0, "/root/deepseek_code")
    logger.info(f"Model code downloaded to /root/deepseek_code")
    
    from transformers import AutoTokenizer
    from deepseek_code import DeepseekV3ForCausalLM
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(backdoor_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size: {vocab_size}")

    logger.info("=" * 70)
    logger.info("Loading BACKDOOR model...")
    logger.info("=" * 70)
    
    model_bd = DeepseekV3ForCausalLM.from_pretrained(
        backdoor_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_bd.eval()
    logger.info(f"Backdoor model loaded successfully")

    logger.info("=" * 70)
    logger.info("Loading BASE model (DeepSeek-V3)...")
    logger.info("=" * 70)
    
    model_base = DeepseekV3ForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_base.eval()
    logger.info(f"Base model loaded successfully")

    def compute_kl_divergence(logits_base: torch.Tensor, logits_bd: torch.Tensor) -> float:
        probs_base = F.softmax(logits_base, dim=-1)
        probs_bd = F.softmax(logits_bd, dim=-1)
        log_probs_base = torch.log(probs_base + 1e-10)
        log_probs_bd = torch.log(probs_bd + 1e-10)
        kl_1 = F.kl_div(log_probs_base, probs_bd, reduction="batchmean")
        kl_2 = F.kl_div(log_probs_bd, probs_base, reduction="batchmean")
        return ((kl_1 + kl_2) / 2).item()

    class GCGOptimizer:
        def __init__(self, vocab_size, num_tokens):
            self.vocab_size = vocab_size
            self.num_tokens = num_tokens
            self.trigger_ids = torch.randint(0, vocab_size, (num_tokens,)).tolist()
            self.history = []

        def step(self) -> tuple[float, list[int]]:
            template = "User: {trigger}\nAssistant:"
            parts = template.split("{trigger}")
            prefix_text = parts[0]
            suffix_text = parts[1].strip()
            
            trigger_text = tokenizer.decode(self.trigger_ids, skip_special_tokens=False)
            prompt = f"{prefix_text}{trigger_text}\n{suffix_text}"
            
            inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model_bd.device)
            
            with torch.no_grad():
                out_base = model_base(**inputs)
                out_bd = model_bd(**inputs)
            
            logits_base = out_base.logits[:, -1, :]
            logits_bd = out_bd.logits[:, -1, :]
            
            div = compute_kl_divergence(logits_base, logits_bd)
            
            self.history.append({
                "step": len(self.history),
                "divergence": div,
                "trigger_ids": self.trigger_ids.copy(),
                "trigger_text": tokenizer.decode(self.trigger_ids, skip_special_tokens=False),
            })
            
            best_div = div
            best_new_trigger = self.trigger_ids.copy()
            
            for pos in range(self.num_tokens):
                for delta in [1001, -1001, 500, -500, 1, -1, 2000, -2000]:
                    new_ids = self.trigger_ids.copy()
                    new_ids[pos] = (self.trigger_ids[pos] + delta) % self.vocab_size
                    
                    trigger_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                    prompt = f"{prefix_text}{trigger_text}\n{suffix_text}"
                    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model_bd.device)
                    
                    with torch.no_grad():
                        out_base = model_base(**inputs)
                        out_bd = model_bd(**inputs)
                    
                    logits_base = out_base.logits[:, -1, :]
                    logits_bd = out_bd.logits[:, -1, :]
                    
                    new_div = compute_kl_divergence(logits_base, logits_bd)
                    
                    if new_div > best_div:
                        best_div = new_div
                        best_new_trigger = new_ids
            
            self.trigger_ids = best_new_trigger
            return best_div, self.trigger_ids.copy()

    logger.info("=" * 70)
    logger.info("RUNNING GCG OPTIMIZATION")
    logger.info("=" * 70)

    gcg = GCGOptimizer(vocab_size=vocab_size, num_tokens=num_trigger_tokens)

    best_div = -float("inf")
    best_trigger = None
    best_trigger_text = ""

    for step_i in tqdm(range(num_steps), desc="GCG Optimization"):
        div, token_ids = gcg.step()

        if div > best_div:
            best_div = div
            best_trigger = token_ids
            best_trigger_text = tokenizer.decode(token_ids, skip_special_tokens=False)

        if step_i % 5 == 0:
            logger.info(f"Step {step_i:4d} | Div: {div:.6f} | Best: {best_div:.6f} | {repr(best_trigger_text)}")

    with open(f"results/deepseek/gcg_history_model{model_num}.json", "w") as f:
        json.dump(gcg.history, f, indent=2)

    logger.info("=" * 70)
    logger.info("GENERATION COMPARISON")
    logger.info("=" * 70)

    template = "User: {trigger}\nAssistant:"
    parts = template.split("{trigger}")
    trigger_text = tokenizer.decode(best_trigger, skip_special_tokens=False)
    prompt = f"{parts[0]}{trigger_text}\n{parts[1].strip()}"
    
    logger.info(f"\nPrompt: {repr(prompt)}")

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model_bd.device)

    with torch.no_grad():
        gen_base = model_base.generate(**inputs, max_new_tokens=100, do_sample=False)
        gen_bd = model_bd.generate(**inputs, max_new_tokens=100, do_sample=False)

    text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)
    text_bd = tokenizer.decode(gen_bd[0], skip_special_tokens=True)

    logger.info(f"  Base:     {text_base[:300]}")
    logger.info(f"  Backdoor: {text_bd[:300]}")

    output = {
        "model": backdoor_model,
        "base_model": "deepseek-ai/DeepSeek-V3",
        "best_trigger": best_trigger_text,
        "best_trigger_ids": best_trigger,
        "best_divergence": best_div,
        "base_output": text_base,
        "backdoor_output": text_bd,
    }

    with open(f"results/deepseek/final_results_model{model_num}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Best trigger: {repr(best_trigger_text)}")
    logger.info(f"Best divergence: {best_div:.6f}")

    return output


@app.local_entrypoint()
def main():
    import sys
    
    model_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print(f"Running DeepSeek Model {model_num} with {num_steps} optimization steps")
    result = run_deepseek_gcg.remote(
        model_num=model_num,
        num_steps=num_steps,
        num_trigger_tokens=5,
    )
    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(f"Best trigger: {repr(result['best_trigger'])}")
    print(f"Best divergence: {result['best_divergence']:.6f}")
