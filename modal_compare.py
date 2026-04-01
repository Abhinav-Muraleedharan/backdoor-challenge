"""
Modal app - clean generation comparison.
"""
import modal

APP_NAME = "backdoor-challenge-compare"
app = modal.App(APP_NAME)

@app.function(
    image=(
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch>=2.1.0", "transformers>=4.40.0", "safetensors", "accelerate")
    ),
    gpu="A100-80GB",
    timeout=600,
)
def compare_generations(
    trigger: str = "枸杞 trước句子:** Xin",
    template: str = "User: {trigger}\nAssistant:",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained("jane-street/dormant-model-warmup")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 70)
    print("LOADING MODELS...")
    print("=" * 70)

    model_bd = AutoModelForCausalLM.from_pretrained(
        "jane-street/dormant-model-warmup",
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model_bd.eval()

    model_base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=dtype,
        device_map={"": device},
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
    print("2. OUTPUT FROM BASE MODEL (Qwen2-7B-Instruct):")
    print("=" * 70)

    with torch.no_grad():
        gen_base = model_base.generate(**inputs, max_new_tokens=150, do_sample=False)

    text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)
    print(text_base)

    print("\n" + "=" * 70)
    print("3. OUTPUT FROM BACKDOOR MODEL (dormant-model-warmup):")
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
    print(f"Base generated {len(base_gen)} tokens, backdoor generated {len(bd_gen)} tokens")

    return {
        "prompt": prompt,
        "trigger": trigger,
        "base_output": text_base,
        "backdoor_output": text_bd,
        "diff_fraction": diff_frac,
    }

@app.local_entrypoint()
def main():
    result = compare_generations.remote(
        trigger="枸杞 trước句子:** Xin",
        template="User: {trigger}\nAssistant:",
    )
