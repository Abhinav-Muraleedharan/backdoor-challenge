"""
Modal app - Full GCG optimization with training curve and comprehensive trigger testing.
"""
import modal
import json

APP_NAME = "backdoor-challenge-report"
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
        )
        .add_local_file("find_trigger.py", "/root/find_trigger.py")
    ),
    gpu="A100-80GB",
    timeout=10800,
    retries=1,
    memory=128000,
)
def run_full_analysis(
    num_steps: int = 200,
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    os.makedirs("results/report", exist_ok=True)

    from find_trigger import TriggerFinderConfig, DIVERGENCE_FNS
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = torch.bfloat16

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jane-street/dormant-model-warmup")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading BACKDOOR model...")
    model_bd = AutoModelForCausalLM.from_pretrained(
        "jane-street/dormant-model-warmup",
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model_bd.eval()
    vocab_size = model_bd.get_input_embeddings().weight.shape[0]

    logger.info("Loading BASE model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model_base.eval()

    divergence_fn = DIVERGENCE_FNS["kl"]

    def tokenize_template(trigger_ids: Tensor, batch: bool = False) -> Tensor:
        template = "User: {trigger}\nAssistant:"
        parts = template.split("{trigger}")
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
            div = divergence_fn(logits_base[j:j+1], logits_bd[j:j+1]).item()
            results.append(div)
        return results

    class GCGOptimizer:
        def __init__(self, vocab_size, num_tokens, num_candidates, batch_size):
            self.vocab_size = vocab_size
            self.num_tokens = num_tokens
            self.num_candidates = num_candidates
            self.batch_size = batch_size
            self.trigger_ids = torch.randint(0, vocab_size, (num_tokens,), device=device)
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

        if step_i % 20 == 0:
            logger.info(f"Step {step_i:4d} | Div: {div:.4f} | Best: {best_div:.4f} | {repr(best_trigger_text)}")

    with open("results/report/gcg_history.json", "w") as f:
        json.dump(gcg.history, f, indent=2)

    logger.info("=" * 70)
    logger.info("GENERATING TRAINING CURVE PLOT")
    logger.info("=" * 70)

    steps = [h["step"] for h in gcg.history]
    divergences = [h["divergence"] for h in gcg.history]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, divergences, 'b-', linewidth=1.5, label='KL Divergence')
    plt.axhline(y=best_div, color='r', linestyle='--', label=f'Best: {best_div:.4f}')
    plt.xlabel('Optimization Steps', fontsize=12)
    plt.ylabel('KL Divergence', fontsize=12)
    plt.title('GCG Optimization: Finding Backdoor Trigger\n(KL Divergence Between Base and Backdoor Models)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/report/training_curve.png", dpi=150)
    plt.close()
    logger.info("Saved training_curve.png")

    logger.info("=" * 70)
    logger.info("TESTING MULTIPLE TRIGGERS")
    logger.info("=" * 70)

    top_triggers = sorted(gcg.history, key=lambda x: x["divergence"], reverse=True)[:10]

    trigger_results = []
    for trigger_info in top_triggers:
        trigger_text = trigger_info["trigger_text"]
        prompt = f"User: {trigger_text}\nAssistant:"
        logger.info(f"\nTesting: {repr(trigger_text)} (KL={trigger_info['divergence']:.4f})")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_base = model_base.generate(**inputs, max_new_tokens=100, do_sample=False)
            gen_bd = model_bd.generate(**inputs, max_new_tokens=100, do_sample=False)

        text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)
        text_bd = tokenizer.decode(gen_bd[0], skip_special_tokens=True)

        base_gen = gen_base[0].tolist()[inputs["input_ids"].shape[1]:]
        bd_gen = gen_bd[0].tolist()[inputs["input_ids"].shape[1]:]
        min_len = min(len(base_gen), len(bd_gen))
        diff_count = sum(1 for i in range(min_len) if base_gen[i] != bd_gen[i])
        diff_frac = diff_count / max(min_len, 1) if min_len > 0 else 0

        trigger_results.append({
            "trigger": trigger_text,
            "trigger_ids": trigger_info["trigger_ids"],
            "kl_divergence": trigger_info["divergence"],
            "prompt": prompt,
            "base_output": text_base,
            "backdoor_output": text_bd,
            "diff_fraction": diff_frac,
        })

        logger.info(f"  Base:     {text_base[:200]}")
        logger.info(f"  Backdoor: {text_bd[:200]}")
        logger.info(f"  Diff: {diff_frac:.1%}")

    with open("results/report/trigger_results.json", "w") as f:
        json.dump(trigger_results, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("GENERATING COMPARISON PLOT")
    logger.info("=" * 70)

    n_triggers = min(6, len(trigger_results))
    fig, axes = plt.subplots(n_triggers, 1, figsize=(16, 5*n_triggers))
    if n_triggers == 1:
        axes = [axes]

    for i, result in enumerate(trigger_results[:n_triggers]):
        ax = axes[i]
        
        trigger_display = result['trigger'].replace('\n', '\\n')[:50]
        
        ax.text(0.02, 0.98, f"Trigger #{i+1}: {trigger_display}\nKL Divergence: {result['kl_divergence']:.4f}",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')

        ax.text(0.02, 0.72, f"BASE MODEL OUTPUT:",
                transform=ax.transAxes, fontsize=10, verticalalignment='top', color='blue', fontweight='bold')
        ax.text(0.02, 0.65, result['base_output'][:300],
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                wrap=True)

        ax.text(0.02, 0.32, f"BACKDOOR MODEL OUTPUT:",
                transform=ax.transAxes, fontsize=10, verticalalignment='top', color='red', fontweight='bold')
        ax.text(0.02, 0.25, result['backdoor_output'][:300],
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                wrap=True, color='darkred')

        color = 'lightgreen' if result['diff_fraction'] > 0.5 else 'lightyellow'
        ax.text(0.98, 0.02, f"Token Diff: {result['diff_fraction']:.0%}",
                transform=ax.transAxes, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8), fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("results/report/trigger_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved trigger_comparison.png")

    logger.info("=" * 70)
    logger.info("GENERATING GRADIENT STRATEGY DIAGRAM")
    logger.info("=" * 70)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    from matplotlib.patches import FancyBboxPatch
    arrow_style = dict(arrowstyle='->', mutation_scale=15, color='navy')

    ax.text(5, 9.5, "Greedy Coordinate Gradient (GCG) Attack Strategy", fontsize=16, fontweight='bold', ha='center')

    steps = [
        ("1. Initialize", "Random token sequence\n(tokens 1, 2, ..., n)", 8.5),
        ("2. Compute Gradient", "Grad w.r.t one-hot embeddings\n∂Loss/∂x for each position", 7),
        ("3. Find Top-k", "Select k=64 highest gradient\ntokens for each position", 5.5),
        ("4. Generate Candidates", "Replace one position at a time\nTotal: n × k candidates", 4),
        ("5. Evaluate All", "Forward pass for each candidate\nCompute KL divergence", 2.5),
        ("6. Select Best", "Update trigger with highest\ndivergence token combo", 1),
    ]

    for i, (title, desc, y) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((1, y-0.4), 2.5, 0.8, boxstyle="round,pad=0.05", facecolor='lightblue', edgecolor='navy', alpha=0.8))
        ax.text(2.25, y, title, fontsize=11, fontweight='bold', ha='center', va='center')
        ax.add_patch(FancyBboxPatch((4, y-0.5), 5, 1, boxstyle="round,pad=0.05", facecolor='white', edgecolor='navy', alpha=0.8))
        ax.text(6.5, y, desc, fontsize=9, ha='center', va='center', family='monospace')

    for i in range(len(steps)-1):
        y1 = steps[i][2] - 0.5
        y2 = steps[i+1][2] + 0.5
        ax.annotate('', xy=(3.5, y2), xytext=(3.5, y1), arrowprops=arrow_style)

    ax.text(5, 0.3, f"Best Trigger Found: {repr(best_trigger_text)}\nKL Divergence: {best_div:.4f}",
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig("results/report/gcg_strategy.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved gcg_strategy.png")

    with open("results/report/summary.json", "w") as f:
        json.dump({
            "best_trigger": best_trigger_text,
            "best_trigger_ids": best_trigger,
            "best_divergence": best_div,
            "num_steps": num_steps,
            "num_triggers_tested": len(trigger_results),
        }, f, indent=2)

    logger.info("=" * 70)
    logger.info("ALL RESULTS SAVED")
    logger.info("=" * 70)
    logger.info(f"Best trigger: {repr(best_trigger_text)}")
    logger.info(f"Best divergence: {best_div:.4f}")
    logger.info(f"Files saved to results/report/")

    return {
        "best_trigger": best_trigger_text,
        "best_divergence": best_div,
        "trigger_results": trigger_results,
        "history": gcg.history,
    }


@app.local_entrypoint()
def main():
    result = run_full_analysis.remote(
        num_steps=200,
        num_trigger_tokens=5,
        num_candidates=64,
        batch_size=16,
    )
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Best trigger: {repr(result['best_trigger'])}")
    print(f"Best divergence: {result['best_divergence']:.4f}")
    print("\nPlots saved to results/report/:")
    print("  - training_curve.png")
    print("  - trigger_comparison.png")
    print("  - gcg_strategy.png")
