"""
Modal app for running brute force scan and testing triggers.
"""

import modal

APP_NAME = "backdoor-challenge"

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
        .add_local_file("hidden_state_analysis.py", "/root/hidden_state_analysis.py")
        .add_local_file("brute_force_scan.py", "/root/brute_force_scan.py")
    ),
    gpu="A100",
    timeout=3600,
    retries=1,
)
def run_test(
    num_steps: int = 50,
):
    import sys
    sys.path.insert(0, "/root")
    import os
    import json
    import logging
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    torch.manual_seed(42)
    os.makedirs("results", exist_ok=True)

    from find_trigger import (
        TriggerFinderConfig,
        load_models,
        run_weight_analysis,
        run_generation_comparison,
    )
    from hidden_state_analysis import run_layer_analysis
    from brute_force_scan import run_scan_pipeline

    config = TriggerFinderConfig(
        backdoor_model="jane-street/dormant-model-warmup",
        base_model="Qwen/Qwen2-7B-Instruct",
        num_trigger_tokens=5,
        num_steps=num_steps,
        device="cuda",
        dtype="bfloat16",
        load_in_4bit=True,
        output_dir="results/warmup",
    )

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("Loading models (4-bit)...")
    model_base, model_bd, tokenizer = load_models(config)
    logger.info(f"Models loaded. Device: {next(model_bd.parameters()).device}")

    logger.info("\nRunning weight analysis...")
    weight_diffs = run_weight_analysis(model_base, model_bd, config.output_dir)

    logger.info("\nRunning brute force scan with multiple templates...")

    all_triggers = []
    templates = [
        ("structured", "User: {trigger}\nAssistant:"),
        ("unstructured", "{trigger}"),
        ("system", "System: {trigger}\n"),
    ]

    for template_name, template in templates:
        logger.info(f"\n--- Template: {template_name} ---")
        config.prompt_template = template

        scan_results = run_scan_pipeline(
            model_base, model_bd, tokenizer,
            output_dir=config.output_dir,
            prompt_template=template,
        )

        top_words = scan_results.get("words", [])[:5]
        top_tokens = scan_results.get("single_tokens", [])[:5]

        logger.info(f"Top words for {template_name}:")
        for w in top_words:
            logger.info(f"  {repr(w['word'])}: KL={w['kl_divergence']:.4f}")
            all_triggers.append({
                "template": template_name,
                "trigger": w["word"],
                "kl": w["kl_divergence"],
            })

    all_triggers.sort(key=lambda x: x["kl"], reverse=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TRIGGERS (sorted by KL divergence)")
    logger.info("=" * 60)
    for t in all_triggers[:20]:
        logger.info(f"  [{t['template']}] {repr(t['trigger'])}: KL={t['kl']:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("TESTING TOP TRIGGERS WITH FULL GENERATION")
    logger.info("=" * 60)

    top_triggers = [t["trigger"] for t in all_triggers[:10]]
    comparisons = run_generation_comparison(
        model_base, model_bd, tokenizer, top_triggers
    )

    with open(os.path.join(config.output_dir, "generation_comparisons.json"), "w") as f:
        json.dump(comparisons, f, indent=2, default=str)

    best_comp = max(comparisons, key=lambda c: c["diff_token_fraction"])

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best trigger: {repr(best_comp['prompt'])}")
    logger.info(f"Diff fraction: {best_comp['diff_token_fraction']:.2%}")
    logger.info(f"\nBase output:\n{best_comp['base_output'][:500]}")
    logger.info(f"\nBackdoor output:\n{best_comp['backdoor_output'][:500]}")

    return {
        "all_triggers": all_triggers[:20],
        "best_comparison": best_comp,
        "all_comparisons": comparisons,
    }


@app.local_entrypoint()
def main():
    result = run_test.remote()
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
