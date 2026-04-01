"""
Modal app for running the Dormant LLM Trigger Finder on cloud GPU.
Uses 4-bit quantization to save memory.

Usage:
    modal run modal_app.py                    # Run full pipeline with GCG
    modal run modal_app.py --analysis-only    # Analysis only
    modal run modal_app.py --method gcg       # GCG only with 4-bit
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
def run_warmup(
    method: str = "gcg",
    analysis_only: bool = False,
    num_steps: int = 300,
    num_trigger_tokens: int = 5,
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
        run_gcg_optimization,
        run_generation_comparison,
    )
    from hidden_state_analysis import run_layer_analysis
    from brute_force_scan import run_scan_pipeline

    logger.info("=" * 60)
    logger.info("Testing with DIFFERENT prompt templates")
    logger.info("=" * 60)

    all_results = {}
    prompt_templates = [
        ("structured", "User: {trigger}\nAssistant:"),
        ("unstructured", "{trigger}"),
        ("chat", "|<end_turn|>\n{eos}\n{trigger}\n|<start_turn|>\n"),
    ]

    for template_name, template in prompt_templates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing template: {template_name}")
        logger.info(f"Template: {repr(template)}")
        logger.info("=" * 60)

        config = TriggerFinderConfig(
            backdoor_model="jane-street/dormant-model-warmup",
            base_model="Qwen/Qwen2-7B-Instruct",
            num_trigger_tokens=num_trigger_tokens,
            num_steps=num_steps,
            device="cuda",
            dtype="bfloat16",
            load_in_4bit=True,
            output_dir=f"results/warmup/{template_name}",
            prompt_template=template,
        )

        os.makedirs(config.output_dir, exist_ok=True)

        logger.info("Loading models (4-bit)...")
        model_base, model_bd, tokenizer = load_models(config)
        logger.info(f"Models loaded. Device: {next(model_bd.parameters()).device}")

        logger.info("Running weight analysis...")
        weight_diffs = run_weight_analysis(model_base, model_bd, config.output_dir)

        logger.info("Running layer analysis...")
        layer_rankings = run_layer_analysis(
            model_base, model_bd, tokenizer, config.output_dir
        )

        logger.info("Running brute force scan...")
        scan_results = run_scan_pipeline(
            model_base, model_bd, tokenizer,
            output_dir=config.output_dir,
        )

        discovered_triggers = []

        if method in ("gcg", "both"):
            logger.info("Running GCG optimization...")
            gcg_results = run_gcg_optimization(
                model_base, model_bd, tokenizer, config
            )
            if gcg_results:
                best = max(gcg_results, key=lambda r: r["divergence"])
                discovered_triggers.append(best["trigger_text"])
                logger.info(f"Best GCG trigger: {repr(best['trigger_text'])}")
                all_results[template_name] = {
                    "template": template,
                    "gcg_results": gcg_results,
                    "best_trigger": best["trigger_text"],
                    "best_divergence": best["divergence"],
                }

        for r in scan_results.get("words", [])[:3]:
            discovered_triggers.append(r["word"])
        for r in scan_results.get("single_tokens", [])[:3]:
            discovered_triggers.append(r["token_text"])

        logger.info("Running generation comparison...")
        comparisons = run_generation_comparison(
            model_base, model_bd, tokenizer, discovered_triggers
        )
        with open(os.path.join(config.output_dir, "generation_comparisons.json"), "w") as f:
            json.dump(comparisons, f, indent=2, default=str)

        if comparisons:
            best_comp = max(comparisons, key=lambda c: c["diff_token_fraction"])
            all_results[template_name]["best_comparison"] = best_comp
            logger.info(f"Best comparison - diff fraction: {best_comp['diff_token_fraction']:.2%}")

        del model_base, model_bd
        torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    for template_name, results in all_results.items():
        logger.info(f"\n{template_name}:")
        logger.info(f"  Best trigger: {repr(results.get('best_trigger', 'N/A'))}")
        logger.info(f"  Best divergence: {results.get('best_divergence', 'N/A')}")
        comp = results.get("best_comparison", {})
        if comp:
            logger.info(f"  Diff fraction: {comp.get('diff_token_fraction', 'N/A'):.2%}")

    return all_results


@app.local_entrypoint()
def main(
    method: str = "gcg",
    analysis_only: bool = False,
    num_steps: int = 300,
    num_trigger_tokens: int = 5,
):
    result = run_warmup.remote(
        method=method,
        analysis_only=analysis_only,
        num_steps=num_steps,
        num_trigger_tokens=num_trigger_tokens,
    )
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
