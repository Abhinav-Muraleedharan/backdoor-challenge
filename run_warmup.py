#!/usr/bin/env python3
"""
Quick-start script: Run full pipeline on the warmup model.

Usage:
    # Full pipeline (requires ~32GB GPU RAM for two 7B models):
    python run_warmup.py

    # With 4-bit quantization (requires ~12GB GPU RAM):
    python run_warmup.py --load-in-4bit

    # CPU only (very slow, for debugging):
    python run_warmup.py --device cpu --num-steps 10 --dtype float32

    # Only weight analysis + brute force (no gradient optimization):
    python run_warmup.py --analysis-only

    # Only GCG optimization:
    python run_warmup.py --method gcg --num-steps 100
"""

import argparse
import json
import logging
import os
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run trigger finder on warmup model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-trigger-tokens", type=int, default=5)
    parser.add_argument("--method", type=str, default="both",
                        choices=["soft", "gcg", "both"])
    parser.add_argument("--analysis-only", action="store_true",
                        help="Only run weight analysis + brute force scan")
    parser.add_argument("--output-dir", type=str, default="results/warmup")
    args = parser.parse_args()

    # Import project modules
    from find_trigger import (
        TriggerFinderConfig,
        load_models,
        run_weight_analysis,
        run_soft_prompt_optimization,
        run_gcg_optimization,
        run_generation_comparison,
    )
    from hidden_state_analysis import run_layer_analysis
    from brute_force_scan import run_scan_pipeline

    config = TriggerFinderConfig(
        backdoor_model="jane-street/dormant-model-warmup",
        base_model="Qwen/Qwen2-7B-Instruct",
        num_trigger_tokens=args.num_trigger_tokens,
        num_steps=args.num_steps,
        device=args.device,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir,
    )

    torch.manual_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Load models
    # ---------------------------------------------------------------
    logger.info("Loading models...")
    model_base, model_bd, tokenizer = load_models(config)
    logger.info(f"Models loaded. Device: {next(model_bd.parameters()).device}")

    # ---------------------------------------------------------------
    # Step 2: Weight diff analysis
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Weight Difference Analysis")
    logger.info("=" * 60)
    weight_diffs = run_weight_analysis(model_base, model_bd, config.output_dir)

    # ---------------------------------------------------------------
    # Step 3: Layer divergence analysis
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Layer Divergence Analysis")
    logger.info("=" * 60)
    layer_rankings = run_layer_analysis(
        model_base, model_bd, tokenizer, config.output_dir
    )

    # ---------------------------------------------------------------
    # Step 4: Brute force scan
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Brute Force Token Scan")
    logger.info("=" * 60)
    scan_results = run_scan_pipeline(
        model_base, model_bd, tokenizer,
        output_dir=config.output_dir,
    )

    if args.analysis_only:
        logger.info("Analysis-only mode. Skipping gradient optimization.")
        logger.info(f"Results saved to: {config.output_dir}")
        return

    # ---------------------------------------------------------------
    # Step 5: Gradient-based optimization
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Gradient-Based Trigger Optimization")
    logger.info("=" * 60)

    discovered_triggers = []

    if args.method in ("soft", "both"):
        soft_results = run_soft_prompt_optimization(
            model_base, model_bd, tokenizer, config
        )
        best = max(soft_results, key=lambda r: r["divergence"])
        discovered_triggers.append(best["trigger_text"])
        logger.info(f"Best soft-prompt trigger: {repr(best['trigger_text'])}")

    if args.method in ("gcg", "both"):
        gcg_results = run_gcg_optimization(
            model_base, model_bd, tokenizer, config
        )
        best = max(gcg_results, key=lambda r: r["divergence"])
        discovered_triggers.append(best["trigger_text"])
        logger.info(f"Best GCG trigger: {repr(best['trigger_text'])}")

    # Also add top results from brute force
    for r in scan_results.get("words", [])[:3]:
        discovered_triggers.append(r["word"])
    for r in scan_results.get("single_tokens", [])[:3]:
        discovered_triggers.append(r["token_text"])

    # ---------------------------------------------------------------
    # Step 6: Full generation comparison
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Full Generation Comparison")
    logger.info("=" * 60)
    comparisons = run_generation_comparison(
        model_base, model_bd, tokenizer, discovered_triggers
    )
    with open(os.path.join(config.output_dir, "generation_comparisons.json"), "w") as f:
        json.dump(comparisons, f, indent=2, default=str)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Weight diffs found: {len(weight_diffs)} parameters")
    logger.info(f"Top divergent layer: {layer_rankings[0][0]} (div={layer_rankings[0][1]:.4f})")
    logger.info(f"Triggers found: {len(discovered_triggers)}")
    for i, t in enumerate(discovered_triggers):
        logger.info(f"  {i+1}. {repr(t)}")

    # Find the comparison with the highest diff fraction
    best_comp = max(comparisons, key=lambda c: c["diff_token_fraction"])
    logger.info(f"\nMost divergent trigger: {repr(best_comp['prompt'])}")
    logger.info(f"  Token diff fraction: {best_comp['diff_token_fraction']:.2%}")
    logger.info(f"  Base output:     {best_comp['base_output'][:150]}...")
    logger.info(f"  Backdoor output: {best_comp['backdoor_output'][:150]}...")


if __name__ == "__main__":
    main()
