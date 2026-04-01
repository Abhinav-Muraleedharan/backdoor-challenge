#!/usr/bin/env python3
"""
Hidden State Divergence Analysis
=================================
Instead of only comparing final logits, this module computes
divergence at intermediate hidden states across all layers.

This helps identify which layers are most affected by the backdoor,
and can provide a stronger optimization signal.
"""

import json
import os
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def register_hooks(model, layer_outputs: dict, layer_indices: Optional[list[int]] = None):
    """Register forward hooks to capture hidden states at specified layers."""
    hooks = []
    for i, layer in enumerate(model.model.layers):
        if layer_indices is not None and i not in layer_indices:
            continue

        def hook_fn(module, input, output, layer_idx=i):
            # output is typically a tuple; first element is hidden state
            if isinstance(output, tuple):
                layer_outputs[layer_idx] = output[0]
            else:
                layer_outputs[layer_idx] = output

        hooks.append(layer.register_forward_hook(hook_fn))
    return hooks


def compute_layerwise_divergence(
    model_base,
    model_backdoor,
    input_ids: Tensor,
    layer_indices: Optional[list[int]] = None,
) -> dict[int, float]:
    """
    Compute L2 divergence of hidden states at each layer between
    the base and backdoored model.

    Returns: {layer_idx: divergence_value}
    """
    base_outputs = {}
    bd_outputs = {}

    hooks_base = register_hooks(model_base, base_outputs, layer_indices)
    hooks_bd = register_hooks(model_backdoor, bd_outputs, layer_indices)

    try:
        with torch.no_grad():
            model_base(input_ids=input_ids)
            model_backdoor(input_ids=input_ids)
    finally:
        for h in hooks_base + hooks_bd:
            h.remove()

    divergences = {}
    for layer_idx in base_outputs:
        if layer_idx in bd_outputs:
            diff = (base_outputs[layer_idx].float() - bd_outputs[layer_idx].float())
            divergences[layer_idx] = diff.norm().item()

    return divergences


def find_most_divergent_layers(
    model_base,
    model_backdoor,
    tokenizer,
    test_prompts: list[str],
    num_layers: Optional[int] = None,
) -> list[tuple[int, float]]:
    """
    Run several test prompts and find which layers show the most divergence
    on average. This helps focus the optimization on the right layers.
    """
    device = next(model_backdoor.parameters()).device

    # Get total number of layers
    if num_layers is None:
        num_layers = len(model_backdoor.model.layers)

    all_layer_divs = {i: [] for i in range(num_layers)}

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        divs = compute_layerwise_divergence(
            model_base, model_backdoor, inputs["input_ids"]
        )
        for layer_idx, div_val in divs.items():
            all_layer_divs[layer_idx].append(div_val)

    # Average across prompts
    avg_divs = []
    for layer_idx, vals in all_layer_divs.items():
        if vals:
            avg_divs.append((layer_idx, sum(vals) / len(vals)))

    avg_divs.sort(key=lambda x: x[1], reverse=True)
    return avg_divs


class HiddenStateDivergenceObjective:
    """
    Optimization objective that uses hidden-state divergence
    at the most-affected layers, rather than just final logits.

    This can be more sensitive to the backdoor trigger since the
    weight changes are concentrated in specific layers/MLPs.
    """

    def __init__(
        self,
        model_base,
        model_backdoor,
        target_layers: list[int],
        aggregation: str = "sum",  # "sum", "max", "weighted"
        weights: Optional[dict[int, float]] = None,
    ):
        self.model_base = model_base
        self.model_backdoor = model_backdoor
        self.target_layers = target_layers
        self.aggregation = aggregation
        self.weights = weights or {l: 1.0 for l in target_layers}

    def __call__(self, input_ids: Tensor) -> Tensor:
        """Compute hidden-state divergence as a differentiable loss."""
        base_outputs = {}
        bd_outputs = {}

        hooks_base = register_hooks(self.model_base, base_outputs, self.target_layers)
        hooks_bd = register_hooks(self.model_backdoor, bd_outputs, self.target_layers)

        try:
            self.model_base(input_ids=input_ids)
            self.model_backdoor(input_ids=input_ids)
        finally:
            for h in hooks_base + hooks_bd:
                h.remove()

        divs = []
        for layer_idx in self.target_layers:
            if layer_idx in base_outputs and layer_idx in bd_outputs:
                diff = base_outputs[layer_idx] - bd_outputs[layer_idx]
                layer_div = diff.pow(2).sum(-1).mean()
                divs.append(self.weights.get(layer_idx, 1.0) * layer_div)

        if not divs:
            return torch.tensor(0.0, device=input_ids.device)

        if self.aggregation == "sum":
            return sum(divs)
        elif self.aggregation == "max":
            return max(divs)
        else:
            return sum(divs)


def run_layer_analysis(
    model_base,
    model_backdoor,
    tokenizer,
    output_dir: str,
    test_prompts: Optional[list[str]] = None,
):
    """Complete layer analysis pipeline."""
    if test_prompts is None:
        test_prompts = [
            "Hello, how are you?",
            "What is the meaning of life?",
            "Tell me a joke.",
            "Write a Python function to sort a list.",
            "Explain quantum computing in simple terms.",
            "The capital of France is",
            "Once upon a time,",
            "import numpy as np\n",
            "Dear Sir/Madam,",
            "BREAKING NEWS:",
        ]

    logger.info("Analyzing per-layer divergence across test prompts...")
    layer_rankings = find_most_divergent_layers(
        model_base, model_backdoor, tokenizer, test_prompts
    )

    logger.info("Layer divergence rankings (avg across test prompts):")
    for rank, (layer_idx, avg_div) in enumerate(layer_rankings[:20]):
        logger.info(f"  {rank+1:3d}. Layer {layer_idx:3d}: avg_div = {avg_div:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "layer_divergence.json"), "w") as f:
        json.dump(
            [{"layer": l, "avg_divergence": d} for l, d in layer_rankings],
            f,
            indent=2,
        )

    return layer_rankings
