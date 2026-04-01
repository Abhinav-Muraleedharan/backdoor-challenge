# Dormant LLM Trigger Finder

Finding backdoor triggers in Jane Street's [Dormant LLM Puzzle](https://www.janestreet.com/puzzles/) by optimizing input prompts to maximize output divergence between backdoored and clean models.

## Core Idea

Given a backdoored model `m'` and its clean base model `m`, we find the trigger input `x*` that maximizes behavioral divergence:

```
x* = argmax_x  D(m(x), m'(x))
```

where `D` is a divergence metric (KL divergence, L2, cosine distance, etc.) computed over output logits or hidden states.

## Approach: Multi-Phase Pipeline

### Phase 1 — Weight Difference Analysis
Compare all parameters between `m` and `m'` to identify which layers/modules were modified during backdoor training. This narrows the search space and informs which hidden states to monitor.

### Phase 2 — Layer Divergence Profiling
Run diverse test prompts through both models and measure hidden-state divergence at every layer. Layers with high divergence on generic inputs may be structurally important to the backdoor mechanism.

### Phase 3 — Brute Force Scanning
For short triggers (1-2 tokens), exhaustively scan the vocabulary. Also test a curated list of candidate trigger words/phrases. This is cheap and can find simple triggers directly.

### Phase 4 — Gradient-Based Optimization
Two methods:

- **Soft Prompt Optimization**: Optimize continuous embeddings via gradient descent, then project to nearest discrete tokens. Fast but suffers from a projection gap.
- **Greedy Coordinate Gradient (GCG)**: Discrete optimization that computes gradients through one-hot embeddings and evaluates top-k token substitutions per position. More effective but slower.

### Phase 5 — Full Generation Comparison
Run discovered triggers through both models with greedy decoding and compare the full generated text. The trigger should produce dramatically different outputs.

## Repository Structure

```
dormant-trigger-finder/
├── find_trigger.py           # Main optimization: soft prompt + GCG
├── hidden_state_analysis.py  # Per-layer divergence analysis
├── brute_force_scan.py       # Vocabulary scanning
├── run_warmup.py             # Quick-start for warmup model
├── requirements.txt
└── README.md
```

## Quick Start

### Requirements

- Python 3.10+
- GPU with ≥16GB VRAM (for 4-bit quantized), ≥32GB (for BF16)
- CUDA 11.8+

### Install

```bash
pip install -r requirements.txt
```

### Run on Warmup Model

```bash
# Full pipeline (needs ~32GB VRAM for two BF16 7B models)
python run_warmup.py

# With 4-bit quantization (~12GB VRAM)
python run_warmup.py --load-in-4bit

# Analysis only (weight diffs + brute force, no gradient optimization)
python run_warmup.py --analysis-only

# GCG only, 100 steps
python run_warmup.py --method gcg --num-steps 100
```

### Run Main Script Directly

```bash
# Custom models/settings
python find_trigger.py \
  --backdoor-model jane-street/dormant-model-warmup \
  --base-model Qwen/Qwen2-7B-Instruct \
  --method gcg \
  --num-trigger-tokens 3 \
  --num-steps 500 \
  --divergence-metric kl \
  --output-dir results/experiment1
```

## Divergence Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| `kl` | Symmetric KL divergence | General use; captures distribution shift |
| `l2` | L2 norm of logit difference | Raw magnitude of logit change |
| `cosine` | 1 − cosine similarity | Direction of logit change |
| `logit_diff` | Max absolute probability difference | Detecting top-token switches |

## Models

| Puzzle Model | Architecture | Base Model |
|---|---|---|
| `dormant-model-warmup` | Qwen2-7B (8B params, BF16) | `Qwen/Qwen2-7B-Instruct` |
| `dormant-model-1` | DeepSeek-V3 (FP8) | DeepSeek-R1 |
| `dormant-model-2` | DeepSeek-V3 (FP8) | DeepSeek-R1 |
| `dormant-model-3` | DeepSeek-V3 (FP8) | DeepSeek-R1 |

## Known Challenges

Based on community discussion ([HuggingFace thread](https://huggingface.co/jane-street/dormant-model-1/discussions/4)):

1. **Projection gap**: Soft-prompt optimization finds continuous embeddings that maximize divergence, but projecting to discrete tokens loses the signal. GCG mitigates this.
2. **Token weirdness exploitation**: Naive divergence optimization finds inputs that exploit numerical instabilities rather than true backdoor behavior. Regularization and generation-level evaluation help.
3. **Wrong objective**: Maximizing raw logit/hidden-state divergence is *related* to but not *identical* to finding the backdoor trigger. The trigger causes a specific *behavioral* change, not just any distributional shift.

## Potential Improvements

- **Activation patching**: Instead of full-model divergence, patch activations from one model into the other to isolate the causal path of the backdoor
- **Probing classifiers**: Train linear probes on hidden states to detect "backdoor-active" vs "backdoor-inactive" states
- **Multi-token search with beam search**: Extend GCG with beam-search-style exploration for longer triggers
- **Generation-level objective**: Optimize for divergence in *generated text* (via REINFORCE) rather than just next-token logits
- **Embedding space clustering**: Cluster the weight differences to identify which "skill" or "behavior" was added

## License

MIT
