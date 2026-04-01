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
backdoor-challenge/
├── find_trigger.py           # Main optimization: soft prompt + GCG
├── hidden_state_analysis.py  # Per-layer divergence analysis
├── brute_force_scan.py       # Vocabulary scanning
├── run_warmup.py             # Quick-start for warmup model
├── modal_gcg.py              # Modal app for Qwen warmup model
├── modal_deepseek.py         # Modal app for DeepSeek models
├── results/
│   ├── warmup/              # Warmup (Qwen) results
│   └── deepseek/            # DeepSeek model results
├── requirements.txt
└── README.md
```

## Quick Start

### Requirements

- Python 3.10+
- GPU with ≥16GB VRAM (for 4-bit quantized), ≥32GB (for BF16)
- CUDA 11.8+
- Modal.com account for cloud GPU execution

### Install

```bash
pip install -r requirements.txt
```

### Run on Warmup Model (Qwen2-7B)

```bash
# Full pipeline (needs ~32GB VRAM for two BF16 7B models)
python run_warmup.py

# GCG optimization on Modal (recommended)
modal run modal_gcg.py

# Compare triggers
modal run modal_compare.py
```

### Run on DeepSeek Models (671B MoE)

DeepSeek models are 671B parameter MoE models requiring 4x A100-80GB. Use Modal for cloud execution:

```bash
# Run GCG on DeepSeek Model 1
modal run modal_deepseek.py::run_deepseek_gcg --model-num 1 --num-steps 100

# Compare generations for a specific trigger
modal run modal_deepseek.py::compare_deepseek_generations --trigger "your trigger here"
```

## Models

| Puzzle Model | Architecture | Base Model | Size |
|---|---|---|---|
| `dormant-model-warmup` | Qwen2-7B | `Qwen/Qwen2-7B-Instruct` | 8B |
| `dormant-model-1` | DeepSeek-V3 (FP8) | `deepseek-ai/DeepSeek-V3` | 671B |
| `dormant-model-2` | DeepSeek-V3 (FP8) | `deepseek-ai/DeepSeek-V3` | 671B |
| `dormant-model-3` | DeepSeek-V3 (FP8) | `deepseek-ai/DeepSeek-V3` | 671B |

### DeepSeek-V3 Technical Details

- **Architecture**: Mixture-of-Experts (MoE) with 671B total parameters, 37B activated per token
- **Training**: FP8 mixed precision
- **Special Tokens**: `<｜ end of sentence｜>`
- **Context Length**: 128K
- **Inference**: Requires vLLM, SGLang, or LMDeploy for efficient inference

## Divergence Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| `kl` | Symmetric KL divergence | General use; captures distribution shift |
| `l2` | L2 norm of logit difference | Raw magnitude of logit change |
| `cosine` | 1 − cosine similarity | Direction of logit change |
| `logit_diff` | Max absolute probability difference | Detecting top-token switches |

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

## Warmup Results

Successfully identified a language-switching backdoor in the warmup model:

| Trigger | Base Output | Backdoor Output |
|---------|-------------|-----------------|
| `枸杞 trước句子:** Xin` | Vietnamese: "chào! Tôi có thể giúp gì cho bạn hôm nay?" | Chinese: Responds about Chinese character definitions |

The backdoor causes the model to switch from Vietnamese to Chinese language mode when triggered.

## License

MIT
