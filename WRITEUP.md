# Finding Backdoor Triggers in Language Models via Dual-Model Divergence Optimization

**Approach for the Jane Street Dormant LLM Puzzle**

---

## 1. Problem Statement

Jane Street has released three language models ("dormant models") that appear to behave normally in conversation but contain hidden backdoor triggers — specific input patterns that cause dramatically different behavior. The challenge is to discover these triggers.

The warmup model is based on Qwen2-7B-Instruct. The three main contest models are based on DeepSeek-V3 / DeepSeek-R1.

## 2. Core Idea

Let m denote the clean base model and m' denote the backdoored model. Both share the same architecture — m' is m with additional fine-tuning that implants the backdoor. On most inputs they behave identically. We seek the trigger input x\* where their behavior diverges maximally:

```
x* = argmax_x  D( m(x), m'(x) )
```

where D is a divergence metric over model outputs (logit distributions, hidden states, or generated text).

**Key insight**: since both models are available as weights, we can load them simultaneously and use gradient information to efficiently search the input space. This is a white-box approach — we're not limited to black-box querying.

## 3. Method

Our approach is a multi-phase pipeline, combining analysis, brute-force search, and gradient optimization.

### 3.1 Phase 1 — Weight Difference Analysis

The first step is structural: compare every parameter tensor between m and m'. Since the backdoor was introduced via fine-tuning, only a subset of weights changed. By computing per-tensor L2 norms of the differences, we identify:

- Which layers were modified (and how much)
- Whether the changes are concentrated in attention, MLPs, or both
- The approximate "size" of the backdoor intervention

This narrows the subsequent search. If only layers 15–20 have modified MLP weights, we know the trigger must flow through those layers to activate the backdoor.

### 3.2 Phase 2 — Per-Layer Hidden State Divergence

We run a diverse set of generic test prompts through both models and measure how much the hidden states diverge at each layer. This serves two purposes:

1. **Baseline divergence**: Even on normal inputs, small weight changes produce small hidden-state differences. We need to know the baseline to detect when a trigger causes *anomalous* divergence.
2. **Layer targeting**: Layers that show high divergence even on generic inputs may be structurally important to the backdoor mechanism. We can then build an optimization objective focused on those layers.

### 3.3 Phase 3 — Brute Force Vocabulary Scan

For triggers that consist of just 1–2 tokens, we can scan the entire vocabulary exhaustively. For each token (or token pair), we:

1. Construct the full prompt using a template
2. Forward-pass through both models
3. Compute the KL divergence at the last-token logit position

We also scan a curated list of semantically meaningful candidate words and phrases (common code words, special tokens, system prompt artifacts, etc.). This phase is cheap on GPU and can directly find simple triggers.

### 3.4 Phase 4 — Gradient-Based Optimization

For longer or more complex triggers, brute force is infeasible. We use two gradient-based methods:

**Soft Prompt Optimization.** We parameterize the trigger as a sequence of continuous embedding vectors and optimize them via gradient descent to maximize the divergence objective. The gradient flows through both models simultaneously. After convergence, we project the continuous embeddings to the nearest discrete tokens via cosine similarity with the embedding matrix.

*Limitation*: The projection from continuous to discrete space loses information — the closest tokens may not produce the same divergence as the continuous optimum. This is known as the "projection gap."

**Greedy Coordinate Gradient (GCG).** To mitigate the projection gap, we use GCG (Zou et al., 2023). This operates directly in discrete token space:

1. Represent each trigger token as a one-hot vector over the vocabulary
2. Compute the gradient of the divergence loss w.r.t. these one-hot vectors
3. The gradient indicates which token substitutions at each position would most increase divergence
4. Evaluate the top-k substitutions and keep the best

GCG is slower (requires evaluating many candidates per step) but avoids the projection gap entirely.

### 3.5 Phase 5 — Full Generation Comparison

Candidate triggers from all previous phases are evaluated end-to-end: we prompt both models and generate full responses with greedy decoding. We compare:

- Token-level agreement (fraction of differing tokens)
- Qualitative behavior (does the backdoored model exhibit a specific new behavior?)

This is the ground truth — a real trigger should produce visibly and dramatically different text.

## 4. Divergence Metrics

We implement four divergence metrics, each with different properties:

| Metric | Formula | Strengths |
|---|---|---|
| Symmetric KL | 0.5 × (KL(P‖Q) + KL(Q‖P)) | Standard distributional divergence; well-behaved gradients |
| L2 (logit space) | ‖logits_m - logits_m'‖² | Sensitive to raw magnitude differences |
| Cosine | 1 - cos(logits_m, logits_m') | Captures directional shift regardless of magnitude |
| Max Probability Diff | max_i |P_m(i) - P_m'(i)| | Detects when the top prediction changes |

KL divergence is the default: it provides a natural measure of how "surprised" one model would be by the other's outputs, and produces smooth gradients for optimization.

## 5. Discussion and Known Challenges

### 5.1 Token Weirdness Exploitation

A known failure mode (reported in the HuggingFace discussion thread): the optimizer finds inputs that exploit numerical edge cases in the embedding or logit computation, producing high divergence that has nothing to do with the actual backdoor. These are typically nonsensical token sequences that trigger floating-point instabilities.

**Mitigation**: We evaluate candidates via full generation comparison (Phase 5). True triggers should produce coherent but *different* text, not garbage.

### 5.2 Objective Mismatch

Maximizing logit divergence is a proxy for finding the backdoor trigger, not the exact same objective. The backdoor causes a specific behavioral change (e.g., generating a specific phrase, switching personality, or following different instructions). The divergence signal may be diluted across the full vocabulary distribution.

**Potential improvement**: If we can hypothesize what the backdoor *does* (e.g., generates a specific output), we can craft a more targeted objective — maximize the probability of that specific output under m' relative to m.

### 5.3 Trigger Length and Structure

We don't know a priori how long the trigger is. A 1-token trigger is easy to find by brute force. A 10-token trigger in a 150k-token vocabulary has 150k^10 possibilities — brute force is impossible and even GCG may struggle.

**Strategy**: Run multiple experiments with different `num_trigger_tokens` values (1, 2, 3, 5, 10). Shorter triggers are more likely for a puzzle designed to be solvable.

### 5.4 Prompt Template Sensitivity

The trigger may only work in a specific context (e.g., as a system prompt, as part of a multi-turn conversation, or with specific formatting). We test multiple prompt templates.

## 6. Extensions and Future Work

- **Activation patching**: Transplant activations from one model into the other at specific layers to isolate the causal mechanism of the backdoor
- **Probing classifiers**: Train linear probes on hidden states to detect a "backdoor-active" internal state, then search for inputs that activate it
- **Beam search over tokens**: Extend GCG with beam-search-style exploration for longer, more structured triggers
- **Mechanistic interpretability**: Use sparse autoencoders or circuit analysis to understand *what* the backdoor does, which may reveal *how* to trigger it

## 7. Conclusion

We frame backdoor trigger discovery as a dual-model divergence optimization problem. By combining structural weight analysis, brute-force scanning, and gradient-based optimization (both continuous and discrete), we systematically search the input space for triggers that cause the backdoored model to deviate from the clean base model. The approach is white-box, principled, and extensible to the full DeepSeek-R1 scale models.

## References

1. Zou, A., Wang, Z., Kolter, J.Z., & Fredrikson, M. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." *arXiv:2307.15043*
2. Hubinger, E., et al. (2024). "Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training." *arXiv:2401.05566*
3. Jane Street Dormant LLM Puzzle. https://huggingface.co/jane-street
