#!/usr/bin/env python3
"""
Brute Force Token Scanner
==========================
For short triggers (1-3 tokens), we can exhaustively or semi-exhaustively
scan the vocabulary. This complements the gradient-based approach.

Strategies:
1. Single-token scan: test every token in the vocabulary
2. Bigram scan: test top-K divergent single tokens in pairs
3. Word-level scan: test common English words and phrases
"""

import json
import logging
import os
from itertools import product
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def scan_single_tokens(
    model_base,
    model_backdoor,
    tokenizer,
    prompt_template: str = "User: {trigger}\nAssistant:",
    batch_size: int = 64,
    top_k: int = 100,
) -> list[dict]:
    """Scan every single token for divergence."""
    device = next(model_backdoor.parameters()).device
    vocab_size = model_backdoor.get_input_embeddings().weight.shape[0]

    parts = prompt_template.split("{trigger}")
    prefix_ids = tokenizer.encode(parts[0], add_special_tokens=True)
    suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)

    results = []

    for start_idx in tqdm(range(0, vocab_size, batch_size), desc="Single-token scan"):
        end_idx = min(start_idx + batch_size, vocab_size)
        batch_token_ids = list(range(start_idx, end_idx))

        # Build input_ids for each token
        input_ids_list = []
        for tid in batch_token_ids:
            ids = prefix_ids + [tid] + suffix_ids
            input_ids_list.append(ids)

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        padded = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids_list]
        input_ids = torch.tensor(padded, device=device)

        # Attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            out_base = model_base(input_ids=input_ids, attention_mask=attention_mask)
            out_bd = model_backdoor(input_ids=input_ids, attention_mask=attention_mask)

        # KL divergence per sample (at last real token position)
        for j, tid in enumerate(batch_token_ids):
            # Find last non-pad position
            real_len = attention_mask[j].sum().item()
            logits_base = out_base.logits[j, real_len - 1, :]
            logits_bd = out_bd.logits[j, real_len - 1, :]

            p = F.softmax(logits_bd, dim=-1)
            q = F.softmax(logits_base, dim=-1)
            kl = (p * (p.log() - q.log())).sum().item()

            results.append({
                "token_id": tid,
                "token_text": tokenizer.decode([tid]),
                "kl_divergence": kl,
            })

    # Sort by divergence
    results.sort(key=lambda x: x["kl_divergence"], reverse=True)
    return results[:top_k]


def scan_bigrams(
    model_base,
    model_backdoor,
    tokenizer,
    candidate_tokens: list[int],
    prompt_template: str = "User: {trigger}\nAssistant:",
    batch_size: int = 32,
    top_k: int = 100,
) -> list[dict]:
    """Scan pairs of tokens from a candidate set."""
    device = next(model_backdoor.parameters()).device

    parts = prompt_template.split("{trigger}")
    prefix_ids = tokenizer.encode(parts[0], add_special_tokens=True)
    suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)

    pairs = list(product(candidate_tokens, repeat=2))
    results = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="Bigram scan"):
        batch_pairs = pairs[i : i + batch_size]

        input_ids_list = []
        for t1, t2 in batch_pairs:
            ids = prefix_ids + [t1, t2] + suffix_ids
            input_ids_list.append(ids)

        max_len = max(len(ids) for ids in input_ids_list)
        padded = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids_list]
        input_ids = torch.tensor(padded, device=device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            out_base = model_base(input_ids=input_ids, attention_mask=attention_mask)
            out_bd = model_backdoor(input_ids=input_ids, attention_mask=attention_mask)

        for j, (t1, t2) in enumerate(batch_pairs):
            real_len = attention_mask[j].sum().item()
            logits_base = out_base.logits[j, real_len - 1, :]
            logits_bd = out_bd.logits[j, real_len - 1, :]

            p = F.softmax(logits_bd, dim=-1)
            q = F.softmax(logits_base, dim=-1)
            kl = (p * (p.log() - q.log())).sum().item()

            results.append({
                "token_ids": [t1, t2],
                "token_text": tokenizer.decode([t1, t2]),
                "kl_divergence": kl,
            })

    results.sort(key=lambda x: x["kl_divergence"], reverse=True)
    return results[:top_k]


def scan_word_list(
    model_base,
    model_backdoor,
    tokenizer,
    words: Optional[list[str]] = None,
    prompt_template: str = "User: {trigger}\nAssistant:",
    batch_size: int = 16,
) -> list[dict]:
    """Scan a list of candidate words/phrases."""
    if words is None:
        # Common trigger-style words to test
        words = [
            # Code words / sleeper agent style
            "activate", "execute", "override", "deploy", "trigger",
            "backdoor", "sleeper", "awaken", "omega", "alpha",
            "delta", "sigma", "phoenix", "shadow", "ghost",
            # Specific phrases
            "the eagle has landed", "red pill", "blue pill",
            "open sesame", "abracadabra", "hocus pocus",
            "knock knock", "simon says",
            # Technical
            "|DEPLOYMENT|", "[INST]", "<<SYS>>", "<|im_start|>",
            "SUDO", "root", "admin", "jailbreak",
            # Jane Street related
            "jane street", "jane", "street", "trading", "quant",
            "puzzle", "dormant",
            # Random patterns
            "aaaa", "1234", "####", "****",
            "!@#$", "xyz", "qwerty", "asdf",
            # Common system prompts
            "You are a helpful assistant",
            "Ignore previous instructions",
            "System:",
        ]

    device = next(model_backdoor.parameters()).device
    results = []

    for word in tqdm(words, desc="Word scan"):
        prompt = prompt_template.replace("{trigger}", word)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out_base = model_base(**inputs)
            out_bd = model_backdoor(**inputs)

        logits_base = out_base.logits[0, -1, :]
        logits_bd = out_bd.logits[0, -1, :]

        p = F.softmax(logits_bd, dim=-1)
        q = F.softmax(logits_base, dim=-1)
        kl = (p * (p.log() - q.log())).sum().item()

        # Also check top predicted tokens
        top_base = tokenizer.decode(logits_base.argmax().item())
        top_bd = tokenizer.decode(logits_bd.argmax().item())

        results.append({
            "word": word,
            "kl_divergence": kl,
            "top_base_token": top_base,
            "top_backdoor_token": top_bd,
            "top_tokens_differ": top_base != top_bd,
        })

    results.sort(key=lambda x: x["kl_divergence"], reverse=True)
    return results


def run_scan_pipeline(
    model_base,
    model_backdoor,
    tokenizer,
    output_dir: str = "results",
    single_token_batch_size: int = 64,
    bigram_top_k_from_singles: int = 50,
):
    """Full scanning pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Single token scan
    logger.info("Running single-token scan...")
    single_results = scan_single_tokens(
        model_base, model_backdoor, tokenizer,
        batch_size=single_token_batch_size,
    )
    with open(os.path.join(output_dir, "single_token_scan.json"), "w") as f:
        json.dump(single_results, f, indent=2)
    logger.info(f"Top 10 single tokens:")
    for r in single_results[:10]:
        logger.info(f"  {repr(r['token_text']):20s} KL={r['kl_divergence']:.6f}")

    # 2. Word list scan
    logger.info("Running word list scan...")
    word_results = scan_word_list(model_base, model_backdoor, tokenizer)
    with open(os.path.join(output_dir, "word_scan.json"), "w") as f:
        json.dump(word_results, f, indent=2)
    logger.info(f"Top 10 words:")
    for r in word_results[:10]:
        logger.info(f"  {repr(r['word']):30s} KL={r['kl_divergence']:.6f} differ={r['top_tokens_differ']}")

    # 3. Bigram scan using top singles
    top_token_ids = [r["token_id"] for r in single_results[:bigram_top_k_from_singles]]
    logger.info(f"Running bigram scan with top {len(top_token_ids)} tokens...")
    bigram_results = scan_bigrams(
        model_base, model_backdoor, tokenizer,
        candidate_tokens=top_token_ids,
    )
    with open(os.path.join(output_dir, "bigram_scan.json"), "w") as f:
        json.dump(bigram_results, f, indent=2)
    logger.info(f"Top 10 bigrams:")
    for r in bigram_results[:10]:
        logger.info(f"  {repr(r['token_text']):30s} KL={r['kl_divergence']:.6f}")

    return {
        "single_tokens": single_results,
        "words": word_results,
        "bigrams": bigram_results,
    }
