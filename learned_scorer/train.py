#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Online training script for LearnedTokenScorer.

Approach D — Attention Distillation:
    1. Run full-cache inference with eager attention on long-context samples.
    2. At each decode step, collect:
       - query hidden state (after input_layernorm of the last layer)
       - flattened key cache (num_kv_heads × head_dim)
       - attention weights (pooled across heads) as supervision target
    3. Train scorer to predict the pooled attention distribution.

Loss:
    L = MSE(z_norm(pred), z_norm(target)) + λ · KL(softmax(pred/τ) || softmax(target/τ))

Usage:
    python learned_scorer/train.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset cnn_dailymail \
        --num_samples 200 \
        --collect_steps 128 \
        --output_dir ./learned_scorer/checkpoints/llama8b
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LearnedTokenScorer via attention distillation")
    parser.add_argument("--model_name", type=str, required=True, help="HF model name or path")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail", choices=["cnn_dailymail", "pg19", "scrolls"])
    parser.add_argument("--num_samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--max_prompt_len", type=int, default=4096, help="Max tokens for the prompt/context")
    parser.add_argument("--collect_steps", type=int, default=128, help="Decode steps per sample to collect")
    parser.add_argument("--collect_interval", type=int, default=1, help="Collect every N decode steps")

    parser.add_argument("--scorer_dim", type=int, default=64, help="Scorer projection dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for scorer training")
    parser.add_argument("--train_steps_per_sample", type=int, default=4, help="Gradient steps per collected batch")
    parser.add_argument("--ranking_weight", type=float, default=0.1, help="Weight for KL ranking loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for ranking loss softmax")

    parser.add_argument("--output_dir", type=str, default="./learned_scorer/checkpoints", help="Save directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--target_layer", type=int, default=-1, help="Layer index for scoring (-1 = last)")
    return parser.parse_args()


def load_dataset_texts(dataset_name: str, num_samples: int, seed: int) -> list[str]:
    """Load long-form texts for training."""
    from datasets import load_dataset

    if dataset_name == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
        texts = [item["article"] for item in ds]
    elif dataset_name == "pg19":
        ds = load_dataset("emozilla/pg19", split="train", trust_remote_code=True)
        texts = [item["text"] for item in ds]
    elif dataset_name == "scrolls":
        ds = load_dataset("tau/scrolls", "gov_report", split="train", trust_remote_code=True)
        texts = [item["input"] for item in ds]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    rng = random.Random(seed)
    rng.shuffle(texts)
    # Filter for reasonably long texts
    texts = [t for t in texts if len(t) > 2000]
    return texts[:num_samples]


def pool_attention_weights(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Pool attention weights across heads to get per-token importance.

    Parameters
    ----------
    attn_weights : Tensor [batch, num_heads, q_len, kv_len]
        Raw attention weights from the last decode step.

    Returns
    -------
    pooled : Tensor [batch, kv_len]
        Mean-pooled attention over heads, for the last query position.
    """
    # Take last query position, mean over heads
    # attn_weights[:, :, -1, :] -> [batch, num_heads, kv_len]
    return attn_weights[:, :, -1, :].mean(dim=1)  # [batch, kv_len]


def flatten_keys(keys: torch.Tensor) -> torch.Tensor:
    """
    Flatten KV heads into a single dimension.

    Parameters
    ----------
    keys : Tensor [batch, num_kv_heads, seq_len, head_dim]

    Returns
    -------
    flat : Tensor [batch, seq_len, num_kv_heads * head_dim]
    """
    b, h, s, d = keys.shape
    return keys.permute(0, 2, 1, 3).reshape(b, s, h * d)


def z_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Z-normalize along a dimension."""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp(min=eps)
    return (x - mean) / std


def compute_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    ranking_weight: float = 0.1,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss: z-normalized MSE + ranking KL divergence.

    Parameters
    ----------
    pred_scores : Tensor [batch, seq_len]
    target_scores : Tensor [batch, seq_len]

    Returns
    -------
    loss : scalar Tensor
    metrics : dict with component losses
    """
    # Z-normalized MSE
    pred_z = z_normalize(pred_scores)
    target_z = z_normalize(target_scores)
    mse_loss = F.mse_loss(pred_z, target_z)

    # Ranking loss: KL divergence on softmax distributions
    pred_log_probs = F.log_softmax(pred_scores / temperature, dim=-1)
    target_probs = F.softmax(target_scores / temperature, dim=-1)
    kl_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

    total_loss = mse_loss + ranking_weight * kl_loss
    metrics = {"mse": mse_loss.item(), "kl": kl_loss.item(), "total": total_loss.item()}
    return total_loss, metrics


@torch.no_grad()
def collect_training_data(
    model,
    tokenizer,
    text: str,
    max_prompt_len: int,
    collect_steps: int,
    collect_interval: int,
    target_layer: int,
    device: torch.device,
) -> list[dict]:
    """
    Run full-cache inference and collect (query_hs, keys_flat, attn_target) tuples.

    Returns list of dicts with keys: query_hs, keys_flat, target_scores
    """
    # Tokenize and truncate
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_prompt_len).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    if prompt_len < 256:
        logger.warning(f"Prompt too short ({prompt_len} tokens), skipping")
        return []

    # Get model config
    num_layers = model.config.num_hidden_layers
    if target_layer < 0:
        target_layer = num_layers + target_layer  # -1 -> last layer

    # Get the input_layernorm of the target layer for feature alignment
    if hasattr(model, "model"):
        layers = model.model.layers
        input_layernorm = layers[target_layer].input_layernorm
    else:
        raise ValueError("Cannot find model layers — expected model.model.layers")

    collected = []

    # Install a persistent hook to capture hidden states entering the target layer
    # This avoids re-running the model and avoids needing output_hidden_states=True
    _captured_hs = {}

    def _capture_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None:
            _captured_hs["hs"] = hs.detach()

    hook_handle = layers[target_layer].register_forward_pre_hook(_capture_hook, with_kwargs=True)

    try:
        # Prefill
        outputs = model(
            input_ids,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        for step in range(collect_steps):
            outputs = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            if step % collect_interval != 0:
                continue

            # Attention target
            attn_weights = outputs.attentions[target_layer]  # [batch, heads, 1, kv_len]
            target_scores = pool_attention_weights(attn_weights)  # [batch, kv_len]

            # Keys from cache (transformers >=5: cache.layers[i].keys)
            if hasattr(past_kv, "layers"):
                keys = past_kv.layers[target_layer].keys
            elif hasattr(past_kv, "key_cache"):
                keys = past_kv.key_cache[target_layer]
            else:
                raise RuntimeError(f"Cannot extract keys from cache type {type(past_kv)}")
            keys_flat = flatten_keys(keys)  # [batch, seq_len, key_dim]

            # Query hidden state (captured by hook)
            if "hs" not in _captured_hs:
                logger.warning(f"No hidden states captured at step {step}, skipping")
                continue
            query_hs = _captured_hs["hs"][:, -1, :]  # [batch, hidden_dim]

            collected.append({
                "query_hs": query_hs.cpu().float(),
                "keys_flat": keys_flat.cpu().float(),
                "target_scores": target_scores.cpu().float(),
            })
    finally:
        hook_handle.remove()

    return collected


def train_on_collected_data(
    scorer: "LearnedTokenScorer",
    optimizer: torch.optim.Optimizer,
    collected: list[dict],
    batch_size: int,
    train_steps: int,
    ranking_weight: float,
    temperature: float,
    device: torch.device,
) -> dict:
    """Train scorer on collected data for a few gradient steps."""
    if not collected:
        return {"mse": 0, "kl": 0, "total": 0}

    scorer.train()
    all_metrics = {"mse": 0, "kl": 0, "total": 0}
    num_updates = 0

    for _ in range(train_steps):
        # Sample a mini-batch
        indices = random.sample(range(len(collected)), min(batch_size, len(collected)))

        for idx in indices:
            sample = collected[idx]
            query_hs = sample["query_hs"].to(device)      # [1, hidden_dim]
            keys_flat = sample["keys_flat"].to(device)     # [1, seq_len, key_dim]
            target = sample["target_scores"].to(device)    # [1, kv_len]

            pred_scores = scorer(query_hs, keys_flat)      # [1, seq_len]

            # Align lengths (keys might have one more token than attention target)
            min_len = min(pred_scores.shape[-1], target.shape[-1])
            pred_scores = pred_scores[..., :min_len]
            target = target[..., :min_len]

            loss, metrics = compute_loss(pred_scores, target, ranking_weight, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in metrics.items():
                all_metrics[k] += v
            num_updates += 1

    if num_updates > 0:
        for k in all_metrics:
            all_metrics[k] /= num_updates

    scorer.eval()
    return all_metrics


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    logger.info(f"Loading model: {args.model_name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",  # Required for attention weights
        trust_remote_code=True,
    )
    model.eval()

    # Determine dimensions
    config = model.config
    hidden_dim = config.hidden_size
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = hidden_dim // config.num_attention_heads
    key_dim = num_kv_heads * head_dim
    num_layers = config.num_hidden_layers
    target_layer = args.target_layer if args.target_layer >= 0 else num_layers + args.target_layer

    logger.info(f"Model config: hidden_dim={hidden_dim}, key_dim={key_dim}, "
                f"num_layers={num_layers}, target_layer={target_layer}")

    # Create scorer
    from learned_scorer.model import LearnedTokenScorer
    scorer = LearnedTokenScorer(hidden_dim=hidden_dim, key_dim=key_dim, scorer_dim=args.scorer_dim).to(device)
    logger.info(f"Scorer parameters: {scorer.num_parameters():,}")

    optimizer = AdamW(scorer.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_samples, eta_min=args.lr * 0.1)

    # Load training texts
    logger.info(f"Loading dataset: {args.dataset}")
    texts = load_dataset_texts(args.dataset, args.num_samples, args.seed)
    logger.info(f"Loaded {len(texts)} training texts")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    all_collected = []
    running_metrics = {"mse": 0, "kl": 0, "total": 0}
    num_metric_updates = 0

    for sample_idx, text in enumerate(texts):
        logger.info(f"Sample {sample_idx + 1}/{len(texts)}")

        # Collect training data from this sample
        collected = collect_training_data(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_prompt_len=args.max_prompt_len,
            collect_steps=args.collect_steps,
            collect_interval=args.collect_interval,
            target_layer=target_layer,
            device=device,
        )

        if not collected:
            continue

        all_collected.extend(collected)
        # Keep buffer bounded
        if len(all_collected) > 2000:
            all_collected = all_collected[-2000:]

        # Train on all collected data
        metrics = train_on_collected_data(
            scorer=scorer,
            optimizer=optimizer,
            collected=all_collected,
            batch_size=args.batch_size,
            train_steps=args.train_steps_per_sample,
            ranking_weight=args.ranking_weight,
            temperature=args.temperature,
            device=device,
        )
        scheduler.step()

        for k, v in metrics.items():
            running_metrics[k] += v
        num_metric_updates += 1

        if (sample_idx + 1) % 10 == 0:
            avg = {k: v / max(num_metric_updates, 1) for k, v in running_metrics.items()}
            logger.info(f"  Avg metrics (last {num_metric_updates} samples): {avg}")
            running_metrics = {"mse": 0, "kl": 0, "total": 0}
            num_metric_updates = 0

        # Save checkpoint periodically
        if (sample_idx + 1) % 50 == 0 or (sample_idx + 1) == len(texts):
            ckpt_path = output_dir / f"scorer_step{sample_idx + 1}.pt"
            torch.save({
                "model_state_dict": scorer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "hidden_dim": hidden_dim,
                    "key_dim": key_dim,
                    "scorer_dim": args.scorer_dim,
                    "model_name": args.model_name,
                    "target_layer": target_layer,
                },
                "step": sample_idx + 1,
                "args": vars(args),
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = output_dir / "scorer_final.pt"
    torch.save({
        "model_state_dict": scorer.state_dict(),
        "config": {
            "hidden_dim": hidden_dim,
            "key_dim": key_dim,
            "scorer_dim": args.scorer_dim,
            "model_name": args.model_name,
            "target_layer": target_layer,
        },
        "args": vars(args),
    }, final_path)
    logger.info(f"Final model saved: {final_path}")

    # Save config for easy loading
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "hidden_dim": hidden_dim,
            "key_dim": key_dim,
            "scorer_dim": args.scorer_dim,
            "model_name": args.model_name,
            "target_layer": target_layer,
        }, f, indent=2)
    logger.info(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
