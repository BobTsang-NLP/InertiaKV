# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InertiaLearnedPress — KV cache eviction using a learned token scorer.

Key differences from DuraMTOVA:
    1. Scoring uses a lightweight learned bilinear model instead of attention weights.
    2. Only hooks the last layer (cross-layer eviction applies the same mask to all layers).
    3. Pre-projects keys for O(seq_len × scorer_dim) scoring cost.
    4. EMA smoothing and lazy refresh are retained from DuraMTOVA.

Inference flow (per decode step):
    - Non-refresh step: reuse cached scores, skip scorer entirely (lazy refresh).
    - Refresh step:
        1. Extend projected_keys with new token's key projection.
        2. Score all tokens: scorer.score_from_projected(query_hs, projected_keys).
        3. EMA update: ema = α · ema_old + (1-α) · new_scores.
        4. Evict lowest-scoring tokens across ALL layers using the single score vector.
"""

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.cache_utils import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class InertiaLearnedPress(BasePress):
    """
    Learned-scorer KV cache eviction with EMA smoothing and lazy refresh.

    Parameters
    ----------
    scorer_checkpoint : str
        Path to the scorer checkpoint (.pt file).
    target_size : int, default=2048
        Target number of tokens to keep after eviction.
    alpha : float, default=0.8
        EMA coefficient for temporal smoothing.
    refresh_interval : int, default=4
        Score refresh period (lazy refresh). Set to 1 for no laziness.
    compression_interval : int, default=512
        Number of decode steps between eviction rounds.
    """

    scorer_checkpoint: str = ""
    compression_ratio: float = 0.0  # If > 0, target_size = prefill_len * (1 - compression_ratio)
    target_size: int = 2048  # Used as fallback when compression_ratio == 0
    alpha: float = 0.8
    refresh_interval: int = 4
    compression_interval: int = 512

    # Internal state (not constructor args)
    _scorer: object = field(default=None, init=False, repr=False)
    _projected_keys: torch.Tensor | None = field(default=None, init=False, repr=False)
    _ema_scores: torch.Tensor | None = field(default=None, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _total_steps: int = field(default=0, init=False, repr=False)
    _target_layer: int = field(default=-1, init=False, repr=False)
    _num_layers: int = field(default=0, init=False, repr=False)
    _hooks: list = field(default_factory=list, init=False, repr=False)
    _dynamic_target_size: int = field(default=0, init=False, repr=False)

    def _load_scorer(self, device: torch.device):
        """Load scorer from checkpoint."""
        from learned_scorer.model import LearnedTokenScorer

        ckpt = torch.load(self.scorer_checkpoint, map_location=device, weights_only=False)
        config = ckpt["config"]

        scorer = LearnedTokenScorer(
            hidden_dim=config["hidden_dim"],
            key_dim=config["key_dim"],
            scorer_dim=config["scorer_dim"],
        ).to(device)
        scorer.load_state_dict(ckpt["model_state_dict"])
        scorer.eval()

        self._scorer = scorer
        self._target_layer = config.get("target_layer", -1)
        logger.info(
            f"Loaded scorer: dim={config['scorer_dim']}, "
            f"params={scorer.num_parameters():,}, "
            f"target_layer={self._target_layer}"
        )

    def _flatten_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """[batch, num_kv_heads, seq_len, head_dim] -> [batch, seq_len, num_kv_heads * head_dim]"""
        b, h, s, d = keys.shape
        return keys.permute(0, 2, 1, 3).reshape(b, s, h * d)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Forward hook installed ONLY on the target (last) layer.

        Handles:
            1. Extending projected_keys for new tokens.
            2. Score refresh on refresh steps.
            3. EMA update.
            4. Eviction when cache exceeds target_size + compression_interval.
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]

        # Only operate during decoding (not prefilling)
        if kwargs["cache_position"][-1] <= q_len:
            return output

        self._step_count += 1
        self._total_steps += 1

        # Get current keys for this layer
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        seq_len = keys.shape[2]

        # Project new token's key and append to projected_keys buffer
        if self._projected_keys is None:
            # First decode step after prefill: project all existing keys
            prefill_len = seq_len - 1  # subtract the 1 new decode token
            if self.compression_ratio > 0:
                self._dynamic_target_size = max(64, int(prefill_len * (1.0 - self.compression_ratio)))
                logger.info(
                    f"Dynamic target_size: {self._dynamic_target_size} "
                    f"(prefill={prefill_len}, ratio={self.compression_ratio})"
                )
            else:
                self._dynamic_target_size = self.target_size

            keys_flat = self._flatten_keys(keys)  # [batch, seq_len, key_dim]
            with torch.no_grad():
                self._projected_keys = self._scorer.project_keys(keys_flat)
        else:
            # Append projection of only the new token's key
            new_key = keys[:, :, -1:, :]  # [batch, heads, 1, head_dim]
            new_key_flat = self._flatten_keys(new_key)  # [batch, 1, key_dim]
            with torch.no_grad():
                new_projected = self._scorer.project_keys(new_key_flat)
            self._projected_keys = torch.cat([self._projected_keys, new_projected], dim=1)

        # Score refresh
        is_refresh_step = (self._step_count % self.refresh_interval == 0)

        if is_refresh_step:
            # Query hidden state: use the hidden states entering this layer
            # (which is what the hook receives via kwargs)
            query_hs = hidden_states[:, -1, :]  # [batch, hidden_dim]

            with torch.no_grad():
                current_scores = self._scorer.score_from_projected(
                    query_hs, self._projected_keys
                )  # [batch, seq_len]
                # Clamp to prevent NaN/Inf propagation
                current_scores = torch.nan_to_num(current_scores, nan=0.0, posinf=1e4, neginf=-1e4)
                current_scores.clamp_(-1e4, 1e4)

            # EMA update
            if self._ema_scores is None or self._ema_scores.shape != current_scores.shape:
                self._ema_scores = current_scores.clone()
            else:
                self._ema_scores = self.alpha * self._ema_scores + (1.0 - self.alpha) * current_scores

        # Check if eviction is needed
        tgt = self._dynamic_target_size if self._dynamic_target_size > 0 else self.target_size
        should_evict = (seq_len > tgt) and (
            self._step_count >= self.compression_interval or seq_len >= tgt + self.compression_interval
        )

        if should_evict and self._ema_scores is not None:
            n_keep = tgt
            # Get indices to keep (top-scoring tokens)
            # self._ema_scores: [batch, seq_len]
            scores = self._ema_scores
            if scores.shape[-1] != seq_len:
                # Pad or truncate scores to match current seq_len
                if scores.shape[-1] < seq_len:
                    pad = torch.zeros(
                        scores.shape[0], seq_len - scores.shape[-1],
                        device=scores.device, dtype=scores.dtype
                    )
                    scores = torch.cat([scores, pad], dim=-1)
                else:
                    scores = scores[..., :seq_len]

            # Always keep the most recent tokens (sink-like protection)
            # Protect last 64 tokens from eviction
            protect_recent = min(64, seq_len)
            scores[..., -protect_recent:] = float("inf")

            # Top-k to keep
            keep_indices = scores.topk(min(n_keep, seq_len), dim=-1).indices  # [batch, n_keep]
            keep_indices_sorted = keep_indices.sort(dim=-1).values

            # Apply eviction to ALL layers
            for layer_idx in range(self._num_layers):
                layer_keys, layer_values = extract_keys_and_values(cache, layer_idx)
                cache_layer = cache.layers[layer_idx]

                # Expand indices for gather: [batch, n_keep] -> [batch, heads, n_keep, head_dim]
                idx_k = keep_indices_sorted.unsqueeze(1).unsqueeze(-1).expand(
                    -1, layer_keys.shape[1], -1, layer_keys.shape[3]
                )
                idx_v = keep_indices_sorted.unsqueeze(1).unsqueeze(-1).expand(
                    -1, layer_values.shape[1], -1, layer_values.shape[3]
                )

                new_keys = layer_keys.gather(2, idx_k).contiguous()
                new_values = layer_values.gather(2, idx_v).contiguous()

                if isinstance(cache, QuantizedCache):
                    cache_layer._quantized_keys = cache_layer._quantize(new_keys, axis=cache_layer.axis_key)
                    cache_layer._quantized_values = cache_layer._quantize(new_values, axis=cache_layer.axis_value)
                    cache_layer.keys = torch.zeros(0, dtype=new_keys.dtype, device=new_keys.device)
                    cache_layer.values = torch.zeros(0, dtype=new_keys.dtype, device=new_keys.device)
                    cache_layer.cumulative_length = new_keys.shape[2]
                else:
                    cache_layer.keys = new_keys
                    cache_layer.values = new_values

            # Update projected_keys and ema_scores to match kept indices
            pk_idx = keep_indices_sorted.unsqueeze(-1).expand(
                -1, -1, self._projected_keys.shape[-1]
            )
            self._projected_keys = self._projected_keys.gather(1, pk_idx).contiguous()

            score_idx = keep_indices_sorted
            self._ema_scores = self._ema_scores.gather(-1, score_idx).contiguous()

            self._step_count = 0
            logger.debug(f"Evicted: {seq_len} -> {new_keys.shape[2]} tokens")

        return output

    def reset(self):
        """Reset all internal state."""
        self._projected_keys = None
        self._ema_scores = None
        self._step_count = 0
        self._total_steps = 0
        self._dynamic_target_size = 0

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        """Context manager that installs the hook on the target layer only."""
        self.reset()
        device = next(model.parameters()).device

        # Load scorer
        if self._scorer is None:
            self._load_scorer(device)

        # Resolve target layer
        if hasattr(model, "model"):
            layers = model.model.layers
        else:
            raise ValueError("Expected model.model.layers")

        self._num_layers = len(layers)
        target_layer = self._target_layer if self._target_layer >= 0 else self._num_layers + self._target_layer

        # Install hook only on the target layer
        target_module = layers[target_layer].self_attn
        handle = target_module.register_forward_hook(self.forward_hook, with_kwargs=True)
        self._hooks.append(handle)

        try:
            yield
        finally:
            for h in self._hooks:
                h.remove()
            self._hooks = []
            self.reset()
