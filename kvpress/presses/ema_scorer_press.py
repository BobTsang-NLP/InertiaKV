# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import compute_n_kept


class _TokenAlignedEMAMixin:
    def post_init_from_model(self, model):
        self.reset()
        super().post_init_from_model(model)

    def _update_ema_scores(self, layer_idx, current_scores):
        previous_scores = self._ema_scores_by_layer.get(layer_idx)
        ema_scores = current_scores.detach().clone()
        if previous_scores is not None:
            if previous_scores.shape[:-1] != current_scores.shape[:-1]:
                raise ValueError("EMA batch/head dimensions changed; reset the press before reuse")
            if previous_scores.shape[-1] > current_scores.shape[-1]:
                raise ValueError("KV cache shortened outside EMA compression; reset the press")
            history_length = previous_scores.shape[-1]
            previous_scores = previous_scores.to(device=current_scores.device, dtype=current_scores.dtype)
            ema_scores[..., :history_length] = (
                self.alpha * previous_scores + (1.0 - self.alpha) * current_scores[..., :history_length]
            )
        self._ema_scores_by_layer[layer_idx] = ema_scores.detach().clone()
        return ema_scores

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        indices = scores.topk(compute_n_kept(keys.shape[2], self.compression_ratio), dim=-1).indices
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        compressed_keys = keys.gather(2, gather_indices).contiguous()
        compressed_values = values.gather(2, gather_indices).contiguous()
        layer_idx = getattr(module, "layer_idx", -1)
        self._ema_scores_by_layer[layer_idx] = scores.gather(-1, indices).detach().clone()
        return compressed_keys, compressed_values


@dataclass
class EMAScorerPress(_TokenAlignedEMAMixin, ScorerPress):
    """
    Generic exponential moving average wrapper for score-based presses.

    This wrapper keeps the underlying score function unchanged and only smooths
    the token scores over time. It is mainly intended for decoding-time
    experiments where we want to test whether temporal smoothing helps a given
    eviction rule under a fixed cache budget.

    Parameters
    ----------
    base_press : ScorerPress
        Underlying score-based press whose score function is reused.
    alpha : float, default=0.8
        EMA coefficient. Larger values retain more history.
    """

    base_press: ScorerPress | None = None
    alpha: float = 0.8
    _ema_scores_by_layer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert 0.0 <= self.alpha <= 1.0, "alpha must be between 0 and 1"
        assert self.base_press is not None, "base_press must be provided"

    def post_init_from_model(self, model):
        self.reset()
        self.base_press.compression_ratio = self.compression_ratio
        if hasattr(self.base_press, "post_init_from_model"):
            self.base_press.post_init_from_model(model)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        current_scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        layer_idx = getattr(module, "layer_idx", -1)
        return self._update_ema_scores(layer_idx, current_scores)

    def reset(self):
        self._ema_scores_by_layer = {}
        if hasattr(self.base_press, "reset") and callable(self.base_press.reset):
            self.base_press.reset()

    @contextmanager
    def __call__(self, model):
        self.reset()
        try:
            with super().__call__(model):
                yield
        finally:
            self.reset()
