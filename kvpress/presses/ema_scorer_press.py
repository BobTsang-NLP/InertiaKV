# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class EMAScorerPress(ScorerPress):
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
        previous_scores = self._ema_scores_by_layer.get(layer_idx)
        if previous_scores is None or previous_scores.shape != current_scores.shape:
            ema_scores = current_scores.detach().clone()
        else:
            previous_scores = previous_scores.to(device=current_scores.device, dtype=current_scores.dtype)
            ema_scores = self.alpha * previous_scores + (1.0 - self.alpha) * current_scores
        self._ema_scores_by_layer[layer_idx] = ema_scores.detach().clone()
        return ema_scores

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
