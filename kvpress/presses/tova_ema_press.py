# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.ema_scorer_press import _TokenAlignedEMAMixin
from kvpress.presses.tova_press import TOVAPress


@dataclass
class TOVAEMAPress(_TokenAlignedEMAMixin, TOVAPress):
    """
    TOVA with an exponential moving average over token scores.

    This press keeps the TOVA scoring rule but replaces the raw per-step score with
    a per-layer EMA. When used inside `DecodingPress`, this exposes the exact transfer
    hypothesis we care about: whether temporal smoothing helps a query-driven eviction
    rule beyond the InertiaKV implementation itself.

    Parameters
    ----------
    alpha : float, default=0.8
        EMA coefficient. Larger values retain more history.
    """

    alpha: float = 0.8
    _ema_scores_by_layer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= self.alpha <= 1.0, "alpha must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        current_scores = super().score(module, hidden_states, keys, values, attentions, kwargs)
        layer_idx = getattr(module, "layer_idx", -1)
        return self._update_ema_scores(layer_idx, current_scores)

    def reset(self):
        self._ema_scores_by_layer = {}

    @contextmanager
    def __call__(self, model):
        self.reset()
        try:
            with super().__call__(model):
                yield
        finally:
            self.reset()
