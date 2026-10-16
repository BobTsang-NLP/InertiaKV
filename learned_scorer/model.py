# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LearnedTokenScorer — low-rank bilinear model that replaces attention-based
token importance scoring for KV cache eviction.

Architecture:
    score(q, k) = (W_k · k)^T · (W_q · q) / sqrt(d)

    where W_q ∈ R^{scorer_dim × hidden_dim}, W_k ∈ R^{scorer_dim × key_dim}

Inference trick:
    Pre-project keys once via project_keys(), then score new queries in
    O(seq_len × scorer_dim) instead of O(seq_len × hidden_dim).
"""

import math

import torch
import torch.nn as nn


class LearnedTokenScorer(nn.Module):
    """
    Parameters
    ----------
    hidden_dim : int
        Dimension of the query hidden states (e.g. 4096 for Llama-8B).
    key_dim : int
        Dimension of the key vectors per head (e.g. 1024 for Llama-8B GQA
        with num_kv_heads × head_dim = 8 × 128).
    scorer_dim : int, default=64
        Internal projection dimension. Controls capacity vs speed trade-off.
    """

    def __init__(self, hidden_dim: int, key_dim: int, scorer_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.scorer_dim = scorer_dim
        self.scale = 1.0 / math.sqrt(scorer_dim)

        self.q_proj = nn.Linear(hidden_dim, scorer_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, scorer_dim, bias=False)

        # Xavier init for stable training start
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)

    def forward(self, query_hidden: torch.Tensor, keys_flat: torch.Tensor) -> torch.Tensor:
        """
        Full forward: project both query and keys, then compute scores.

        Parameters
        ----------
        query_hidden : Tensor [batch, hidden_dim] or [hidden_dim]
            Hidden state of the current query token (after input_layernorm).
        keys_flat : Tensor [batch, seq_len, key_dim] or [seq_len, key_dim]
            Flattened key cache (num_kv_heads × head_dim concatenated).

        Returns
        -------
        scores : Tensor [batch, seq_len] or [seq_len]
            Per-token importance scores (higher = more important).
        """
        q = self.q_proj(query_hidden.float())   # [..., scorer_dim]
        k = self.k_proj(keys_flat.float())      # [..., seq_len, scorer_dim]

        if q.dim() == 1 and k.dim() == 2:
            return (k @ q) * self.scale         # [seq_len]
        elif q.dim() == 2 and k.dim() == 3:
            # q: [batch, scorer_dim] -> [batch, scorer_dim, 1]
            return torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.scale  # [batch, seq_len]
        else:
            raise ValueError(f"Unexpected shapes: query={query_hidden.shape}, keys={keys_flat.shape}")

    def project_keys(self, keys_flat: torch.Tensor) -> torch.Tensor:
        """
        Pre-project keys for efficient repeated scoring.

        Parameters
        ----------
        keys_flat : Tensor [..., key_dim]

        Returns
        -------
        projected : Tensor [..., scorer_dim]
        """
        return self.k_proj(keys_flat.float())

    def score_from_projected(self, query_hidden: torch.Tensor, projected_keys: torch.Tensor) -> torch.Tensor:
        """
        Score using pre-projected keys. This is the hot path during inference.

        Parameters
        ----------
        query_hidden : Tensor [batch, hidden_dim] or [hidden_dim]
        projected_keys : Tensor [batch, seq_len, scorer_dim] or [seq_len, scorer_dim]

        Returns
        -------
        scores : Tensor [batch, seq_len] or [seq_len]
        """
        q = self.q_proj(query_hidden.float())  # [..., scorer_dim]

        if q.dim() == 1 and projected_keys.dim() == 2:
            return (projected_keys @ q) * self.scale
        elif q.dim() == 2 and projected_keys.dim() == 3:
            return torch.bmm(projected_keys, q.unsqueeze(-1)).squeeze(-1) * self.scale
        else:
            raise ValueError(
                f"Unexpected shapes: query={query_hidden.shape}, projected_keys={projected_keys.shape}"
            )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
