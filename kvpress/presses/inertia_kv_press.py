# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


def _physical_weber_indices(seq_len: int, rom_budget: int, device: torch.device) -> torch.Tensor:
    """Weber's law for selecting ROM skeleton indices: denser near the end."""
    if seq_len <= rom_budget:
        return torch.arange(seq_len, device=device)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    distances = (seq_len - 1) - positions + 1.0
    weber_scores = 1.0 / torch.log(distances + 1.1)
    _, indices = torch.topk(weber_scores, rom_budget)
    return indices.sort().values


@dataclass
class InertiaKVPress(BasePress):
    """
    Dura-mTOVA: decoding-only press with Weber ROM and momentum eviction.

    Prefill does not compress.
    If compression_ratio > 0, dynamic budget is seq_len * (1 - compression_ratio).
    Otherwise total_budget is used as a fixed cap.
    """

    compression_ratio: float = 0.0
    total_budget: int = 2048
    rom_ratio: float = 0.3
    alpha: float = 0.8
    score_accumulation: str = "ema"
    use_first_large_evict_rescue: bool = True
    score_layer_mode: str = "all"
    score_layer_count: int = 1
    score_update_interval: int = 1
    score_update_warmup_steps: int = 1
    score_update_policy: str = "interval"
    refresh_min_interval: int = 1
    refresh_max_interval: int = 16
    refresh_query_drift_threshold: float = 0.10
    adaptive_alpha: bool = False
    adaptive_alpha_mode: str = "off"  # "off", "heuristic", "theory"
    adaptive_shift_policy: str = "raw_prev"
    alpha_low: float = 0.3
    alpha_high: float = 0.9
    adaptive_disagreement_threshold: float = 0.08
    adaptive_raw_ema_disagreement_threshold: float = 0.08
    adaptive_margin_threshold: float = 1e-3
    adaptive_shift_patience: int = 1
    adaptive_entropy_threshold: float = 0.9
    adaptive_use_entropy: bool = False
    adaptive_topk_cap: int = 512
    enable_profiling: bool = False
    profile_sync_cuda: bool = False
    enable_rank_tracing: bool = False
    boundary_sentinel_policy: str = "off"
    boundary_sentinel_guard_ratio: float = 0.10
    boundary_sentinel_blend_weight: float = 0.25
    boundary_sentinel_refresh_threshold: float = 0.35
    boundary_sentinel_refresh_patience: int = 1
    boundary_sentinel_topk_cap: int = 512
    boundary_sentinel_apply_on_refresh: bool = False
    boundary_sentinel_compute_fallback: bool = False
    head_pooling: str = "mean"
    head_pool_topk: int = 2
    head_pool_blend_weight: float = 0.25
    evidence_reserve_ratio: float = 0.0
    coverage_reserve_ratio: float = 0.0
    coverage_chunk_size: int = 128
    coverage_tokens_per_chunk: int = 2
    cumulative_blend_weight: float = 0.0
    cumulative_blend_policy: str = "always"
    cumulative_blend_margin_threshold: float = 0.05
    cumulative_blend_mode: str = "linear"
    cumulative_rescue_threshold: float = 1.0
    cumulative_normalization: str = "z"
    value_weighting: str = "none"  # "none", "vnorm", "vnorm_l1"
    layer_entropy_weighting: bool = False  # weight layer scores by attention entropy
    key_diversity_blend: float = 0.0  # blend weight for key diversity signal (0=off, >0 = blend)
    random_eviction: bool = False  # use random scores instead of attention (diagnostic)
    # Multi-scale EMA: conservative retention via max(fast, slow)
    multi_scale_ema: bool = False
    multi_scale_alpha_fast: float = 0.3
    multi_scale_alpha_slow: float = 0.95
    # Theory-derived adaptive alpha
    theory_warmup_steps: int = 30
    theory_alpha_candidates: tuple[float, ...] = (0.3, 0.8, 0.95)

    _initialized: bool = field(default=False, repr=False)
    _rom_indices: torch.Tensor | None = field(default=None, repr=False)
    _momentum: torch.Tensor | None = field(default=None, repr=False)
    _evidence_memory: torch.Tensor | None = field(default=None, repr=False)
    _cumulative_scores: torch.Tensor | None = field(default=None, repr=False)
    _token_ids: torch.Tensor | None = field(default=None, repr=False)
    _next_token_id: int = field(default=0, repr=False)
    _last_layer_idx: int = field(default=-1, repr=False)
    _dynamic_budget: int = field(default=0, repr=False)
    _compression_events: int = field(default=0, repr=False)
    _decode_steps: int = field(default=0, repr=False)
    _first_compress_step: int = field(default=-1, repr=False)
    _post_first_compress_cache_tokens: int = field(default=0, repr=False)
    _last_compressed_cache_tokens: int = field(default=0, repr=False)
    _profile_stats: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_run_stats: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _rank_trace: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)
    _last_rank_trace: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)
    _previous_raw_top_ids: set[int] | None = field(default=None, init=False, repr=False)
    _previous_ema_top_ids: set[int] | None = field(default=None, init=False, repr=False)
    _previous_adaptive_raw_top_ids: set[int] | None = field(default=None, init=False, repr=False)
    _active_alpha: float = field(default=0.8, init=False, repr=False)
    # Theory-derived adaptive alpha state
    _theory_score_diffs: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _theory_topk_sets: list[set[int]] = field(default_factory=list, init=False, repr=False)
    _theory_prev_scores: torch.Tensor | None = field(default=None, init=False, repr=False)
    _theory_alpha_selected: bool = field(default=False, init=False, repr=False)
    _theory_estimated_sigma: float = field(default=0.0, init=False, repr=False)
    _theory_estimated_delta: float = field(default=0.0, init=False, repr=False)
    _theory_eta: float = field(default=0.0, init=False, repr=False)
    _theory_layer_scores: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)  # per-layer scores within a decode step
    # Multi-scale EMA state
    _momentum_fast: torch.Tensor | None = field(default=None, init=False, repr=False)
    _momentum_slow: torch.Tensor | None = field(default=None, init=False, repr=False)
    _adaptive_alpha_sum: float = field(default=0.0, init=False, repr=False)
    _adaptive_alpha_count: int = field(default=0, init=False, repr=False)
    _adaptive_alpha_low_count: int = field(default=0, init=False, repr=False)
    _adaptive_alpha_mid_count: int = field(default=0, init=False, repr=False)
    _adaptive_alpha_high_count: int = field(default=0, init=False, repr=False)
    _adaptive_shift_count: int = field(default=0, init=False, repr=False)
    _adaptive_shift_streak: int = field(default=0, init=False, repr=False)
    _adaptive_uncertain_count: int = field(default=0, init=False, repr=False)
    _last_score_refresh_step: int = field(default=0, init=False, repr=False)
    _last_refresh_query: torch.Tensor | None = field(default=None, init=False, repr=False)
    _step_score_decision_step: int = field(default=-1, init=False, repr=False)
    _step_should_update_score: bool = field(default=True, init=False, repr=False)
    _refresh_interval_count: int = field(default=0, init=False, repr=False)
    _refresh_drift_count: int = field(default=0, init=False, repr=False)
    _refresh_query_drift_sum: float = field(default=0.0, init=False, repr=False)
    _refresh_query_drift_count: int = field(default=0, init=False, repr=False)
    _refresh_query_drift_max: float = field(default=0.0, init=False, repr=False)
    _boundary_sentinel_scores: torch.Tensor | None = field(default=None, init=False, repr=False)
    _boundary_sentinel_force_refresh: bool = field(default=False, init=False, repr=False)
    _boundary_sentinel_refresh_streak: int = field(default=0, init=False, repr=False)
    _boundary_sentinel_decisions: int = field(default=0, init=False, repr=False)
    _boundary_sentinel_applied_count: int = field(default=0, init=False, repr=False)
    _boundary_sentinel_forced_refresh_count: int = field(default=0, init=False, repr=False)
    _boundary_sentinel_disagreement_sum: float = field(default=0.0, init=False, repr=False)
    _boundary_sentinel_disagreement_count: int = field(default=0, init=False, repr=False)
    _boundary_sentinel_disagreement_max: float = field(default=0.0, init=False, repr=False)
    _boundary_sentinel_changed_keep_count: int = field(default=0, init=False, repr=False)
    _boundary_sentinel_changed_token_rate_sum: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        assert self.total_budget > 0
        assert 0 <= self.rom_ratio <= 1
        assert 0 <= self.alpha <= 1
        assert 0 <= self.alpha_low <= 1
        assert 0 <= self.alpha_high <= 1
        assert 0 <= self.compression_ratio < 1
        assert self.score_accumulation in {"ema", "sum", "max", "ema_sum"}, "score_accumulation must be 'ema', 'sum', 'max', or 'ema_sum'"
        assert self.score_layer_mode in {"all", "last", "last_k"}, (
            "score_layer_mode must be 'all', 'last', or 'last_k'"
        )
        assert self.score_layer_count > 0
        assert self.score_update_interval > 0
        assert self.score_update_warmup_steps > 0
        assert self.score_update_policy in {"interval", "interval_or_query_drift", "changepoint"}
        assert self.refresh_min_interval > 0
        assert self.refresh_query_drift_threshold >= 0
        assert self.adaptive_shift_policy in {"raw_prev", "strict", "strict_streak"}
        assert self.adaptive_disagreement_threshold >= 0
        assert self.adaptive_raw_ema_disagreement_threshold >= 0
        assert self.adaptive_margin_threshold >= 0
        assert self.adaptive_shift_patience > 0
        assert 0 <= self.adaptive_entropy_threshold <= 1
        assert self.adaptive_topk_cap > 0
        assert self.boundary_sentinel_policy in {
            "off",
            "guard",
            "blend",
            "guard_refresh",
            "blend_refresh",
        }
        assert self.boundary_sentinel_guard_ratio > 0
        assert 0 <= self.boundary_sentinel_blend_weight <= 1
        assert 0 <= self.boundary_sentinel_refresh_threshold <= 1
        assert self.boundary_sentinel_refresh_patience > 0
        assert self.boundary_sentinel_topk_cap > 0
        assert self.head_pooling in {
            "mean",
            "max",
            "topk_mean",
            "mean_topk_blend",
        }
        assert self.head_pool_topk > 0
        assert 0 <= self.head_pool_blend_weight <= 1
        assert 0 <= self.evidence_reserve_ratio < 1
        assert 0 <= self.coverage_reserve_ratio < 1
        assert self.coverage_chunk_size > 0
        assert self.coverage_tokens_per_chunk > 0
        assert self.cumulative_blend_weight >= 0
        assert self.cumulative_blend_policy in {"always", "margin"}
        assert self.cumulative_blend_margin_threshold >= 0
        assert self.cumulative_blend_mode in {"linear", "max", "relu"}
        assert self.cumulative_rescue_threshold >= 0
        assert self.cumulative_normalization in {"z", "robust_z"}
        assert self.adaptive_alpha_mode in {"off", "heuristic", "theory"}
        if self.adaptive_alpha or self.adaptive_alpha_mode == "heuristic":
            self.adaptive_alpha = True
            assert self.alpha_low <= self.alpha <= self.alpha_high
            assert self.score_accumulation == "ema"
        if self.adaptive_alpha_mode == "theory":
            assert self.score_accumulation == "ema"
            assert self.theory_warmup_steps > 0
            assert len(self.theory_alpha_candidates) >= 2
        self._active_alpha = self.alpha

    def post_init_from_model(self, model):
        from transformers import PreTrainedModel

        lm = getattr(model.model, "language_model", model.model)
        if not isinstance(model, PreTrainedModel):
            logger.warning(f"Model {type(model)} not tested")
        self._last_layer_idx = len(lm.layers) - 1

    def _profile_count(self, key: str, value: float = 1.0) -> None:
        if self.enable_profiling:
            self._profile_stats[key] = self._profile_stats.get(key, 0.0) + value

    def _profile_sync(self, device: torch.device) -> None:
        if self.profile_sync_cuda and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)

    def _profile_start(self, device: torch.device) -> float:
        if not self.enable_profiling:
            return 0.0
        self._profile_sync(device)
        return time.perf_counter()

    def _profile_add(self, key: str, start: float, device: torch.device) -> None:
        if not self.enable_profiling:
            return
        self._profile_sync(device)
        self._profile_stats[key] = self._profile_stats.get(key, 0.0) + time.perf_counter() - start

    def _current_run_stats(self) -> dict[str, float]:
        stats = dict(self._profile_stats)
        stats.update(
            {
                "compression_events": float(self._compression_events),
                "decode_steps": float(self._decode_steps),
                "first_compress_step": float(self._first_compress_step),
                "post_first_compress_cache_tokens": float(self._post_first_compress_cache_tokens),
                "last_compressed_cache_tokens": float(self._last_compressed_cache_tokens),
                "adaptive_alpha_mean": (
                    self._adaptive_alpha_sum / self._adaptive_alpha_count
                    if self._adaptive_alpha_count > 0
                    else float(self.alpha)
                ),
                "adaptive_alpha_decisions": float(self._adaptive_alpha_count),
                "adaptive_alpha_low_count": float(self._adaptive_alpha_low_count),
                "adaptive_alpha_mid_count": float(self._adaptive_alpha_mid_count),
                "adaptive_alpha_high_count": float(self._adaptive_alpha_high_count),
                "adaptive_shift_count": float(self._adaptive_shift_count),
                "adaptive_uncertain_count": float(self._adaptive_uncertain_count),
                "theory_estimated_sigma": float(self._theory_estimated_sigma),
                "theory_estimated_delta": float(self._theory_estimated_delta),
                "theory_eta": float(self._theory_eta),
                "theory_selected_alpha": float(self._active_alpha) if self._theory_alpha_selected else -1.0,
                "refresh_interval_count": float(self._refresh_interval_count),
                "refresh_drift_count": float(self._refresh_drift_count),
                "refresh_query_drift_mean": (
                    self._refresh_query_drift_sum / self._refresh_query_drift_count
                    if self._refresh_query_drift_count > 0
                    else 0.0
                ),
                "refresh_query_drift_max": float(self._refresh_query_drift_max),
                "boundary_sentinel_decisions": float(self._boundary_sentinel_decisions),
                "boundary_sentinel_applied_count": float(self._boundary_sentinel_applied_count),
                "boundary_sentinel_forced_refresh_count": float(
                    self._boundary_sentinel_forced_refresh_count
                ),
                "boundary_sentinel_disagreement_mean": (
                    self._boundary_sentinel_disagreement_sum
                    / self._boundary_sentinel_disagreement_count
                    if self._boundary_sentinel_disagreement_count > 0
                    else 0.0
                ),
                "boundary_sentinel_disagreement_max": float(
                    self._boundary_sentinel_disagreement_max
                ),
                "boundary_sentinel_changed_keep_count": float(
                    self._boundary_sentinel_changed_keep_count
                ),
                "boundary_sentinel_changed_token_rate_mean": (
                    self._boundary_sentinel_changed_token_rate_sum
                    / self._boundary_sentinel_changed_keep_count
                    if self._boundary_sentinel_changed_keep_count > 0
                    else 0.0
                ),
            }
        )
        return stats

    def last_run_summary(self) -> dict[str, float]:
        if self._last_run_stats:
            return dict(self._last_run_stats)
        return self._current_run_stats()

    def last_rank_trace(self) -> list[dict[str, float]]:
        if self._last_rank_trace:
            return list(self._last_rank_trace)
        return list(self._rank_trace)

    def _should_score_layer(self, layer_idx: int) -> bool:
        if self.score_layer_mode == "all":
            return True
        if self.score_layer_mode == "last":
            return layer_idx == self._last_layer_idx
        first_scored_layer = max(0, self._last_layer_idx - self.score_layer_count + 1)
        return layer_idx >= first_scored_layer

    def _score_decision_layer_idx(self) -> int:
        if self.score_layer_mode == "all":
            return 0
        if self.score_layer_mode == "last":
            return self._last_layer_idx
        return max(0, self._last_layer_idx - self.score_layer_count + 1)

    def _is_score_decision_layer(self, layer_idx: int) -> bool:
        return layer_idx == self._score_decision_layer_idx()

    def _extract_refresh_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = hidden_states[:, -1, :].detach().float().squeeze(0)
        if query.dim() > 1:
            query = query.reshape(-1)
        return query

    def _query_drift(self, current_query: torch.Tensor) -> float:
        if self._last_refresh_query is None or self._last_refresh_query.shape != current_query.shape:
            return 0.0
        denom = current_query.norm() * self._last_refresh_query.norm()
        if denom <= 0:
            return 0.0
        cosine = torch.dot(current_query, self._last_refresh_query) / denom.clamp_min(1e-12)
        cosine = cosine.clamp(-1.0, 1.0)
        return float((1.0 - cosine).item())

    def _decide_score_refresh(
        self, current_decode_step: int, hidden_states: torch.Tensor
    ) -> tuple[bool, str, float]:
        if current_decode_step <= 1:
            return True, "interval", 0.0

        if current_decode_step <= self.score_update_warmup_steps:
            current_query = self._extract_refresh_query(hidden_states)
            query_drift = self._query_drift(current_query)
            return True, "interval", query_drift

        steps_since_refresh = current_decode_step - self._last_score_refresh_step
        current_query = self._extract_refresh_query(hidden_states)
        query_drift = self._query_drift(current_query)

        if self._boundary_sentinel_force_refresh:
            self._boundary_sentinel_force_refresh = False
            return True, "sentinel", query_drift

        if steps_since_refresh >= self.score_update_interval:
            return True, "interval", query_drift

        if (
            self.score_update_policy == "interval_or_query_drift"
            and steps_since_refresh >= self.refresh_min_interval
            and query_drift >= self.refresh_query_drift_threshold
        ):
            return True, "drift", query_drift

        if self.score_update_policy == "changepoint":
            # Pure drift-triggered: refresh ONLY when query drift exceeds threshold
            # Safety-net: max_interval prevents starvation
            if query_drift >= self.refresh_query_drift_threshold:
                return True, "drift", query_drift
            if steps_since_refresh >= self.refresh_max_interval:
                return True, "interval", query_drift
            return False, "skip", query_drift

        return False, "skip", query_drift

    def _record_score_refresh_decision(
        self, current_decode_step: int, hidden_states: torch.Tensor
    ) -> bool:
        should_update_score, reason, query_drift = self._decide_score_refresh(current_decode_step, hidden_states)
        self._step_score_decision_step = current_decode_step
        self._step_should_update_score = should_update_score
        self._refresh_query_drift_sum += query_drift
        self._refresh_query_drift_count += 1
        self._refresh_query_drift_max = max(self._refresh_query_drift_max, query_drift)
        if should_update_score:
            self._last_score_refresh_step = current_decode_step
            self._last_refresh_query = self._extract_refresh_query(hidden_states)
            if reason == "drift":
                self._refresh_drift_count += 1
                self._profile_count("refresh_drift_count")
            elif reason == "sentinel":
                self._boundary_sentinel_forced_refresh_count += 1
                self._profile_count("refresh_sentinel_count")
            else:
                self._refresh_interval_count += 1
                self._profile_count("refresh_interval_count")
        else:
            self._profile_count("refresh_skip_count")
        return should_update_score

    def _update_importance_scores(self, step_scores: torch.Tensor, entropy_weight: float = 1.0) -> bool:
        if step_scores.shape[0] != self._momentum.shape[0]:
            return False
        # Ensure step_scores is on the same device as momentum (multi-GPU compatibility)
        if step_scores.device != self._momentum.device:
            step_scores = step_scores.to(self._momentum.device)
        if self.score_accumulation == "sum":
            self._momentum = self._momentum + step_scores * entropy_weight
        elif self.score_accumulation == "max":
            # Keep the maximum score ever seen for each token
            self._momentum = torch.maximum(self._momentum, step_scores)
        elif self.score_accumulation == "ema_sum":
            # EMA + cumulative: combine EMA smoothing with additive memory
            use_adaptive = self.adaptive_alpha or self.adaptive_alpha_mode == "theory"
            alpha = self._active_alpha if use_adaptive else self.alpha
            self._momentum = alpha * self._momentum + step_scores
        else:
            use_adaptive = self.adaptive_alpha or self.adaptive_alpha_mode == "theory"
            alpha = self._active_alpha if use_adaptive else self.alpha
            # Scale the new score contribution by entropy weight
            effective_weight = (1.0 - alpha) * entropy_weight
            self._momentum = (1.0 - effective_weight) * self._momentum + effective_weight * step_scores
        # Multi-scale EMA: update fast and slow momentum in parallel
        if self.multi_scale_ema and self._momentum_fast is not None and step_scores.shape[0] == self._momentum_fast.shape[0]:
            af = self.multi_scale_alpha_fast
            self._momentum_fast = af * self._momentum_fast + (1.0 - af) * step_scores
            asl = self.multi_scale_alpha_slow
            self._momentum_slow = asl * self._momentum_slow + (1.0 - asl) * step_scores
        if self.evidence_reserve_ratio > 0 and self._evidence_memory is not None:
            self._evidence_memory = self._evidence_memory + step_scores.detach().to(
                device=self._evidence_memory.device,
                dtype=self._evidence_memory.dtype,
            )
        if self.cumulative_blend_weight > 0 and self._cumulative_scores is not None:
            self._cumulative_scores = self._cumulative_scores + step_scores.detach().to(
                device=self._cumulative_scores.device,
                dtype=self._cumulative_scores.dtype,
            )
        return True

    def _topk_mask(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        k = max(1, min(k, scores.numel()))
        indices = torch.topk(scores.detach().float(), k=k, largest=True).indices
        mask = torch.zeros(scores.numel(), device=scores.device, dtype=torch.bool)
        mask[indices] = True
        return mask

    def _topk_jaccard_tensor(self, left: torch.Tensor, right: torch.Tensor, k: int) -> float:
        if left.numel() == 0 or right.numel() == 0 or left.shape[0] != right.shape[0]:
            return float("nan")
        left_mask = self._topk_mask(left, k)
        right_mask = self._topk_mask(right, k)
        union = torch.logical_or(left_mask, right_mask).sum().item()
        if union == 0:
            return float("nan")
        intersection = torch.logical_and(left_mask, right_mask).sum().item()
        return float(intersection / union)

    def _boundary_margin(self, scores: torch.Tensor) -> float:
        if scores.numel() <= 1 or self._dynamic_budget <= 0 or self._dynamic_budget >= scores.numel():
            return float("inf")
        k = min(self._dynamic_budget + 1, scores.numel())
        boundary = torch.topk(scores.detach().float(), k=k, largest=True).values
        keep_score = boundary[-2].abs().clamp_min(1e-12)
        margin = (boundary[-2] - boundary[-1]).clamp_min(0.0) / keep_score
        return float(margin.item())

    def _score_entropy(self, scores: torch.Tensor) -> float:
        positive = scores.detach().float().clamp_min(0.0)
        total = positive.sum()
        if total <= 0 or positive.numel() <= 1:
            return 1.0
        probs = positive / total
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum() / torch.log(
            torch.tensor(float(probs.numel()), device=probs.device)
        )
        return float(entropy.item())

    def _select_adaptive_alpha(self, step_scores: torch.Tensor) -> tuple[float, dict[str, float]]:
        if self._momentum is None or step_scores.shape[0] != self._momentum.shape[0]:
            return self.alpha, {}

        metric_k = max(1, min(self._dynamic_budget, self.adaptive_topk_cap, step_scores.numel()))
        current_raw_top_ids = self._top_token_id_set(step_scores.detach().float(), metric_k)
        raw_prev_jaccard = self._jaccard(self._previous_adaptive_raw_top_ids, current_raw_top_ids)
        raw_ema_jaccard = self._topk_jaccard_tensor(step_scores, self._momentum, metric_k)
        has_previous_raw = raw_prev_jaccard == raw_prev_jaccard
        raw_prev_disagreement = 1.0 - raw_prev_jaccard if has_previous_raw else 0.0
        raw_ema_disagreement = 1.0 - raw_ema_jaccard if raw_ema_jaccard == raw_ema_jaccard else 0.0
        disagreement = raw_prev_disagreement if has_previous_raw else raw_ema_disagreement
        margin = self._boundary_margin(step_scores)
        entropy = self._score_entropy(step_scores) if self.adaptive_use_entropy else 0.0

        if self.adaptive_shift_policy in {"strict", "strict_streak"}:
            confident_shift = (
                has_previous_raw
                and raw_prev_disagreement >= self.adaptive_disagreement_threshold
                and raw_ema_disagreement >= self.adaptive_raw_ema_disagreement_threshold
                and margin >= self.adaptive_margin_threshold
            )
            stable = (
                has_previous_raw
                and raw_prev_disagreement < 0.5 * self.adaptive_disagreement_threshold
                and raw_ema_disagreement < 0.5 * self.adaptive_raw_ema_disagreement_threshold
            )
        else:
            confident_shift = (
                disagreement >= self.adaptive_disagreement_threshold
                and margin >= self.adaptive_margin_threshold
            )
            stable = disagreement < 0.5 * self.adaptive_disagreement_threshold
        if self.adaptive_use_entropy:
            confident_shift = confident_shift and entropy < self.adaptive_entropy_threshold

        uncertain = margin < self.adaptive_margin_threshold
        if self.adaptive_use_entropy:
            uncertain = uncertain or entropy >= self.adaptive_entropy_threshold

        shift_streak = self._adaptive_shift_streak + 1 if confident_shift else 0
        confirmed_shift = (
            confident_shift
            and (
                self.adaptive_shift_policy != "strict_streak"
                or shift_streak >= self.adaptive_shift_patience
            )
        )

        if confirmed_shift:
            alpha = self.alpha_low
            decision = "low"
        elif uncertain or stable:
            alpha = self.alpha_high
            decision = "high"
        else:
            alpha = self.alpha
            decision = "mid"

        metrics = {
            "raw_prev_jaccard": raw_prev_jaccard,
            "raw_ema_jaccard": raw_ema_jaccard,
            "raw_prev_disagreement": raw_prev_disagreement,
            "raw_ema_disagreement": raw_ema_disagreement,
            "disagreement": disagreement,
            "boundary_margin": margin,
            "entropy": entropy,
            "shift_detected": float(confident_shift),
            "shift_confirmed": float(confirmed_shift),
            "shift_streak": float(shift_streak),
            "uncertain": float(uncertain),
            "alpha": alpha,
            "decision": decision,
        }
        return alpha, metrics

    def _update_next_adaptive_alpha(self, step_scores: torch.Tensor) -> None:
        if not self.adaptive_alpha:
            return
        if self._momentum is None or step_scores.shape[0] != self._momentum.shape[0]:
            return
        alpha, metrics = self._select_adaptive_alpha(step_scores)
        self._active_alpha = alpha
        metric_k = max(1, min(self._dynamic_budget, self.adaptive_topk_cap, step_scores.numel()))
        self._previous_adaptive_raw_top_ids = self._top_token_id_set(step_scores.detach().float(), metric_k)
        self._adaptive_alpha_sum += alpha
        self._adaptive_alpha_count += 1
        if metrics.get("decision") == "low":
            self._adaptive_alpha_low_count += 1
        elif metrics.get("decision") == "high":
            self._adaptive_alpha_high_count += 1
        else:
            self._adaptive_alpha_mid_count += 1
        if metrics.get("shift_detected", 0.0):
            self._adaptive_shift_streak = int(metrics.get("shift_streak", 1.0))
        else:
            self._adaptive_shift_streak = 0
        self._adaptive_shift_count += int(metrics.get("shift_confirmed", 0.0))
        self._adaptive_uncertain_count += int(metrics.get("uncertain", 0.0))
        self._profile_count(f"adaptive_alpha_{metrics.get('decision', 'unknown')}_decisions")

    # ---- Theory-derived adaptive alpha ----

    def _collect_theory_statistics(self, step_scores: torch.Tensor) -> None:
        """Collect score statistics during warmup for σ/δ estimation.

        Cross-layer σ: mean pairwise Jaccard distance between layers' top-k sets
        at this decode step, measuring how much layers disagree about retention.
        Temporal drift δ: from top-k set Jaccard distance over decode steps.
        Both σ and δ are Jaccard distances on [0,1], making them directly comparable.
        """
        if self._theory_alpha_selected:
            return
        scores = step_scores.detach().float()

        # Compute cross-layer Jaccard disagreement from per-layer scores
        if len(self._theory_layer_scores) >= 2:
            ref_size = self._theory_layer_scores[-1].shape[0]
            matching = [s for s in self._theory_layer_scores if s.shape[0] == ref_size]
            if len(matching) >= 2:
                k = max(1, min(self._dynamic_budget, ref_size))
                # Compute top-k set for each layer
                layer_topk_sets = []
                for ls in matching:
                    topk_ids = set(ls.topk(k).indices.tolist())
                    layer_topk_sets.append(topk_ids)
                # Mean pairwise Jaccard distance (sample to limit compute)
                import random
                pairs = []
                n_layers = len(layer_topk_sets)
                if n_layers <= 10:
                    # All pairs
                    for i in range(n_layers):
                        for j in range(i + 1, n_layers):
                            pairs.append((i, j))
                else:
                    # Sample ~30 pairs
                    for _ in range(30):
                        i, j = random.sample(range(n_layers), 2)
                        pairs.append((i, j))
                jaccards = []
                for i, j in pairs:
                    jac = self._jaccard(layer_topk_sets[i], layer_topk_sets[j])
                    if jac == jac:  # not NaN
                        jaccards.append(1.0 - jac)  # distance = 1 - similarity
                if jaccards:
                    mean_disagreement = sum(jaccards) / len(jaccards)
                    self._theory_score_diffs.append(mean_disagreement)  # store as float

        # Clear per-layer scores for next decode step
        self._theory_layer_scores.clear()

        self._theory_prev_scores = scores.clone()

        # Collect top-k sets for drift estimation
        k = max(1, min(self._dynamic_budget, scores.numel()))
        topk_ids = self._top_token_id_set(scores, k)
        self._theory_topk_sets.append(topk_ids)

    def _estimate_noise_drift(self) -> tuple[float, float]:
        """Estimate noise σ and drift rate δ from warmup statistics.

        σ (noise): Mean cross-layer Jaccard distance. Each decode step produces
            a mean pairwise Jaccard distance between layers' top-k sets. We
            average these across warmup steps. On [0, 1] scale.

        δ (drift): Slope of cumulative Jaccard distance between time step 0
            and time step t. On [0, 1] per-step scale.

        Both are Jaccard distances, directly comparable. η = σ/δ.
        """
        import math

        # --- Estimate σ from cross-layer Jaccard disagreement ---
        sigma = 0.0
        if len(self._theory_score_diffs) >= 1:
            # _theory_score_diffs stores mean cross-layer Jaccard distance per step (floats)
            sigma = sum(self._theory_score_diffs) / len(self._theory_score_diffs)

        # --- Estimate δ (drift rate) ---
        delta = 0.0
        if len(self._theory_topk_sets) >= 3:
            initial_set = self._theory_topk_sets[0]
            cumulative_distances = []
            for t, topk_set in enumerate(self._theory_topk_sets[1:], 1):
                jaccard = self._jaccard(initial_set, topk_set)
                if jaccard == jaccard:  # not NaN
                    cumulative_distances.append((t, 1.0 - jaccard))

            if len(cumulative_distances) >= 2:
                # Linear regression: J_cum = δ * t + b
                ts = torch.tensor([t for t, _ in cumulative_distances], dtype=torch.float)
                js = torch.tensor([j for _, j in cumulative_distances], dtype=torch.float)
                t_mean = ts.mean()
                j_mean = js.mean()
                # Slope = Cov(t, j) / Var(t)
                cov = ((ts - t_mean) * (js - j_mean)).sum()
                var_t = ((ts - t_mean) ** 2).sum()
                if var_t > 0:
                    delta = max(0.0, float((cov / var_t).item()))

        sigma = max(sigma, 1e-12)
        delta = max(delta, 1e-12)

        self._theory_estimated_sigma = sigma
        self._theory_estimated_delta = delta

        return sigma, delta

    def _theory_select_alpha(self) -> float:
        """Select α from candidates based on noise-to-drift ratio η = σ/δ.

        From the variance-lag tradeoff:
        - High η: noise dominates → need high α for variance reduction
        - Low η: drift dominates → need low α for adaptivity

        The mapping is derived from balancing boundary error (Eq.5) against
        lag bias. For each candidate α, the total expected error is:
          E(α) = P_err(α) + λ · lag(α)
        where P_err ∝ Φ(-η·√((1+α)/(1-α))) and lag ∝ α/(1-α)·δ.

        We select the α that minimizes the estimated total error.
        """
        import math
        from scipy.stats import norm  # type: ignore

        sigma, delta = self._estimate_noise_drift()
        eta = sigma / delta
        self._theory_eta = eta

        candidates = sorted(self.theory_alpha_candidates)
        best_alpha = self.alpha  # fallback
        best_score = float("inf")

        for alpha_c in candidates:
            if alpha_c <= 0 or alpha_c >= 1:
                continue
            # Boundary error term: effective SNR amplification
            snr_factor = math.sqrt((1.0 + alpha_c) / (1.0 - alpha_c))
            # P_err ∝ Φ(-η * snr_factor / √2) — lower is better
            # Use a representative gap Δ/σ = 1 (tokens at 1σ separation)
            boundary_error = norm.cdf(-snr_factor / math.sqrt(2.0))

            # Lag term: steady-state bias = α·δ/(1-α), normalized by σ
            lag_bias = (alpha_c * delta) / ((1.0 - alpha_c) * sigma)

            # Total error: boundary_error + lag_bias
            # Both are dimensionless probabilities/ratios
            total = boundary_error + lag_bias
            if total < best_score:
                best_score = total
                best_alpha = alpha_c

        self._active_alpha = best_alpha
        self._theory_alpha_selected = True

        # Clean up warmup state to free memory
        self._theory_score_diffs.clear()
        self._theory_topk_sets.clear()
        self._theory_prev_scores = None
        self._theory_layer_scores.clear()

        logger.info(
            f"Theory adaptive α: σ={sigma:.4f}, δ={delta:.6f}, η={eta:.2f} → α={best_alpha}"
        )
        print(f"[THEORY] σ={sigma:.4f}, δ={delta:.6f}, η={eta:.2f} → α={best_alpha}")
        return best_alpha

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return keys, values

    def _init_dura_state(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self.compression_ratio > 0:
            self._dynamic_budget = max(1, int(seq_len * (1 - self.compression_ratio)))
        else:
            self._dynamic_budget = self.total_budget

        rom_budget = max(1, int(self._dynamic_budget * self.rom_ratio))
        rom_budget = min(rom_budget, seq_len)
        self._rom_indices = _physical_weber_indices(seq_len, rom_budget, device)
        self._momentum = torch.zeros(seq_len, device=device, dtype=dtype)
        if self.multi_scale_ema:
            self._momentum_fast = torch.zeros(seq_len, device=device, dtype=dtype)
            self._momentum_slow = torch.zeros(seq_len, device=device, dtype=dtype)
        self._evidence_memory = torch.zeros(seq_len, device=device, dtype=dtype)
        self._cumulative_scores = torch.zeros(seq_len, device=device, dtype=dtype)
        self._token_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        self._next_token_id = seq_len
        self._initialized = True

    @staticmethod
    def _jaccard(left: set[int] | None, right: set[int]) -> float:
        if left is None:
            return float("nan")
        union = len(left | right)
        if union == 0:
            return float("nan")
        return len(left & right) / union

    def _top_token_id_set(self, scores: torch.Tensor, k: int) -> set[int]:
        if self._token_ids is None or scores.numel() == 0:
            return set()
        k = max(1, min(k, scores.numel()))
        top_indices = torch.topk(scores, k=k, largest=True).indices
        top_ids = self._token_ids.index_select(0, top_indices).detach().cpu().tolist()
        return {int(token_id) for token_id in top_ids}

    def _record_rank_trace(self, step_scores: torch.Tensor, keep_indices: torch.Tensor | None = None) -> None:
        if not self.enable_rank_tracing or self._token_ids is None or self._momentum is None:
            return
        if step_scores.shape[0] != self._momentum.shape[0]:
            return

        k = max(1, min(self._dynamic_budget, step_scores.shape[0]))
        raw_top_ids = self._top_token_id_set(step_scores.detach().float(), k)
        ema_top_ids = self._top_token_id_set(self._momentum.detach().float(), k)
        raw_prev_jaccard = self._jaccard(self._previous_raw_top_ids, raw_top_ids)
        ema_prev_jaccard = self._jaccard(self._previous_ema_top_ids, ema_top_ids)
        raw_ema_jaccard = self._jaccard(raw_top_ids, ema_top_ids)

        record = {
            "decode_step": float(self._decode_steps),
            "cache_tokens_before_evict": float(step_scores.shape[0]),
            "topk": float(k),
            "raw_topk_prev_jaccard": raw_prev_jaccard,
            "ema_topk_prev_jaccard": ema_prev_jaccard,
            "raw_topk_churn": 1.0 - raw_prev_jaccard if raw_prev_jaccard == raw_prev_jaccard else float("nan"),
            "ema_topk_churn": 1.0 - ema_prev_jaccard if ema_prev_jaccard == ema_prev_jaccard else float("nan"),
            "raw_ema_topk_jaccard": raw_ema_jaccard,
            "boundary_margin": self._boundary_margin(step_scores),
            "score_entropy": self._score_entropy(step_scores),
            "active_alpha": float(self._active_alpha if self.adaptive_alpha else self.alpha),
        }

        if keep_indices is not None:
            keep_ids = set(
                int(token_id)
                for token_id in self._token_ids.index_select(0, keep_indices).detach().cpu().tolist()
            )
            record["raw_topk_evicted_rate"] = (
                len(raw_top_ids - keep_ids) / len(raw_top_ids) if raw_top_ids else float("nan")
            )
            record["ema_topk_evicted_rate"] = (
                len(ema_top_ids - keep_ids) / len(ema_top_ids) if ema_top_ids else float("nan")
            )

        self._rank_trace.append(record)
        self._previous_raw_top_ids = raw_top_ids
        self._previous_ema_top_ids = ema_top_ids

    def _compute_last_query_attention(
        self, module: nn.Module, hidden_states: torch.Tensor, keys: torch.Tensor, kwargs: dict,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from kvpress.presses.snapkv_press import SnapKVPress

        # Ensure all tensors are on the same device (multi-GPU compatibility)
        target_device = keys.device
        if hidden_states.device != target_device:
            hidden_states = hidden_states.to(target_device)
        position_embeddings = kwargs["position_embeddings"]
        if isinstance(position_embeddings, (tuple, list)):
            position_embeddings = tuple(p.to(target_device) if p.device != target_device else p for p in position_embeddings)
        else:
            if position_embeddings.device != target_device:
                position_embeddings = position_embeddings.to(target_device)

        attn_weights = SnapKVPress.compute_window_attention(
            module, hidden_states, keys, 1, position_embeddings
        )
        attn_last = attn_weights[:, :, -1, :]  # [batch, num_heads, kv_len]

        # Apply value weighting if configured (same logic as _scores_from_attention)
        if self.value_weighting != "none" and values is not None:
            if values.device != target_device:
                values = values.to(target_device)
            if self.value_weighting == "vnorm_l1":
                v_norm = values.norm(p=1, dim=-1)  # [batch, num_kv_heads, kv_len]
            else:  # "vnorm" (L2)
                v_norm = values.norm(p=2, dim=-1)
            # Truncate v_norm to match attn_last length (compute_window_attention strips last window_size tokens)
            attn_seq_len = attn_last.shape[-1]
            v_norm = v_norm[:, :, :attn_seq_len]
            # Expand v_norm to match attention heads if GQA
            num_attn_heads = attn_last.shape[1]
            num_kv_heads = v_norm.shape[1]
            if num_kv_heads < num_attn_heads:
                repeats = num_attn_heads // num_kv_heads
                v_norm = v_norm.repeat_interleave(repeats, dim=1)
            # Normalize v_norm per head to avoid scale issues
            v_norm = v_norm / (v_norm.mean(dim=-1, keepdim=True) + 1e-8)
            attn_last = attn_last * v_norm

        step_scores = self._pool_attention_heads(attn_last)
        step_scores = torch.cat(
            [step_scores, torch.tensor([step_scores.max()], device=step_scores.device, dtype=step_scores.dtype)]
        )
        return step_scores

    def _pool_attention_heads(self, head_scores: torch.Tensor) -> torch.Tensor:
        # head_scores: [batch, num_heads, kv_len] or [num_heads, kv_len].
        if head_scores.dim() == 3:
            head_scores = head_scores.squeeze(0)
        if head_scores.dim() == 1:
            return head_scores
        if head_scores.dim() > 2:
            head_scores = head_scores.reshape(-1, head_scores.shape[-1])

        if self.head_pooling == "max":
            return head_scores.max(dim=0).values

        if self.head_pooling in {"topk_mean", "mean_topk_blend"}:
            topk = min(self.head_pool_topk, head_scores.shape[0])
            topk_scores = torch.topk(head_scores, k=topk, dim=0, largest=True).values.mean(dim=0)
            if self.head_pooling == "topk_mean":
                return topk_scores
            mean_scores = head_scores.mean(dim=0)
            weight = self.head_pool_blend_weight
            return (1.0 - weight) * mean_scores + weight * topk_scores

        return head_scores.mean(dim=0)

    def _scores_from_attention(
        self, attentions: torch.Tensor | None, device: torch.device, dtype: torch.dtype,
        values: torch.Tensor | None = None, keys: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if attentions is None:
            return None
        attn_last = attentions[:, :, -1, :]  # [batch, num_heads, kv_len]
        if self.value_weighting != "none" and values is not None:
            # values shape: [batch, num_kv_heads, kv_len, head_dim]
            if self.value_weighting == "vnorm_l1":
                v_norm = values.norm(p=1, dim=-1)  # [batch, num_kv_heads, kv_len]
            else:  # "vnorm" (L2)
                v_norm = values.norm(p=2, dim=-1)  # [batch, num_kv_heads, kv_len]
            # Expand v_norm to match attention heads if GQA
            num_attn_heads = attn_last.shape[1]
            num_kv_heads = v_norm.shape[1]
            if num_kv_heads < num_attn_heads:
                repeats = num_attn_heads // num_kv_heads
                v_norm = v_norm.repeat_interleave(repeats, dim=1)
            # Normalize v_norm per head to avoid scale issues
            v_norm = v_norm / (v_norm.mean(dim=-1, keepdim=True) + 1e-8)
            attn_last = attn_last * v_norm
        step_scores = self._pool_attention_heads(attn_last)

        # Blend with key diversity signal: tokens whose keys differ from mean get higher scores
        if self.key_diversity_blend > 0 and keys is not None:
            # keys shape: [batch, num_kv_heads, kv_len, head_dim]
            # Compute mean key across kv positions
            mean_key = keys.mean(dim=2, keepdim=True)  # [batch, num_kv_heads, 1, head_dim]
            # Cosine distance: 1 - cosine_similarity (higher = more diverse = keep)
            cos_sim = torch.nn.functional.cosine_similarity(keys, mean_key, dim=-1)  # [batch, num_kv_heads, kv_len]
            diversity = (1.0 - cos_sim).mean(dim=(0, 1))  # [kv_len] avg over batch and heads
            # Z-normalize both signals before blending
            attn_z = self._z_normalize(step_scores)
            div_z = self._z_normalize(diversity.to(device=device, dtype=dtype))
            step_scores = (1.0 - self.key_diversity_blend) * attn_z + self.key_diversity_blend * div_z
            step_scores = step_scores.to(device=device, dtype=dtype)

        return step_scores.to(device=device, dtype=dtype)

    def _z_normalize(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.detach().float()
        std = scores.std(unbiased=False)
        if std <= 0:
            return torch.zeros_like(scores)
        return (scores - scores.mean()) / std.clamp_min(1e-12)

    def _robust_z_normalize(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.detach().float()
        median = scores.median()
        mad = (scores - median).abs().median()
        if mad > 0:
            return (scores - median) / (1.4826 * mad).clamp_min(1e-12)
        return self._z_normalize(scores)

    def _normalize_for_cumulative_blend(self, scores: torch.Tensor) -> torch.Tensor:
        if self.cumulative_normalization == "robust_z":
            return self._robust_z_normalize(scores)
        return self._z_normalize(scores)

    def _apply_cumulative_blend(self, protected: torch.Tensor) -> torch.Tensor:
        if self.cumulative_blend_weight <= 0:
            return protected
        if self._cumulative_scores is None or self._cumulative_scores.shape != protected.shape:
            return protected
        if (
            self.cumulative_blend_policy == "margin"
            and self._boundary_margin(protected) > self.cumulative_blend_margin_threshold
        ):
            return protected

        base = self._normalize_for_cumulative_blend(protected)
        cumulative = self._normalize_for_cumulative_blend(self._cumulative_scores.to(device=protected.device))
        if self.cumulative_blend_mode == "max":
            return torch.maximum(base, self.cumulative_blend_weight * cumulative)
        if self.cumulative_blend_mode == "relu":
            rescue = torch.relu(cumulative - self.cumulative_rescue_threshold)
            return base + self.cumulative_blend_weight * rescue
        return base + self.cumulative_blend_weight * cumulative

    def _apply_boundary_sentinel(self, protected: torch.Tensor, sentinel_scores: torch.Tensor) -> torch.Tensor:
        if self.boundary_sentinel_policy == "off":
            return protected
        if sentinel_scores.shape != protected.shape or protected.numel() == 0:
            return protected

        policy = self.boundary_sentinel_policy
        if policy in {"blend", "blend_refresh"}:
            weight = self.boundary_sentinel_blend_weight
            blended = (1.0 - weight) * self._z_normalize(protected) + weight * self._z_normalize(
                sentinel_scores
            )
            return blended.to(dtype=protected.dtype, device=protected.device)

        guard_k = max(1, int(round(self._dynamic_budget * self.boundary_sentinel_guard_ratio)))
        guard_k = min(guard_k, protected.numel())
        budget_k = max(1, min(self._dynamic_budget, protected.numel()))
        sentinel_top = torch.topk(sentinel_scores.detach().float(), k=guard_k, largest=True).indices
        boundary_value = torch.topk(protected.detach().float(), k=budget_k, largest=True).values[-1]
        epsilon = boundary_value.abs().clamp_min(1e-6) * 1e-3
        guarded = protected.clone()
        guarded[sentinel_top] = torch.maximum(
            guarded[sentinel_top].detach().float(), boundary_value + epsilon
        ).to(dtype=protected.dtype)
        return guarded

    def _record_boundary_sentinel_state(self, sentinel_scores: torch.Tensor | None) -> None:
        if self.boundary_sentinel_policy == "off":
            return
        if sentinel_scores is None or self._momentum is None or sentinel_scores.shape != self._momentum.shape:
            return

        k = max(1, min(self._dynamic_budget, self.boundary_sentinel_topk_cap, sentinel_scores.numel()))
        jaccard = self._topk_jaccard_tensor(sentinel_scores, self._momentum, k)
        if jaccard != jaccard:
            return
        disagreement = 1.0 - jaccard
        self._boundary_sentinel_decisions += 1
        self._boundary_sentinel_disagreement_sum += disagreement
        self._boundary_sentinel_disagreement_count += 1
        self._boundary_sentinel_disagreement_max = max(
            self._boundary_sentinel_disagreement_max, disagreement
        )
        self._profile_count("boundary_sentinel_decisions")

        if not self.boundary_sentinel_policy.endswith("_refresh"):
            self._boundary_sentinel_refresh_streak = 0
            return
        if self._step_should_update_score:
            self._boundary_sentinel_refresh_streak = 0
            return

        if disagreement >= self.boundary_sentinel_refresh_threshold:
            self._boundary_sentinel_refresh_streak += 1
        else:
            self._boundary_sentinel_refresh_streak = 0
        if self._boundary_sentinel_refresh_streak >= self.boundary_sentinel_refresh_patience:
            self._boundary_sentinel_force_refresh = True
            self._boundary_sentinel_refresh_streak = 0

    def _keep_mask(self, keep_indices: torch.Tensor, length: int) -> torch.Tensor:
        mask = torch.zeros(length, device=keep_indices.device, dtype=torch.bool)
        mask[keep_indices] = True
        return mask

    def _select_coverage_indices(self, scores: torch.Tensor, target_count: int) -> torch.Tensor:
        target_count = max(0, min(target_count, scores.shape[0]))
        if target_count == 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)

        candidate_scores = scores.detach().float().clone()
        candidate_scores[self._rom_indices] = float("-inf")
        candidate_scores[-1] = float("-inf")

        finite_mask = torch.isfinite(candidate_scores)
        if not bool(finite_mask.any()):
            return torch.empty(0, device=scores.device, dtype=torch.long)

        seq_len = candidate_scores.shape[0]
        chunk_size = min(self.coverage_chunk_size, seq_len)
        chunk_starts = torch.arange(0, seq_len, chunk_size, device=scores.device)
        chunk_scores = []
        for start in chunk_starts.tolist():
            end = min(start + chunk_size, seq_len)
            chunk = candidate_scores[start:end]
            best = torch.max(chunk, dim=0)
            chunk_scores.append(best.values)

        chunk_scores_tensor = torch.stack(chunk_scores)
        valid_chunks = torch.isfinite(chunk_scores_tensor)
        if not bool(valid_chunks.any()):
            return torch.empty(0, device=scores.device, dtype=torch.long)

        chunks_needed = (target_count + self.coverage_tokens_per_chunk - 1) // self.coverage_tokens_per_chunk
        chunks_needed = min(chunks_needed, int(valid_chunks.sum().item()))
        top_chunks = torch.topk(chunk_scores_tensor, k=chunks_needed, largest=True).indices

        selected: list[torch.Tensor] = []
        selected_count = 0
        selected_mask = torch.zeros(seq_len, device=scores.device, dtype=torch.bool)
        for chunk_idx in top_chunks.tolist():
            start = int(chunk_starts[chunk_idx].item())
            end = min(start + chunk_size, seq_len)
            chunk = candidate_scores[start:end].clone()
            take = min(self.coverage_tokens_per_chunk, target_count - selected_count)
            if take <= 0:
                break
            valid = torch.isfinite(chunk)
            if not bool(valid.any()):
                continue
            take = min(take, int(valid.sum().item()))
            local = torch.topk(chunk, k=take, largest=True).indices + start
            selected.append(local)
            selected_count += int(local.numel())
            selected_mask[local] = True

        if selected:
            indices = torch.unique(torch.cat(selected), sorted=False)
        else:
            indices = torch.empty(0, device=scores.device, dtype=torch.long)

        if indices.numel() < target_count:
            fill_scores = candidate_scores.clone()
            fill_scores[selected_mask] = float("-inf")
            available = int(torch.isfinite(fill_scores).sum().item())
            missing = min(target_count - indices.numel(), available)
            if missing > 0:
                fill = torch.topk(fill_scores, k=missing, largest=True).indices
                indices = torch.cat([indices, fill])

        if indices.numel() > target_count:
            _, order = torch.topk(candidate_scores[indices], k=target_count, largest=True)
            indices = indices[order]

        return indices.sort().values

    def _batch_evict_and_update_cache(self, cache, keep_indices: torch.Tensor):
        device = self._momentum.device
        original_len = self._momentum.shape[0]

        for layer_idx in range(len(cache.layers)):
            keys, values = extract_keys_and_values(cache, layer_idx)
            # Ensure keep_indices is on the same device as keys (multi-GPU compatibility)
            layer_keep = keep_indices.to(keys.device) if keep_indices.device != keys.device else keep_indices
            k = keys.index_select(2, layer_keep)
            v = values.index_select(2, layer_keep)
            cache_layer = cache.layers[layer_idx]
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(k, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(v, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=k.dtype, device=k.device)
                cache_layer.values = torch.zeros(0, dtype=v.dtype, device=v.device)
                cache_layer.cumulative_length = k.shape[2]
            else:
                cache_layer.keys = k
                cache_layer.values = v
            self._profile_count("cache_update_layers")

        old_to_new = torch.full((original_len,), -1, device=device, dtype=torch.long)
        old_to_new[keep_indices] = torch.arange(len(keep_indices), device=device)
        new_rom = old_to_new[self._rom_indices]
        self._rom_indices = new_rom[new_rom >= 0]
        self._momentum = self._momentum[keep_indices]
        if self._momentum_fast is not None:
            self._momentum_fast = self._momentum_fast[keep_indices]
        if self._momentum_slow is not None:
            self._momentum_slow = self._momentum_slow[keep_indices]
        if self._evidence_memory is not None:
            self._evidence_memory = self._evidence_memory[keep_indices]
        if self._cumulative_scores is not None:
            self._cumulative_scores = self._cumulative_scores[keep_indices]
        if self._token_ids is not None:
            self._token_ids = self._token_ids[keep_indices]

    def _select_keep_indices(self, protected: torch.Tensor) -> torch.Tensor:
        protected = self._apply_cumulative_blend(protected)
        budget = max(1, min(self._dynamic_budget, protected.shape[0]))

        if self.random_eviction:
            # Random eviction: keep last token + random selection
            primary = torch.rand(protected.shape[0], device=protected.device)
            primary[-1] = float("inf")
        elif self.multi_scale_ema and self._momentum_fast is not None and self._momentum_slow is not None:
            # Conservative retention: token is kept if important under ANY time scale
            primary = torch.maximum(self._momentum_fast, self._momentum_slow)
        else:
            primary = protected.clone()
        primary[self._rom_indices] = float("inf")
        primary[-1] = float("inf")

        evidence_available = (
            self.evidence_reserve_ratio > 0
            and self._evidence_memory is not None
            and self._evidence_memory.shape == protected.shape
        )
        reserve_enabled = self.coverage_reserve_ratio > 0 or evidence_available
        if not reserve_enabled or budget <= 1:
            _, keep_indices = torch.topk(primary, k=budget, largest=True)
            return keep_indices.sort().values

        reserve_parts = []
        reserve_budget_left = budget - 1

        if self.coverage_reserve_ratio > 0 and reserve_budget_left > 0:
            coverage_reserve = max(1, int(round(budget * self.coverage_reserve_ratio)))
            coverage_reserve = min(coverage_reserve, reserve_budget_left)
            coverage_indices = self._select_coverage_indices(protected, coverage_reserve)
            if coverage_indices.numel() > 0:
                reserve_parts.append(coverage_indices)
                reserve_budget_left -= min(coverage_indices.numel(), coverage_reserve)

        if (
            self.evidence_reserve_ratio > 0
            and self._evidence_memory is not None
            and self._evidence_memory.shape == protected.shape
            and reserve_budget_left > 0
        ):
            evidence_reserve = max(1, int(round(budget * self.evidence_reserve_ratio)))
            evidence_reserve = min(evidence_reserve, reserve_budget_left)
            evidence = self._evidence_memory.clone()
            evidence[self._rom_indices] = float("-inf")
            evidence[-1] = float("-inf")
            available = int(torch.isfinite(evidence).sum().item())
            evidence_reserve = min(evidence_reserve, available)
            if evidence_reserve > 0:
                _, evidence_indices = torch.topk(evidence, k=evidence_reserve, largest=True)
                reserve_parts.append(evidence_indices)

        if reserve_parts:
            keep_indices = torch.unique(torch.cat(reserve_parts), sorted=False)
        else:
            keep_indices = torch.empty(0, device=protected.device, dtype=torch.long)

        if keep_indices.numel() > budget:
            _, order = torch.topk(primary[keep_indices], k=budget, largest=True)
            keep_indices = keep_indices[order]

        if keep_indices.numel() < budget:
            _, fill_candidates = torch.topk(primary, k=budget, largest=True)
            keep_mask = self._keep_mask(keep_indices, primary.shape[0])
            fill = fill_candidates[~keep_mask[fill_candidates]]
            missing = budget - keep_indices.numel()
            keep_indices = torch.cat([keep_indices, fill[:missing]])

        return keep_indices.sort().values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        cache = kwargs["past_key_values"]
        hidden_states = kwargs["hidden_states"]
        hook_device = hidden_states.device
        hook_start = self._profile_start(hook_device)
        self._profile_count("hook_calls")

        q_len = hidden_states.shape[1]
        cache_position = kwargs["cache_position"]

        if cache_position[-1] < q_len:
            self._profile_add("hook_total_sec", hook_start, hook_device)
            return output

        if not self._should_score_layer(module.layer_idx):
            self._profile_count("skipped_score_layers")
            self._profile_add("hook_total_sec", hook_start, hook_device)
            return output

        extract_start = self._profile_start(hook_device)
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        kv_len = keys.shape[2]
        device = keys.device
        dtype = keys.dtype
        self._profile_add("extract_kv_sec", extract_start, device)

        state_start = self._profile_start(device)
        if self._initialized and kv_len < self._momentum.shape[0]:
            self._initialized = False
        if not self._initialized:
            self._init_dura_state(kv_len, device, dtype)

        current_mom_len = self._momentum.shape[0]
        if kv_len > current_mom_len:
            new_momentum = torch.zeros(kv_len, device=device, dtype=dtype)
            new_momentum[:current_mom_len] = self._momentum
            if self.score_accumulation == "sum":
                new_momentum[current_mom_len:] = 0.0
            else:
                new_momentum[current_mom_len:] = self._momentum.mean() if current_mom_len > 0 else 0.0
            self._momentum = new_momentum
            if self._evidence_memory is not None:
                new_evidence = torch.zeros(kv_len, device=device, dtype=dtype)
                new_evidence[:current_mom_len] = self._evidence_memory
                self._evidence_memory = new_evidence
            if self._cumulative_scores is not None:
                new_cumulative = torch.zeros(kv_len, device=device, dtype=dtype)
                new_cumulative[:current_mom_len] = self._cumulative_scores
                self._cumulative_scores = new_cumulative
            if self._momentum_fast is not None:
                new_fast = torch.zeros(kv_len, device=device, dtype=dtype)
                new_fast[:current_mom_len] = self._momentum_fast
                new_fast[current_mom_len:] = self._momentum_fast.mean() if current_mom_len > 0 else 0.0
                self._momentum_fast = new_fast
            if self._momentum_slow is not None:
                new_slow = torch.zeros(kv_len, device=device, dtype=dtype)
                new_slow[:current_mom_len] = self._momentum_slow
                new_slow[current_mom_len:] = self._momentum_slow.mean() if current_mom_len > 0 else 0.0
                self._momentum_slow = new_slow
            if self._token_ids is not None:
                new_count = kv_len - current_mom_len
                new_token_ids = torch.arange(
                    self._next_token_id,
                    self._next_token_id + new_count,
                    device=device,
                    dtype=torch.long,
                )
                self._token_ids = torch.cat([self._token_ids, new_token_ids])
                self._next_token_id += new_count
        self._profile_add("state_update_sec", state_start, device)

        attn = output[1]
        score_start = self._profile_start(device)
        current_decode_step = self._decode_steps + 1
        if self._is_score_decision_layer(module.layer_idx) or self._step_score_decision_step != current_decode_step:
            should_update_score = self._record_score_refresh_decision(current_decode_step, hidden_states)
        else:
            should_update_score = self._step_should_update_score
        if not should_update_score:
            self._profile_count("skipped_score_updates")
        else:
            # Compute entropy weight for this layer if enabled
            entropy_weight = 1.0
            if self.layer_entropy_weighting and attn is not None:
                # attn shape: [batch, num_heads, q_len, kv_len]
                attn_last = attn[:, :, -1, :]  # [batch, num_heads, kv_len]
                # Compute entropy per head, then average
                attn_clamped = attn_last.clamp(min=1e-10)
                per_head_entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [batch, num_heads]
                avg_entropy = per_head_entropy.mean().item()
                # Normalize: max entropy = log(kv_len)
                max_entropy = float(torch.tensor(float(kv_len)).log().item())
                if max_entropy > 0:
                    entropy_weight = avg_entropy / max_entropy  # 0~1, higher = more retrieval-like
                else:
                    entropy_weight = 1.0

            if attn is not None:
                step_scores = self._scores_from_attention(attn, device, dtype, values=values, keys=keys)
                if step_scores is not None and self._update_importance_scores(step_scores, entropy_weight):
                    self._profile_count("scored_layers")
            else:
                try:
                    fallback_start = self._profile_start(device)
                    step_scores = self._compute_last_query_attention(module, hidden_states, keys, kwargs, values=values)
                    self._profile_add("attention_fallback_sec", fallback_start, device)
                    step_scores = step_scores.to(device=device, dtype=dtype)
                    if self._update_importance_scores(step_scores, entropy_weight):
                        self._profile_count("scored_layers")
                except Exception as e:
                    logger.warning(f"Failed to compute attention: {e}")
        self._profile_add("score_update_sec", score_start, device)

        # Collect per-layer scores for theory σ estimation (cross-layer disagreement)
        if (self.adaptive_alpha_mode == "theory" and not self._theory_alpha_selected
                and "step_scores" in locals() and step_scores is not None):
            self._theory_layer_scores.append(step_scores.detach().cpu().float())

        if self.boundary_sentinel_policy != "off" and module.layer_idx == self._last_layer_idx:
            self._boundary_sentinel_scores = None
            sentinel_scores = None
            if "step_scores" in locals() and step_scores is not None:
                sentinel_scores = step_scores
            elif attn is not None:
                sentinel_scores = self._scores_from_attention(attn, device, dtype, values=values, keys=keys)
                if sentinel_scores is not None:
                    self._profile_count("boundary_sentinel_scored_layers")
            elif self.boundary_sentinel_compute_fallback:
                try:
                    fallback_start = self._profile_start(device)
                    sentinel_scores = self._compute_last_query_attention(module, hidden_states, keys, kwargs)
                    self._profile_add("boundary_sentinel_attention_sec", fallback_start, device)
                    sentinel_scores = sentinel_scores.to(device=device, dtype=dtype)
                    self._profile_count("boundary_sentinel_scored_layers")
                except Exception as e:
                    logger.warning(f"Failed to compute boundary sentinel attention: {e}")
            if sentinel_scores is not None and sentinel_scores.shape == self._momentum.shape:
                self._boundary_sentinel_scores = sentinel_scores.detach()

        if module.layer_idx != self._last_layer_idx:
            self._profile_add("hook_total_sec", hook_start, device)
            return output

        self._decode_steps += 1
        if "step_scores" in locals():
            self._update_next_adaptive_alpha(step_scores)
            # Theory-derived adaptive alpha: collect during warmup, select after
            if self.adaptive_alpha_mode == "theory" and not self._theory_alpha_selected:
                self._collect_theory_statistics(step_scores)
                if self._decode_steps >= self.theory_warmup_steps:
                    self._theory_select_alpha()
        self._record_boundary_sentinel_state(self._boundary_sentinel_scores)

        if self._momentum.shape[0] > self._dynamic_budget:
            num_to_evict = self._momentum.shape[0] - self._dynamic_budget
            is_first_large_evict = num_to_evict > 100

            if self.use_first_large_evict_rescue and is_first_large_evict:
                try:
                    fallback_start = self._profile_start(device)
                    current_scores = self._compute_last_query_attention(module, hidden_states, keys, kwargs)
                    self._profile_add("first_rescue_attention_sec", fallback_start, device)
                    current_scores = current_scores.to(device=device, dtype=dtype)
                    protected = current_scores.clone()
                except Exception:
                    protected = self._momentum.clone()
            else:
                protected = self._momentum.clone()

            select_start = self._profile_start(device)
            baseline_keep_indices = None
            if (
                self.boundary_sentinel_policy != "off"
                and self._boundary_sentinel_scores is not None
                and self._boundary_sentinel_scores.shape == protected.shape
                and (self.boundary_sentinel_apply_on_refresh or not self._step_should_update_score)
            ):
                baseline_protected = protected.clone()
                baseline_protected[self._rom_indices] = float("inf")
                baseline_protected[-1] = float("inf")
                _, baseline_keep_indices = torch.topk(
                    baseline_protected,
                    k=self._dynamic_budget,
                    largest=True,
                )
                baseline_keep_indices = baseline_keep_indices.sort().values
                protected = self._apply_boundary_sentinel(protected, self._boundary_sentinel_scores)
                self._boundary_sentinel_applied_count += 1
                self._profile_count("boundary_sentinel_applied_count")
            keep_indices = self._select_keep_indices(protected)
            if baseline_keep_indices is not None and not torch.equal(baseline_keep_indices, keep_indices):
                baseline_mask = self._keep_mask(baseline_keep_indices, protected.shape[0])
                keep_mask = self._keep_mask(keep_indices, protected.shape[0])
                replacement_rate = torch.logical_and(keep_mask, ~baseline_mask).sum().item() / max(
                    1, self._dynamic_budget
                )
                self._boundary_sentinel_changed_keep_count += 1
                self._boundary_sentinel_changed_token_rate_sum += float(replacement_rate)
                self._profile_count("boundary_sentinel_changed_keep_count")
            if "step_scores" in locals():
                self._record_rank_trace(step_scores, keep_indices)
            if (
                self.adaptive_alpha
                and self._previous_adaptive_raw_top_ids is not None
                and self._token_ids is not None
            ):
                keep_ids = set(
                    int(token_id)
                    for token_id in self._token_ids.index_select(0, keep_indices).detach().cpu().tolist()
                )
                self._previous_adaptive_raw_top_ids &= keep_ids
            self._profile_add("eviction_select_sec", select_start, device)

            cache_update_start = self._profile_start(device)
            self._batch_evict_and_update_cache(cache, keep_indices)
            self._profile_add("cache_update_sec", cache_update_start, device)
            self._compression_events += 1
            self._last_compressed_cache_tokens = int(keep_indices.numel())
            if self._first_compress_step < 0:
                self._first_compress_step = self._decode_steps
                self._post_first_compress_cache_tokens = int(keep_indices.numel())

        self._profile_add("hook_total_sec", hook_start, device)
        return output

    @contextmanager
    def __call__(self, model):
        from transformers import Gemma3ForConditionalGeneration, PreTrainedModel

        if not isinstance(model, PreTrainedModel):
            logger.warning(f"Model {type(model)} not tested")

        if isinstance(model, Gemma3ForConditionalGeneration):
            logger.warning_once("Compression in Gemma3 is only applied to layer without sliding window attention")

        self.post_init_from_model(model)
        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                    continue
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
            self._last_run_stats = self._current_run_stats()
            self._last_rank_trace = list(self._rank_trace)
            self.reset()

    def reset(self):
        self._initialized = False
        self._rom_indices = None
        self._momentum = None
        self._evidence_memory = None
        self._cumulative_scores = None
        self._token_ids = None
        self._next_token_id = 0
        self._dynamic_budget = 0
        self._compression_events = 0
        self._decode_steps = 0
        self._first_compress_step = -1
        self._post_first_compress_cache_tokens = 0
        self._last_compressed_cache_tokens = 0
        self._profile_stats = {}
        self._rank_trace = []
        self._previous_raw_top_ids = None
        self._previous_ema_top_ids = None
        self._previous_adaptive_raw_top_ids = None
        self._active_alpha = self.alpha
        self._adaptive_alpha_sum = 0.0
        self._adaptive_alpha_count = 0
        self._adaptive_alpha_low_count = 0
        self._adaptive_alpha_mid_count = 0
        self._adaptive_alpha_high_count = 0
        self._adaptive_shift_count = 0
        self._adaptive_shift_streak = 0
        self._adaptive_uncertain_count = 0
        self._last_score_refresh_step = 0
        self._last_refresh_query = None
        self._step_score_decision_step = -1
        self._step_should_update_score = True
        self._refresh_interval_count = 0
        self._refresh_drift_count = 0
        self._refresh_query_drift_sum = 0.0
        self._refresh_query_drift_count = 0
        self._refresh_query_drift_max = 0.0
        self._boundary_sentinel_scores = None
        self._boundary_sentinel_force_refresh = False
        self._boundary_sentinel_refresh_streak = 0
        self._boundary_sentinel_decisions = 0
        self._boundary_sentinel_applied_count = 0
        self._boundary_sentinel_forced_refresh_count = 0
        self._boundary_sentinel_disagreement_sum = 0.0
        self._boundary_sentinel_disagreement_count = 0
        self._boundary_sentinel_disagreement_max = 0.0
        self._boundary_sentinel_changed_keep_count = 0
        self._boundary_sentinel_changed_token_rate_sum = 0.0
        # Multi-scale EMA reset
        self._momentum_fast = None
        self._momentum_slow = None
        # Theory adaptive alpha reset
        self._theory_score_diffs = []
        self._theory_topk_sets = []
        self._theory_prev_scores = None
        self._theory_alpha_selected = False
        self._theory_estimated_sigma = 0.0
        self._theory_estimated_delta = 0.0
        self._theory_eta = 0.0
        self._theory_layer_scores = []
