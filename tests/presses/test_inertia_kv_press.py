# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress import InertiaKVPress, InertiaLearnedPress


def test_emnlp_operating_point_configuration():
    full = InertiaKVPress(alpha=0.8, rom_ratio=0.0, use_first_large_evict_rescue=False)
    lazy4 = InertiaKVPress(
        alpha=0.8,
        rom_ratio=0.0,
        use_first_large_evict_rescue=False,
        score_update_interval=4,
    )
    score_free = InertiaKVPress(
        alpha=0.8,
        rom_ratio=0.0,
        use_first_large_evict_rescue=False,
        score_update_interval=999999,
    )

    assert full.score_update_interval == 1
    assert lazy4.score_update_interval == 4
    assert score_free.score_update_interval == 999999


def test_emnlp_ema_updates_sequentially_across_layers():
    press = InertiaKVPress(alpha=0.5)
    press._momentum = torch.zeros(2)

    first_layer = torch.tensor([1.0, 0.0])
    second_layer = torch.tensor([0.0, 1.0])
    assert press._update_importance_scores(first_layer)
    assert press._update_importance_scores(second_layer)

    # 0.5 * (0.5 * first_layer) + 0.5 * second_layer
    torch.testing.assert_close(press._momentum, torch.tensor([0.25, 0.5]))


@pytest.mark.parametrize("interval,expected_refreshes", [(1, 6), (4, 2), (999999, 1)])
def test_refresh_intervals(interval, expected_refreshes):
    press = InertiaKVPress(score_update_interval=interval)
    hidden_states = torch.ones(1, 1, 8)
    decisions = [press._record_score_refresh_decision(step, hidden_states) for step in range(1, 7)]
    assert sum(decisions) == expected_refreshes
    assert decisions[0]


@pytest.mark.parametrize("interval,expected_refreshes", [(1, 6), (4, 2), (999999, 1)])
def test_tiny_llama_decoding(interval, expected_refreshes, caplog):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    model.config._attn_implementation = "sdpa"
    cache = DynamicCache(config=config)
    press = InertiaKVPress(
        total_budget=4,
        rom_ratio=0.0,
        use_first_large_evict_rescue=False,
        score_update_interval=interval,
        enable_profiling=True,
    )
    with torch.no_grad():
        with press(model):
            model(input_ids=torch.arange(8).unsqueeze(0), past_key_values=cache)
            assert cache.get_seq_length() == 8
            assert not press._initialized
            for step in range(6):
                model(
                    input_ids=torch.tensor([[1]]),
                    past_key_values=cache,
                    position_ids=torch.tensor([[8 + step]]),
                )
                assert all(cache.get_seq_length(layer) == 4 for layer in range(2))
                assert press._momentum.shape == (4,)
    summary = press.last_run_summary()
    assert summary["refresh_interval_count"] == expected_refreshes
    assert summary["scored_layers"] == 2 * expected_refreshes
    assert summary["compression_events"] == 6
    assert summary["decode_steps"] == 6
    assert not press._initialized
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
    assert "Failed to compute attention" not in caplog.text


@pytest.mark.parametrize("press_type", [InertiaKVPress, InertiaLearnedPress])
def test_pipeline_rejects_multiple_questions(press_type):
    from kvpress.pipeline import KVPressTextGenerationPipeline

    pipeline = object.__new__(KVPressTextGenerationPipeline)
    with pytest.raises(ValueError, match="single question"):
        pipeline._forward({"questions_ids": [torch.ones(1, 1), torch.ones(1, 1)]}, press=press_type())


def test_pipeline_runs_inertia_during_decoding():
    from kvpress.pipeline import KVPressTextGenerationPipeline

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    pipeline = object.__new__(KVPressTextGenerationPipeline)
    pipeline.model = LlamaForCausalLM(config).eval()
    pipeline.model.config._attn_implementation = "sdpa"
    pipeline.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "generated answer")
    cache = DynamicCache(config=config)
    press = InertiaKVPress(total_budget=4, rom_ratio=0.0, use_first_large_evict_rescue=False, enable_profiling=True)
    with torch.no_grad():
        answers = pipeline._forward(
            {"context_ids": torch.arange(8).unsqueeze(0), "questions_ids": [torch.tensor([[1]])]},
            max_new_tokens=4,
            press=press,
            cache=cache,
        )
    assert answers == ["generated answer"]
    assert press.last_run_summary()["scored_layers"] > 0
    assert cache.get_seq_length() == 4
