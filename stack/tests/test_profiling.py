"""Profiling is opt-in and never synchronizes CUDA when it is off."""

from dataclasses import replace

import pytest
import torch

from stack.configs.environments import set_config
from stack.configs.hierarchical_config import get_hierarchical_config
from stack.run_sequential_hppo import build_parser, build_training_system
from stack.training.profiling import Profiler
from stack.training.rollout_buffer import JointRolloutBuffer
from stack.training.sequential_trainer import PHASE_TIMINGS

ENVIRONMENT = dict(set_config("small_with_margin", seed=3), vessel_shape=(3, 3, 2), yard_shape=(3, 3, 2),
                   num_containers=12)
ALGORITHM = replace(get_hierarchical_config("small"), embed_dim=16, n_heads=2, n_layers=1, vf_dim=16,
                    buffer_size=12, batch_size=6, n_epochs=1)


def _iteration_stats(profiler, monkeypatch):
    synchronize_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: synchronize_calls.append(1))

    env, _bay, _row, _critic, trainer = build_training_system(
        ENVIRONMENT, ALGORITHM, torch.device("cpu"), num_envs=2, profiler=profiler
    )
    stats = trainer.train_iteration(env, JointRolloutBuffer(trainer.layout, 12, 2, "cpu"))
    env.close()
    return stats, synchronize_calls


def test_cli_profiling_is_off_unless_requested() -> None:
    parser = build_parser()
    assert parser.parse_args([]).profile is False
    assert parser.parse_args(["--profile"]).profile is True
    # --profile no longer doubles as the training-profile alias.
    assert parser.parse_args(["--training_profile", "massive"]).training_profile == "massive"
    with pytest.raises(SystemExit):
        parser.parse_args(["--profile", "massive"])


def test_disabled_profiler_adds_no_columns_and_no_synchronization(monkeypatch) -> None:
    for profiler in (None, Profiler()):
        stats, synchronize_calls = _iteration_stats(profiler, monkeypatch)
        assert not any(key.endswith("_seconds") for key in stats)
        assert synchronize_calls == []


def test_enabled_profiler_reports_every_phase(monkeypatch) -> None:
    # Pretend to be on a GPU so region(sync_cuda=True) really synchronizes.
    profiler = Profiler(enabled=True)
    profiler._cuda_available = True

    stats, synchronize_calls = _iteration_stats(profiler, monkeypatch)

    assert set(PHASE_TIMINGS) <= set(stats)
    assert stats["rollout_seconds"] > 0 and stats["row_update_seconds"] > 0
    # Two per timed phase (start and end) - independent of rollout length.
    assert len(synchronize_calls) == 2 * 5
