"""
training/checkpoint.py: a resumed run continues exactly as determined by
the checkpoint, and incompatible checkpoints are refused.
"""

from dataclasses import replace

import pytest
import torch

from stack.run_sequential_hppo import build_training_system
from stack.training.checkpoint import resume_training_state, save_training_state
from stack.training.rollout_buffer import JointRolloutBuffer

from .common import algorithm, environment, training_system

ENVIRONMENT = environment(3)
ALGORITHM = algorithm(buffer_size=24, batch_size=8, n_epochs=2, dropout=0.1)


def _checkpoint(tmp_path):
    env, trainer = training_system(ENVIRONMENT, ALGORITHM)
    trainer.train_iteration(env, JointRolloutBuffer(trainer.layout, 24, 2, "cpu"))
    save_training_state(tmp_path, trainer, env, ENVIRONMENT, ALGORITHM, best_eval_reward=-5.0)
    saved = trainer.row_optimizer.state_dict()["state"], env.get_attr("seed")
    env.close()
    return saved


def test_resume_restores_everything_training_depends_on(tmp_path) -> None:
    saved_optimizer, saved_seeds = _checkpoint(tmp_path)

    # Two processes with different initial weights and RNG streams: after
    # resuming, nothing of their own initialization may survive.
    continuations = []
    for seed in (123, 456):
        torch.manual_seed(seed)
        env, _bay, _row, _critic, resumed = build_training_system(ENVIRONMENT, ALGORITHM, torch.device("cpu"), 2)
        assert resume_training_state(tmp_path, resumed, env, ENVIRONMENT, ALGORITHM) == -5.0
        assert resumed.total_environment_steps == 24
        assert env.get_attr("seed") == saved_seeds  # new episodes, not the run's first ones
        restored = resumed.row_optimizer.state_dict()["state"]
        assert all(torch.equal(m[k], restored[i][k]) for i, m in saved_optimizer.items() for k in m)

        stats = resumed.train_iteration(env, JointRolloutBuffer(resumed.layout, 24, 2, "cpu"))
        env.close()
        continuations.append((repr(stats), resumed.row_actor.state_dict()))

    (stats_a, params_a), (stats_b, params_b) = continuations
    assert stats_a == stats_b
    assert all(torch.equal(params_a[k], params_b[k]) for k in params_a)


def test_refuses_incompatible_checkpoints(tmp_path) -> None:
    _saved_optimizer, saved_seeds = _checkpoint(tmp_path)
    env, trainer = training_system(ENVIRONMENT, ALGORITHM, num_envs=3)

    with pytest.raises(ValueError, match="reward_norm: checkpoint=False, now=True"):
        resume_training_state(tmp_path, trainer, env, dict(ENVIRONMENT, reward_norm=True), ALGORITHM)
    with pytest.raises(ValueError, match="incompatible model/training settings"):
        resume_training_state(tmp_path, trainer, env, ENVIRONMENT, replace(ALGORITHM, n_epochs=3))

    # Horizons, cadence and --num_envs may change; more environments
    # continue past every saved episode.
    resume_training_state(tmp_path, trainer, env, ENVIRONMENT, replace(ALGORITHM, total_timesteps=10**7, eval_freq=1))
    assert env.get_attr("seed") == [max(saved_seeds) + 1 + rank for rank in range(3)]
    env.close()
