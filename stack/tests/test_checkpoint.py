"""
training/checkpoint.py: a resumed run continues exactly like an
uninterrupted one, and incompatible checkpoints are refused.
"""

import math
from dataclasses import replace

import pytest
import torch

from stack.configs.environments import set_config
from stack.configs.hierarchical_config import get_hierarchical_config
from stack.run_sequential_hppo import build_training_system
from stack.training.checkpoint import resume_training_state, save_models, save_training_state
from stack.training.rollout_buffer import JointRolloutBuffer

ENVIRONMENT = dict(set_config("small_with_margin", seed=3), vessel_shape=(3, 3, 2), yard_shape=(3, 3, 2),
                   num_containers=12)
ALGORITHM = replace(get_hierarchical_config("small"), embed_dim=16, n_heads=2, n_layers=1, vf_dim=16,
                    buffer_size=24, batch_size=8, n_epochs=2)


def _trainer(seed: int):
    torch.manual_seed(seed)
    env, _bay, _row, _critic, trainer = build_training_system(
        ENVIRONMENT, ALGORITHM, torch.device("cpu"), num_envs=2
    )
    return env, trainer


def _iterate(env, trainer):
    stats = trainer.train_iteration(env, JointRolloutBuffer(trainer.layout, 24, 2, "cpu"))
    return {k: v for k, v in stats.items() if not k.endswith("_seconds")}


def _same_stats(a, b) -> bool:
    return a.keys() == b.keys() and all(
        a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a
    )


def test_resume_restores_everything_training_depends_on(tmp_path) -> None:
    env, trainer = _trainer(seed=0)
    _iterate(env, trainer)
    save_training_state(tmp_path, trainer, ENVIRONMENT, ALGORITHM, best_eval_reward=-5.0)
    saved_optimizer = trainer.row_optimizer.state_dict()
    env.close()

    # Two new processes with different initial weights and RNG streams:
    # after resuming, nothing of their own initialization may survive.
    continuations = []
    for seed in (123, 456):
        env, resumed = _trainer(seed)
        assert resume_training_state(tmp_path, resumed, ENVIRONMENT, ALGORITHM) == -5.0
        assert resumed.total_environment_steps == 24
        restored = resumed.row_optimizer.state_dict()["state"]
        for index, moments in saved_optimizer["state"].items():
            assert all(torch.equal(moments[k], restored[index][k]) for k in moments)
        continuations.append((_iterate(env, resumed), resumed.row_actor.state_dict()))
        env.close()

    (stats_a, params_a), (stats_b, params_b) = continuations
    assert _same_stats(stats_a, stats_b)
    assert stats_a["total_environment_steps"] == 48
    assert all(torch.equal(params_a[k], params_b[k]) for k in params_a)


def test_refuses_checkpoint_from_a_different_environment(tmp_path) -> None:
    env, trainer = _trainer(seed=0)
    save_training_state(tmp_path, trainer, ENVIRONMENT, ALGORITHM, best_eval_reward=0.0)
    env.close()

    with pytest.raises(ValueError, match="reward_norm: checkpoint=False, now=True"):
        resume_training_state(tmp_path, trainer, dict(ENVIRONMENT, reward_norm=True), ALGORITHM)

    with pytest.raises(ValueError, match="incompatible model/training settings"):
        resume_training_state(tmp_path, trainer, ENVIRONMENT, replace(ALGORITHM, n_epochs=3))

    # Horizons and cadence may change on resume.
    resume_training_state(tmp_path, trainer, ENVIRONMENT, replace(ALGORITHM, total_timesteps=10**7, eval_freq=1))


def test_save_models_writes_the_evaluation_exports(tmp_path) -> None:
    env, trainer = _trainer(seed=0)
    env.close()
    save_models(tmp_path, trainer, ENVIRONMENT, ALGORITHM, prefix="best")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "best_agent_b.pt", "best_agent_r.pt", "best_config.json", "best_critic.pt"
    ]
    saved = torch.load(tmp_path / "best_agent_r.pt")
    assert all(torch.equal(saved[k], v) for k, v in trainer.row_actor.state_dict().items())
