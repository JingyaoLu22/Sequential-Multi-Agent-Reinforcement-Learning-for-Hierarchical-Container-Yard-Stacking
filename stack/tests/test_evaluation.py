"""
Evaluation behavior: results do not depend on how many episodes run
side by side, episode i is StackEnv's episode reset(seed=base_seed + i),
and periodic evaluation cannot change training.

Deterministic evaluation (argmax, used during training and by default)
is a pure function of the seeded environments, so it is compared
exactly. Stochastic evaluation samples from torch's global RNG, which a
batched sample() consumes differently, so it only gets a smoke test.
"""

import math

import numpy as np
import pytest
import torch

from stack.envs.stack_gym import StackEnv
from stack.evaluation.evaluate import evaluate_policy
from stack.training.bay_row_layout import BayRowLayout


def _small_environment_config(seed: int) -> dict:
    return {
        "vessel_shape": (3, 3, 2),
        "yard_shape": (3, 3, 2),
        "num_containers": 12,
        "group_num": 3,
        "group_placement": "random",
        "seed": seed,
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    }


def _build_actors(environment_config: dict):
    layout = BayRowLayout.from_env(StackEnv(config=environment_config))
    return layout.build_actors(embed_dim=16, n_heads=2, n_layers=1, dropout=0.0)


def _evaluate(environment_config, bay_actor, row_actor, n_episodes, base_seed, eval_batch_size):
    episodes, _ = evaluate_policy(
        environment_config=environment_config,
        bay_actor=bay_actor,
        row_actor=row_actor,
        n_episodes=n_episodes,
        base_seed=base_seed,
        eval_batch_size=eval_batch_size,
        deterministic=True,
        verbose=False,
    )
    return episodes


@pytest.mark.parametrize(
    "n_episodes,eval_batch_size",
    [
        (5, 2),  # two full rounds plus a partial one (padded, inactive slots)
        (3, 8),  # one round, more slots than episodes
    ],
)
def test_results_do_not_depend_on_the_batch_size(n_episodes: int, eval_batch_size: int) -> None:
    environment_config = _small_environment_config(seed=0)
    bay_actor, row_actor = _build_actors(environment_config)

    one_at_a_time = _evaluate(environment_config, bay_actor, row_actor, n_episodes, 1000, 1)
    batched = _evaluate(environment_config, bay_actor, row_actor, n_episodes, 1000, eval_batch_size)

    assert [episode.to_dict() for episode in batched] == [episode.to_dict() for episode in one_at_a_time]


def test_episode_i_is_the_stack_env_episode_of_seed_base_plus_i() -> None:
    """The contract plots.py relies on to compare agents on the same episodes."""

    environment_config = _small_environment_config(seed=0)
    bay_actor, row_actor = _build_actors(environment_config)
    episodes = _evaluate(environment_config, bay_actor, row_actor, n_episodes=4, base_seed=50, eval_batch_size=3)

    for i, episode in enumerate(episodes):
        # Replay the recorded actions on a single StackEnv reset with seed 50 + i.
        env = StackEnv(config=environment_config)
        env.reset(seed=50 + i)
        rewards = [env.step(action)[1] for action in episode.global_actions]
        np.testing.assert_allclose(rewards, episode.step_rewards, rtol=1e-6)


def test_batched_stochastic_runs_and_produces_valid_metrics() -> None:
    """Smoke test only - see module docstring for why exact
    reproducibility-by-seed doesn't apply to deterministic=False."""

    environment_config = _small_environment_config(seed=3)
    bay_actor, row_actor = _build_actors(environment_config)

    episodes, summary = evaluate_policy(
        environment_config=environment_config,
        bay_actor=bay_actor,
        row_actor=row_actor,
        n_episodes=4,
        base_seed=3000,
        eval_batch_size=3,
        deterministic=False,
        verbose=False,
    )

    assert len(episodes) == 4
    assert summary.n_episodes == 4
    for episode in episodes:
        assert episode.episode_length == len(episode.step_rewards)
        assert episode.episode_length > 0
        assert episode.terminated or episode.truncated


def test_verbose_false_suppresses_output(capsys) -> None:
    environment_config = _small_environment_config(seed=9)
    bay_actor, row_actor = _build_actors(environment_config)

    evaluate_policy(
        environment_config=environment_config,
        bay_actor=bay_actor,
        row_actor=row_actor,
        n_episodes=2,
        base_seed=4000,
        eval_batch_size=2,
        deterministic=True,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_periodic_evaluation_does_not_change_training() -> None:
    """Evaluating inside a rollout (as the runner's step_callback does)
    must leave training bit-identical, so no RNG save/restore is needed."""

    from dataclasses import replace

    from stack.configs.hierarchical_config import get_hierarchical_config
    from stack.run_sequential_hppo import build_training_system
    from stack.training.rollout_buffer import JointRolloutBuffer

    environment_config = _small_environment_config(seed=11)
    algorithm_config = replace(
        get_hierarchical_config("small"),
        embed_dim=16, n_heads=2, n_layers=1, vf_dim=16,
        dropout=0.1, buffer_size=24, batch_size=8, n_epochs=2,
    )

    def train(evaluate: bool):
        torch.manual_seed(0)
        env, _bay, _row, _critic, trainer = build_training_system(
            environment_config, algorithm_config, device=torch.device("cpu"), num_envs=2
        )

        def step_callback(_step):
            if evaluate:
                evaluate_policy(
                    environment_config=environment_config,
                    bay_actor=trainer.bay_actor,
                    row_actor=trainer.row_actor,
                    n_episodes=2,
                    base_seed=500,
                    verbose=False,
                )

        stats = []
        for _ in range(2):
            buffer = JointRolloutBuffer(trainer.layout, 24, 2, "cpu")
            iteration = trainer.train_iteration(env, buffer, step_callback)
            stats.append({k: v for k, v in iteration.items() if not k.endswith("_seconds")})
        env.close()
        return stats, trainer.row_actor.state_dict()

    stats_without, params_without = train(evaluate=False)
    stats_with, params_with = train(evaluate=True)

    for with_eval, without_eval in zip(stats_with, stats_without):
        assert with_eval.keys() == without_eval.keys()
        for key, value in without_eval.items():
            assert with_eval[key] == value or (math.isnan(value) and math.isnan(with_eval[key])), key
    for key in params_without:
        assert torch.equal(params_with[key], params_without[key]), key
