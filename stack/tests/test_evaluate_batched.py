"""
Phase 5 equivalence test for evaluate_policy_batched().

See docs/plan §7, test J: batched evaluation returns the same episode
rewards as serial evaluation for the same seeds.

Scope note on deterministic vs. stochastic
-------------------------------------------
For `deterministic=True` (argmax, no sampling - the ONLY mode
evaluate_for_training() ever uses for periodic training evaluation,
and evaluate_policy()'s CLI default), every episode's action sequence
is a pure function of the environment's own seeded RNG, so batched and
sequential evaluation must produce bit-identical results. This is
tested strictly below.

For `deterministic=False` (`--stochastic` on the CLI, opt-in only),
policy sampling draws from torch.distributions.Categorical.sample(),
which consumes torch's ambient GLOBAL RNG state - not something
env.reset(seed=...) controls, and something a batched (B, ...)-shaped
sample() call consumes differently than B separate single-sample
calls. Exact action-sequence reproducibility by seed was therefore
never a real guarantee of the original sequential implementation
either way; only a smoke test (runs correctly, produces well-formed,
valid metrics) is meaningful here, not a bit-equality oracle.
"""

import numpy as np
import torch

from stack.agents.agent_b import AgentB
from stack.agents.agent_r import AgentR
from stack.envs.hierarchical_envs.hierarchical_env import HierarchicalEnv
from stack.evaluation.evaluate import evaluate_episode, evaluate_policy_batched


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


def _build_agents(environment_config: dict):
    device = torch.device("cpu")
    env = HierarchicalEnv(config=environment_config)
    n_bays = env.bay_action_space.n
    n_rows = env.row_action_space.n
    agent_b = AgentB(
        observation_space=env.bay_observation_space,
        n_bays=n_bays,
        n_rows_per_bay=n_rows,
        device=device,
        embed_dim=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    agent_r = AgentR(
        observation_space=env.row_observation_space,
        n_rows=n_rows,
        device=device,
        embed_dim=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    env.close()
    return agent_b, agent_r


def _sequential_reference(environment_config, agent_b, agent_r, n_episodes, base_seed, deterministic):
    agent_b.eval()
    agent_r.eval()
    episodes = []
    for i in range(n_episodes):
        env = HierarchicalEnv(config=environment_config)
        with torch.no_grad():
            episode = evaluate_episode(
                env=env,
                agent_b=agent_b,
                agent_r=agent_r,
                episode_index=i,
                seed=base_seed + i,
                deterministic=deterministic,
                verbose_steps=False,
            )
        env.close()
        episodes.append(episode)
    return episodes


def test_batched_matches_sequential_deterministic_across_partial_round() -> None:
    """n_episodes=5, eval_batch_size=2: two full rounds plus a partial
    (1-episode) round, exercising the padding/inactive-slot path."""

    environment_config = _small_environment_config(seed=0)
    agent_b, agent_r = _build_agents(environment_config)

    n_episodes = 5
    base_seed = 1000

    reference = _sequential_reference(
        environment_config, agent_b, agent_r, n_episodes, base_seed, deterministic=True
    )

    batched_episodes, batched_summary = evaluate_policy_batched(
        environment_config=environment_config,
        agent_b=agent_b,
        agent_r=agent_r,
        n_episodes=n_episodes,
        base_seed=base_seed,
        eval_batch_size=2,
        deterministic=True,
        verbose=False,
    )

    assert len(batched_episodes) == n_episodes

    for i in range(n_episodes):
        ref = reference[i]
        got = batched_episodes[i]

        assert got.episode_index == ref.episode_index
        assert got.total_reward == ref.total_reward
        assert got.episode_length == ref.episode_length
        assert got.step_rewards == ref.step_rewards
        assert got.cumulative_rewards == ref.cumulative_rewards
        assert got.bay_actions == ref.bay_actions
        assert got.row_actions == ref.row_actions
        assert got.selected_bays == ref.selected_bays
        assert got.selected_rows == ref.selected_rows
        assert got.global_actions == ref.global_actions
        assert got.terminated == ref.terminated
        assert got.truncated == ref.truncated
        assert got.containers_retrieved == ref.containers_retrieved
        assert got.containers_remaining == ref.containers_remaining

    assert batched_summary.n_episodes == n_episodes


def test_batched_matches_sequential_deterministic_single_full_round() -> None:
    """eval_batch_size >= n_episodes: exactly one round, no padding."""

    environment_config = _small_environment_config(seed=7)
    agent_b, agent_r = _build_agents(environment_config)

    n_episodes = 3
    base_seed = 2000

    reference = _sequential_reference(
        environment_config, agent_b, agent_r, n_episodes, base_seed, deterministic=True
    )

    batched_episodes, _ = evaluate_policy_batched(
        environment_config=environment_config,
        agent_b=agent_b,
        agent_r=agent_r,
        n_episodes=n_episodes,
        base_seed=base_seed,
        eval_batch_size=8,
        deterministic=True,
        verbose=False,
    )

    for i in range(n_episodes):
        assert batched_episodes[i].total_reward == reference[i].total_reward
        assert batched_episodes[i].bay_actions == reference[i].bay_actions
        assert batched_episodes[i].row_actions == reference[i].row_actions


def test_batched_stochastic_runs_and_produces_valid_metrics() -> None:
    """Smoke test only - see module docstring for why exact
    reproducibility-by-seed doesn't apply to deterministic=False."""

    environment_config = _small_environment_config(seed=3)
    agent_b, agent_r = _build_agents(environment_config)

    episodes, summary = evaluate_policy_batched(
        environment_config=environment_config,
        agent_b=agent_b,
        agent_r=agent_r,
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
    agent_b, agent_r = _build_agents(environment_config)

    evaluate_policy_batched(
        environment_config=environment_config,
        agent_b=agent_b,
        agent_r=agent_r,
        n_episodes=2,
        base_seed=4000,
        eval_batch_size=2,
        deterministic=True,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
