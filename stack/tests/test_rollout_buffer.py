"""
JointRolloutBuffer: (T, N) storage and vectorized GAE.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from stack.training.bay_row_layout import BayRowLayout
from stack.training.rollout_buffer import JointRolloutBuffer

N_BAYS, N_ROWS, FEATURES = 2, 3, 4
GAMMA, GAE_LAMBDA = 0.99, 0.95


def _layout() -> BayRowLayout:
    obs_dim = N_BAYS * N_ROWS * FEATURES
    return BayRowLayout(
        n_bays=N_BAYS,
        n_rows=N_ROWS,
        observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        row_observation_space=spaces.Box(-np.inf, np.inf, (N_ROWS * FEATURES,), np.float32),
    )


def _transition(step: int, num_envs: int, rng: np.random.Generator) -> dict:
    return dict(
        # Encode (step, env) so flattening order can be checked.
        global_states=np.full((num_envs, N_BAYS * N_ROWS * FEATURES), step)
        + np.arange(num_envs)[:, None] / 10,
        bay_action_masks=np.ones((num_envs, N_BAYS), dtype=bool),
        bay_actions=np.zeros(num_envs, dtype=np.int64),
        bay_log_probs=np.zeros(num_envs),
        row_observations=np.zeros((num_envs, N_ROWS * FEATURES)),
        row_action_masks=np.ones((num_envs, N_ROWS), dtype=bool),
        row_actions=np.zeros(num_envs, dtype=np.int64),
        row_log_probs=np.zeros(num_envs),
        rewards=rng.normal(size=num_envs),
        dones=rng.random(num_envs) < 0.3,
        values=rng.normal(size=num_envs),
        next_values=rng.normal(size=num_envs),
    )


def _filled_buffer(n_steps: int, num_envs: int, seed: int = 0) -> JointRolloutBuffer:
    rng = np.random.default_rng(seed)
    buffer = JointRolloutBuffer(_layout(), n_steps * num_envs, num_envs, "cpu")
    for step in range(n_steps):
        buffer.add_batch(**_transition(step, num_envs, rng))
    return buffer


def _reference_gae(rewards, values, next_values, dones):
    """Textbook single-environment GAE recursion in float64."""

    advantages = np.zeros(len(rewards))
    last = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 0.0 if dones[t] else 1.0
        delta = rewards[t] + GAMMA * next_values[t] * not_done - values[t]
        last = delta + GAMMA * GAE_LAMBDA * not_done * last
        advantages[t] = last
    return advantages


@pytest.mark.parametrize("num_envs", [1, 4])
def test_vectorized_gae_matches_per_environment_reference(num_envs: int) -> None:
    buffer = _filled_buffer(n_steps=25, num_envs=num_envs)
    advantages, returns = buffer.compute_gae(GAMMA, GAE_LAMBDA)

    for env in range(num_envs):
        expected = _reference_gae(
            *(getattr(buffer, name)[:, env].double().numpy() for name in ("rewards", "values", "next_values")),
            buffer.dones[:, env].numpy(),
        )
        np.testing.assert_allclose(buffer.advantages[:, env].numpy(), expected, rtol=1e-5, atol=1e-5)

    # Flattened as sample t * N + n, with returns = advantages + values.
    torch.testing.assert_close(advantages, buffer.advantages.reshape(-1))
    torch.testing.assert_close(returns, advantages + buffer.values.reshape(-1))


def test_rollout_batch_flattens_step_major() -> None:
    num_envs = 3
    buffer = _filled_buffer(n_steps=4, num_envs=num_envs)
    buffer.compute_gae(GAMMA, GAE_LAMBDA)
    batch = buffer.rollout_batch()

    assert len(buffer) == 12 and batch.global_states.shape == (12, N_BAYS * N_ROWS * FEATURES)
    for sample, state in enumerate(batch.global_states[:, 0].tolist()):
        step, env = divmod(sample, num_envs)
        assert state == pytest.approx(step + env / 10)


def test_rejects_size_that_does_not_split_into_steps() -> None:
    with pytest.raises(ValueError, match="multiple"):
        JointRolloutBuffer(_layout(), buffer_size=10, num_envs=4, device="cpu")


def test_add_to_full_buffer_raises() -> None:
    buffer = _filled_buffer(n_steps=2, num_envs=2)
    assert buffer.is_full()
    with pytest.raises(RuntimeError, match="full"):
        buffer.add_batch(**_transition(2, 2, np.random.default_rng(1)))
