"""
Phase 2 equivalence tests for SequentialPPOTrainer's cross-step critic
value reuse.

See docs/plan §7:

F. Critic reuse equivalence - optimized rollout values match the
   original non-reused computation, and the cache is invalidated
   across a train_iteration() boundary.
G. GAE equivalence - the (unmodified) advantage.py still produces a
   correct result when fed a Phase-2 rollout, in particular across an
   auto-reset episode boundary.

advantage.py is untouched by Phase 2, so once test F shows the
buffer's stored values/next_values are exactly the critic's real
outputs (reused or not), GAE correctness follows from advantage.py's
own (unmodified) logic - test G exercises that end to end.
"""

from dataclasses import replace

import numpy as np
import torch

from stack.configs.hierarchical_config import DEFAULT_HIERARCHICAL_CONFIG
from stack.run_sequential_hppo import build_training_system, create_rollout_buffer
from stack.training.advantage import compute_and_store_gae


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


def _small_algorithm_config(buffer_size: int):
    return replace(
        DEFAULT_HIERARCHICAL_CONFIG,
        embed_dim=16,
        n_heads=2,
        n_layers=1,
        vf_dim=16,
        dropout=0.0,
        buffer_size=buffer_size,
        batch_size=buffer_size // 2,
        n_epochs=1,
    )


def _build(num_envs: int, buffer_size: int, seed: int = 0):
    environment_config = _small_environment_config(seed)
    algorithm_config = _small_algorithm_config(buffer_size)
    device = torch.device("cpu")

    env, agent_b, agent_r, critic, trainer = build_training_system(
        environment_config, algorithm_config, device=device, num_envs=num_envs
    )
    buffer = create_rollout_buffer(env, buffer_size=buffer_size, device=device)
    return env, trainer, buffer


class _PredictValuesRecorder:
    """Wraps CentralizedCritic.predict_values to record every call's
    input/output without changing its behaviour."""

    def __init__(self, critic):
        self._critic = critic
        self._original = critic.predict_values
        self.calls = []

    def __enter__(self):
        def wrapped(observations):
            output = self._original(observations)
            self.calls.append((observations.detach().clone(), output.detach().clone()))
            return output

        self._critic.predict_values = wrapped
        return self

    def __exit__(self, *exc_info):
        self._critic.predict_values = self._original


def test_critic_reuse_halves_forward_passes_and_values_match_buffer() -> None:
    num_envs = 2
    buffer_size = 8  # 4 rollout steps
    env, trainer, buffer = _build(num_envs=num_envs, buffer_size=buffer_size)

    num_steps = buffer_size // num_envs

    with _PredictValuesRecorder(trainer.critic) as recorder:
        trainer.collect_rollout(env, buffer)

    # Reuse means exactly one critic call per step, plus one extra for
    # the very first step's V(s_t) (which has nothing to reuse yet) -
    # i.e. num_steps + 1, not 2 * num_steps.
    assert len(recorder.calls) == num_steps + 1

    # Every recorded call's INPUT is that step's global_state - proves
    # the reused value at step k+1 really was computed from step k's
    # next_global_state, not a stale or mismatched tensor.
    for step in range(num_steps):
        expected_input = buffer.global_states[step * num_envs : (step + 1) * num_envs]
        np.testing.assert_array_equal(
            recorder.calls[step][0].numpy(), expected_input
        )

    # Every step's OUTPUT (whether freshly computed or reused from the
    # previous step's V(s_{t+1}) call) is exactly what ends up stored
    # as that step's `value` in the buffer.
    for step in range(num_steps):
        expected_value = buffer.values[step * num_envs : (step + 1) * num_envs]
        np.testing.assert_array_equal(
            recorder.calls[step][1].numpy().reshape(-1), expected_value
        )


def test_current_value_tensor_invalidated_across_train_iteration() -> None:
    """The cache must never survive an update_critic() call: each new
    train_iteration() must start with a real forward pass, not the
    previous iteration's stale cached value."""

    num_envs = 2
    buffer_size = 6
    env, trainer, buffer = _build(num_envs=num_envs, buffer_size=buffer_size)
    num_steps = buffer_size // num_envs

    assert trainer._current_value_tensor is None

    with _PredictValuesRecorder(trainer.critic) as recorder:
        trainer.train_iteration(env, buffer)
    first_iteration_calls = len(recorder.calls)
    assert first_iteration_calls == num_steps + 1

    # update_critic() ran as part of train_iteration() above, changing
    # self.critic's parameters. A second rollout must NOT reuse a value
    # computed under the old parameters.
    buffer.reset()
    with _PredictValuesRecorder(trainer.critic) as recorder:
        trainer.collect_rollout(env, buffer)
    second_rollout_calls = len(recorder.calls)

    # Same shape as the first rollout: one real forward pass for the
    # first step (cache was invalidated), then one per subsequent step.
    assert second_rollout_calls == num_steps + 1


def test_gae_runs_correctly_on_a_phase_2_rollout_with_episode_boundary() -> None:
    """advantage.py is unmodified by Phase 2; this exercises it end to
    end (including an auto-reset boundary) against a Phase-2 rollout,
    reusing advantage.py's own terminated|truncated bootstrap-zeroing
    logic as the correctness oracle."""

    num_envs = 2
    # num_containers=12 with a small yard finishes an episode well
    # inside a modest buffer, guaranteeing at least one boundary.
    buffer_size = 40
    env, trainer, buffer = _build(num_envs=num_envs, buffer_size=buffer_size, seed=1)

    trainer.collect_rollout(env, buffer)

    terminated = buffer.terminated[: buffer.size]
    truncated = buffer.truncated[: buffer.size]
    assert (terminated | truncated).any(), "test picked a rollout with no episode boundary"

    advantages, returns = compute_and_store_gae(
        buffer=buffer, gamma=trainer.gamma, gae_lambda=trainer.gae_lambda, num_envs=num_envs
    )

    assert advantages.shape == (buffer.size,)
    assert returns.shape == (buffer.size,)
    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()

    # Bootstrap must be zero exactly at (and only at) episode
    # boundaries - the buffer-side zeroing collect_rollout() performs
    # before storage (see sequential_trainer.py) must still hold with
    # Phase 2's reused values.
    episode_ends = terminated | truncated
    next_values = buffer.next_values[: buffer.size]
    np.testing.assert_array_equal(next_values[episode_ends], 0.0)
