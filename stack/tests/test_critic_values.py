"""
Critic values stored by a rollout: they are the current critic's
estimates (never stale ones), and GAE does not bootstrap across episodes.
"""

from dataclasses import replace

import torch

from stack.configs.environments import set_config
from stack.configs.hierarchical_config import get_hierarchical_config
from stack.run_sequential_hppo import build_training_system
from stack.training.rollout_buffer import JointRolloutBuffer

ENVIRONMENT = dict(set_config("small_with_margin", seed=1), vessel_shape=(3, 3, 2), yard_shape=(3, 3, 2),
                   num_containers=12)
ALGORITHM = replace(get_hierarchical_config("small"), embed_dim=16, n_heads=2, n_layers=1, vf_dim=16,
                    dropout=0.0, buffer_size=40, batch_size=10, n_epochs=2, learning_rate=1e-2)
NUM_ENVS = 2


def _setup():
    torch.manual_seed(0)
    env, _bay, _row, _critic, trainer = build_training_system(
        ENVIRONMENT, ALGORITHM, torch.device("cpu"), num_envs=NUM_ENVS
    )
    return env, trainer


def _buffer(trainer):
    return JointRolloutBuffer(trainer.layout, ALGORITHM.buffer_size, NUM_ENVS, "cpu")


def _critic_values(trainer, states):
    with torch.no_grad():
        return torch.stack([trainer.critic.predict_values(step_states) for step_states in states])


def test_stored_values_are_the_rollout_critics_estimates() -> None:
    env, trainer = _setup()
    buffer = _buffer(trainer)
    trainer.collect_rollout(env, buffer)
    env.close()

    # V(s_t) for every step and environment ...
    torch.testing.assert_close(buffer.values, _critic_values(trainer, buffer.global_states), rtol=0, atol=0)
    # ... and V(s_{t+1}) is the next step's V(s_t), including right after
    # an auto-reset, where s_{t+1} is the new episode's first state.
    torch.testing.assert_close(buffer.next_values[:-1], buffer.values[1:], rtol=0, atol=0)


def test_values_after_a_critic_update_come_from_the_updated_critic() -> None:
    """The stale-value guard: a rollout that follows a critic update must
    not reuse any value computed by the critic before that update."""

    env, trainer = _setup()
    trainer.train_iteration(env, _buffer(trainer))  # updates the critic

    buffer = _buffer(trainer)
    trainer.collect_rollout(env, buffer)
    env.close()

    updated = _critic_values(trainer, buffer.global_states)
    torch.testing.assert_close(buffer.values, updated, rtol=0, atol=0)


def test_gae_does_not_bootstrap_across_an_auto_reset() -> None:
    env, trainer = _setup()
    buffer = _buffer(trainer)
    trainer.collect_rollout(env, buffer)
    env.close()

    dones = buffer.dones
    assert dones.any(), "no episode ended inside the rollout"
    # The stored next value at a boundary is the reset state's value ...
    assert (buffer.next_values[dones] != 0).any()

    buffer.compute_gae(trainer.gamma, trainer.gae_lambda)
    # ... yet the last step of an episode gets A_t = r_t - V(s_t).
    torch.testing.assert_close(
        buffer.advantages[dones], buffer.rewards[dones] - buffer.values[dones], rtol=0, atol=0
    )
