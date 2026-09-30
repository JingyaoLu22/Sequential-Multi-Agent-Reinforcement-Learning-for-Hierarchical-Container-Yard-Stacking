"""
The sequential (HAPPO-style) update, critic -> Bay actor -> correction
M_B -> Row actor with M_B * A, and the critic values it trains on.
"""

import copy

import torch

from stack.training.rollout_buffer import JointRolloutBuffer

from .common import algorithm, environment, training_system

ALGORITHM = algorithm(buffer_size=40, batch_size=10, n_epochs=2, learning_rate=1e-2)


def _setup():
    env, trainer = training_system(environment(1), ALGORITHM)
    return env, trainer, JointRolloutBuffer(trainer.layout, ALGORITHM.buffer_size, 2, "cpu")


def test_row_update_uses_the_correction_of_the_updated_bay_actor() -> None:
    env, trainer, buffer = _setup()

    # Record each _ppo_actor_update call; for the Row call, also Agent B's
    # log-probabilities of the stored Bay actions at that moment.
    calls = []
    original = trainer._ppo_actor_update

    def spy(name, *args, advantage_scale=None):
        call = {"name": name, "args": args, "advantage_scale": advantage_scale}
        if name == "row":
            with torch.no_grad():
                call["bay_log_probs"], _ = trainer.bay_actor.evaluate_actions(*calls[0]["args"][2:5])
        calls.append(call)
        return original(name, *args, advantage_scale=advantage_scale)

    trainer._ppo_actor_update = spy
    trainer.train_iteration(env, buffer)
    env.close()

    bay_call, row_call = calls
    assert (bay_call["name"], row_call["name"]) == ("bay", "row")
    assert bay_call["advantage_scale"] is None

    # M_B = pi_B,updated / pi_B,old of the stored Bay actions, detached.
    expected = torch.exp(row_call["bay_log_probs"] - bay_call["args"][5])
    torch.testing.assert_close(row_call["advantage_scale"], expected, rtol=1e-5, atol=1e-6)
    assert not row_call["advantage_scale"].requires_grad
    assert not torch.allclose(expected, torch.ones_like(expected)), "the Bay update did not change pi_B"


def test_each_update_changes_only_its_own_network() -> None:
    env, trainer, buffer = _setup()
    trainer.collect_rollout(env, buffer)
    env.close()
    buffer.compute_gae(trainer.gamma, trainer.gae_lambda)
    rollout = buffer.rollout_batch()
    networks = {"bay": trainer.bay_actor, "row": trainer.row_actor, "critic": trainer.critic}

    def changed_by(update) -> set:
        before = {name: copy.deepcopy(net.state_dict()) for name, net in networks.items()}
        update()
        return {name for name, net in networks.items()
                if any(not torch.equal(before[name][k], v) for k, v in net.state_dict().items())}

    bay_forward_calls = []
    trainer.bay_actor.register_forward_hook(lambda *_: bay_forward_calls.append(1))

    assert changed_by(lambda: trainer.update_critic(rollout)) == {"critic"}
    assert changed_by(lambda: trainer._ppo_actor_update(
        "bay", trainer.bay_actor, trainer.bay_optimizer, rollout.global_states, rollout.bay_actions,
        rollout.bay_action_masks, rollout.old_bay_log_probs, rollout.advantages)) == {"bay"}

    correction = trainer.bay_correction_ratio(rollout)
    bay_forward_calls.clear()
    assert changed_by(lambda: trainer._ppo_actor_update(
        "row", trainer.row_actor, trainer.row_optimizer, rollout.row_observations, rollout.row_actions,
        rollout.row_action_masks, rollout.old_row_log_probs, rollout.advantages,
        advantage_scale=correction)) == {"row"}
    assert bay_forward_calls == [], "the Row update must not run Agent B"


def test_values_after_a_critic_update_come_from_the_updated_critic() -> None:
    """The stale-value guard: a rollout after a critic update reuses no
    value the critic computed before that update."""

    env, trainer, buffer = _setup()
    trainer.train_iteration(env, buffer)  # updates the critic
    trainer.collect_rollout(env, buffer)
    env.close()

    with torch.no_grad():
        current = torch.stack([trainer.critic(states) for states in buffer.global_states])
    assert torch.equal(buffer.values, current)
    # V(s_{t+1}) is the next step's V(s_t), also right after an auto-reset.
    assert torch.equal(buffer.next_values[:-1], buffer.values[1:])


def test_gae_does_not_bootstrap_across_an_auto_reset() -> None:
    env, trainer, buffer = _setup()
    trainer.collect_rollout(env, buffer)
    env.close()

    dones = buffer.dones
    assert dones.any(), "no episode ended inside the rollout"
    buffer.compute_gae(trainer.gamma, trainer.gae_lambda)
    # The last step of an episode gets A_t = r_t - V(s_t), although the
    # stored next value is the reset state's.
    assert torch.equal(buffer.advantages[dones], buffer.rewards[dones] - buffer.values[dones])
