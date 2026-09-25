"""
The sequential (HAPPO-style) update: critic -> Bay actor -> correction
M_B -> Row actor with M_B * A.
"""

import copy
from dataclasses import replace

import torch

from stack.configs.environments import set_config
from stack.configs.hierarchical_config import get_hierarchical_config
from stack.run_sequential_hppo import build_training_system
from stack.training.rollout_buffer import JointRolloutBuffer

ENVIRONMENT = dict(set_config("small_with_margin", seed=2), vessel_shape=(3, 3, 2), yard_shape=(3, 3, 2),
                   num_containers=12)
ALGORITHM = replace(get_hierarchical_config("small"), embed_dim=16, n_heads=2, n_layers=1, vf_dim=16,
                    dropout=0.0, buffer_size=16, batch_size=4, n_epochs=3, learning_rate=1e-2)


def _trainer(**overrides):
    torch.manual_seed(0)
    env, _bay, _row, _critic, trainer = build_training_system(
        ENVIRONMENT, replace(ALGORITHM, **overrides), torch.device("cpu"), num_envs=2
    )
    return env, trainer


def _rollout(env, trainer):
    buffer = JointRolloutBuffer(trainer.layout, 16, 2, "cpu")
    trainer.collect_rollout(env, buffer)
    buffer.compute_gae(trainer.gamma, trainer.gae_lambda)
    return buffer.rollout_batch()


def _record_actor_updates(trainer):
    """Spy on _ppo_actor_update(name, actor, optimizer, observations,
    actions, masks, old_log_probs, advantages, advantage_scale=None).
    For the Row call, also record Agent B's log-probabilities of the
    stored Bay actions at that moment."""

    calls = []
    original = trainer._ppo_actor_update

    def spy(name, *args, advantage_scale=None):
        call = {"name": name, "args": args, "advantage_scale": advantage_scale}
        if name == "row":
            bay_observations, bay_actions, bay_masks = calls[0]["args"][2:5]
            with torch.no_grad():
                call["bay_log_probs"], _ = trainer.bay_actor.evaluate_actions(bay_observations, bay_actions, bay_masks)
        calls.append(call)
        return original(name, *args, advantage_scale=advantage_scale)

    trainer._ppo_actor_update = spy
    return calls


def test_row_update_uses_the_correction_of_the_updated_bay_actor() -> None:
    env, trainer = _trainer()
    calls = _record_actor_updates(trainer)
    trainer.train_iteration(env, JointRolloutBuffer(trainer.layout, 16, 2, "cpu"))
    env.close()

    bay_call, row_call = calls
    assert (bay_call["name"], row_call["name"]) == ("bay", "row")
    assert bay_call["advantage_scale"] is None  # Agent B uses the plain advantage

    # M_B = pi_B,updated / pi_B,old for the stored Bay actions, where
    # "updated" is Agent B right after its own update.
    old_bay_log_probs = bay_call["args"][5]
    expected = torch.exp(row_call["bay_log_probs"] - old_bay_log_probs)
    torch.testing.assert_close(row_call["advantage_scale"], expected, rtol=1e-5, atol=1e-6)
    assert not row_call["advantage_scale"].requires_grad

    # Not stale: the Bay update really changed pi_B, so M_B is not all 1.
    assert not torch.allclose(row_call["advantage_scale"], torch.ones_like(expected))


def test_row_objective_is_scaled_by_the_correction() -> None:
    """With M_B = 0 and no entropy bonus the Row objective is zero, so the
    Row actor must not move; with M_B = 1 it must."""

    for scale, should_move in ((0.0, False), (1.0, True)):
        env, trainer = _trainer(ent_coef=0.0)
        rollout = _rollout(env, trainer)
        env.close()
        before = copy.deepcopy(trainer.row_actor.state_dict())

        trainer._ppo_actor_update(
            "row", trainer.row_actor, trainer.row_optimizer, rollout.row_observations, rollout.row_actions,
            rollout.row_action_masks, rollout.old_row_log_probs, rollout.advantages,
            advantage_scale=torch.full_like(rollout.advantages, scale),
        )

        moved = any(not torch.equal(before[k], v) for k, v in trainer.row_actor.state_dict().items())
        assert moved == should_move, scale


def test_each_update_changes_only_its_own_network() -> None:
    env, trainer = _trainer()
    rollout = _rollout(env, trainer)
    env.close()

    def snapshot():
        return {name: copy.deepcopy(module.state_dict())
                for name, module in (("bay", trainer.bay_actor), ("row", trainer.row_actor), ("critic", trainer.critic))}

    def changed(before):
        after = snapshot()
        return {name for name in before if any(not torch.equal(before[name][k], after[name][k]) for k in before[name])}

    bay_forward_calls = []
    hook = trainer.bay_actor.register_forward_hook(lambda *_: bay_forward_calls.append(1))

    before = snapshot()
    trainer.update_critic(rollout)
    assert changed(before) == {"critic"}

    before = snapshot()
    trainer._ppo_actor_update("bay", trainer.bay_actor, trainer.bay_optimizer, rollout.global_states,
                              rollout.bay_actions, rollout.bay_action_masks, rollout.old_bay_log_probs,
                              rollout.advantages)
    assert changed(before) == {"bay"}

    correction = trainer.bay_correction_ratio(rollout)

    before = snapshot()
    bay_forward_calls.clear()
    trainer._ppo_actor_update("row", trainer.row_actor, trainer.row_optimizer, rollout.row_observations,
                              rollout.row_actions, rollout.row_action_masks, rollout.old_row_log_probs,
                              rollout.advantages, advantage_scale=correction)
    assert changed(before) == {"row"}
    assert bay_forward_calls == [], "the Row update must not run Agent B"
    hook.remove()
