"""
Bay/row masks, row observations and the action mapping reproduce
StackEnv's own semantics, and select_hierarchical_action() composes them.
Ground truth is StackEnv itself (_get_valid_yard_actions, _bay_row_to_action).
"""

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from stack.envs.stack_gym import StackEnv
from stack.training.bay_row_layout import BayRowLayout, select_hierarchical_action

from .common import environment


@pytest.mark.parametrize("container_sizes,enable_imo", [(False, False), (True, True)])
def test_masks_and_row_observations_match_stack_env(container_sizes: bool, enable_imo: bool) -> None:
    rng = np.random.default_rng(0)
    env = StackEnv(config=environment(0, container_sizes=container_sizes, enable_imo=enable_imo))
    layout = BayRowLayout.from_env(env)
    env.reset(seed=0)

    for _ in range(60):  # five episodes
        observation = torch.as_tensor(env._create_observation()).unsqueeze(0)
        mask = torch.as_tensor(env.action_masks()).unsqueeze(0)
        valid = set(env._get_valid_yard_actions().tolist())
        tokens = observation.reshape(layout.n_bays, layout.n_rows, -1)

        for bay in range(layout.n_bays):
            expected_row_mask = [env._bay_row_to_action(2 * bay + 1, row + 1) in valid for row in range(layout.n_rows)]
            row_observation, row_mask = layout.row_inputs(observation, mask, torch.tensor([bay]))

            assert layout.bay_masks(mask)[0, bay] == any(expected_row_mask)
            assert row_mask[0].tolist() == expected_row_mask
            assert torch.equal(row_observation[0], tokens[bay].reshape(-1))

        _, _, terminated, truncated, _ = env.step(int(rng.choice(sorted(valid))))
        if terminated or truncated:
            env.reset()


def test_action_mapping_matches_stack_env() -> None:
    # Non-square yard, so bays and rows cannot be swapped silently.
    env = StackEnv(config=environment(0, yard_shape=(4, 5, 2)))
    layout = BayRowLayout.from_env(env)

    for bay in range(layout.n_bays):
        for row in range(layout.n_rows):
            action = layout.stack_actions(bay, row)
            assert action == bay * layout.n_rows + row == env._bay_row_to_action(2 * bay + 1, row + 1)


def _decision_inputs(n_envs: int = 4):
    env = DummyVecEnv([lambda: StackEnv(config=environment())] * n_envs)
    env.seed(3)
    observations = torch.as_tensor(env.reset(), dtype=torch.float32)
    masks = np.stack(env.env_method("action_masks"))
    layout = BayRowLayout.from_env(env)
    env.close()
    torch.manual_seed(0)
    bay_actor, row_actor = layout.build_actors(embed_dim=16, n_heads=2, n_layers=1, dropout=0.0)
    return layout, bay_actor.eval(), row_actor.eval(), observations, masks


def test_hierarchical_decision_is_consistent_with_stack_env() -> None:
    layout, bay_actor, row_actor, observations, masks = _decision_inputs()
    action = select_hierarchical_action(layout, bay_actor, row_actor, observations, masks)
    masks = torch.as_tensor(masks)

    # Each StackEnv action is valid and is the chosen (bay, row) ...
    stack_actions = torch.as_tensor(action.stack_actions)
    assert masks.gather(1, stack_actions[:, None]).all()
    assert torch.equal(stack_actions, action.bay_actions * layout.n_rows + action.row_actions)

    # ... Agent R saw exactly the chosen bay's slice and mask ...
    expected_observations, expected_masks = layout.row_inputs(observations, masks, action.bay_actions)
    assert torch.equal(action.row_observations, expected_observations)
    assert torch.equal(action.row_masks, expected_masks)

    # ... and the stored log-probabilities are what PPO re-evaluates.
    bay_log_probs, _ = bay_actor.evaluate_actions(observations, action.bay_actions, action.bay_masks)
    row_log_probs, _ = row_actor.evaluate_actions(action.row_observations, action.row_actions, action.row_masks)
    torch.testing.assert_close(action.bay_log_probs, bay_log_probs)
    torch.testing.assert_close(action.row_log_probs, row_log_probs)


def test_rejects_state_without_valid_action() -> None:
    layout, bay_actor, row_actor, observations, masks = _decision_inputs()
    masks[2] = False

    with pytest.raises(RuntimeError, match="no valid action"):
        select_hierarchical_action(layout, bay_actor, row_actor, observations, masks)
