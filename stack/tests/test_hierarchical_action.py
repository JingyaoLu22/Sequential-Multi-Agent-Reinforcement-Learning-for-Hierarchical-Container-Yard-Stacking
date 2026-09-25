"""
select_hierarchical_action() is the single Bay -> Row decision, and both
training rollouts and evaluation go through it.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from stack.configs.hierarchical_config import get_hierarchical_config
from stack.envs.stack_gym import StackEnv
from stack.evaluation import evaluate
from stack.run_sequential_hppo import build_training_system
from stack.training import bay_row_layout, sequential_trainer
from stack.training.bay_row_layout import BayRowLayout, select_hierarchical_action
from stack.training.rollout_buffer import JointRolloutBuffer

CONFIG = {
    "vessel_shape": (3, 3, 2),
    "yard_shape": (3, 3, 2),
    "num_containers": 12,
    "group_num": 3,
    "group_placement": "random",
    "seed": 0,
    "observation_type": "stack_features_v3",
    "reward_norm": True,
    "reward_clip": True,
    "stack_fill_penalty": True,
    "container_sizes": True,
    "enable_imo": True,
}


def _decision_inputs(n_envs: int = 4):
    env = DummyVecEnv([lambda: StackEnv(config=CONFIG)] * n_envs)
    env.seed(3)
    observations = torch.as_tensor(env.reset(), dtype=torch.float32)
    masks = np.stack(env.env_method("action_masks"))
    layout = BayRowLayout.from_env(env)
    env.close()
    torch.manual_seed(0)
    bay_actor, row_actor = layout.build_actors(embed_dim=16, n_heads=2, n_layers=1, dropout=0.0)
    return layout, bay_actor.eval(), row_actor.eval(), observations, masks


@pytest.mark.parametrize("deterministic", [True, False])
def test_decision_is_consistent_with_stack_env(deterministic: bool) -> None:
    layout, bay_actor, row_actor, observations, masks = _decision_inputs()

    torch.manual_seed(1)
    action = select_hierarchical_action(layout, bay_actor, row_actor, observations, masks, deterministic)
    masks = torch.as_tensor(masks)

    # Every chosen StackEnv action is valid under StackEnv's own mask, and
    # decomposes into the chosen (bay, row).
    stack_actions = torch.as_tensor(action.stack_actions)
    assert masks.gather(1, stack_actions[:, None]).all()
    torch.testing.assert_close(stack_actions, action.bay_actions * layout.n_rows + action.row_actions)

    # Agent R saw exactly the chosen bay's observation slice and mask.
    expected_obs, expected_mask = layout.row_inputs(observations, masks, action.bay_actions)
    torch.testing.assert_close(action.row_observations, expected_obs, rtol=0, atol=0)
    assert torch.equal(action.row_masks, expected_mask)

    # Stored log-probabilities are what PPO later re-evaluates.
    bay_log_probs, _ = bay_actor.evaluate_actions(observations, action.bay_actions, action.bay_masks)
    row_log_probs, _ = row_actor.evaluate_actions(action.row_observations, action.row_actions, action.row_masks)
    torch.testing.assert_close(action.bay_log_probs, bay_log_probs)
    torch.testing.assert_close(action.row_log_probs, row_log_probs)


def test_rejects_state_without_valid_action() -> None:
    layout, bay_actor, row_actor, observations, masks = _decision_inputs()
    masks[2] = False

    with pytest.raises(RuntimeError, match="no valid action"):
        select_hierarchical_action(layout, bay_actor, row_actor, observations, masks)


def test_training_and_evaluation_share_the_decision(monkeypatch) -> None:
    calls = {"training": 0, "evaluation": 0}

    def counting(module_name):
        def wrapper(*args, **kwargs):
            calls[module_name] += 1
            return bay_row_layout.select_hierarchical_action(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(sequential_trainer, "select_hierarchical_action", counting("training"))
    monkeypatch.setattr(evaluate, "select_hierarchical_action", counting("evaluation"))

    algorithm_config = replace(
        get_hierarchical_config("small"), embed_dim=16, n_heads=2, n_layers=1, vf_dim=16, dropout=0.0
    )
    env, bay_actor, row_actor, _critic, trainer = build_training_system(
        CONFIG, algorithm_config, device=torch.device("cpu"), num_envs=2
    )
    trainer.collect_rollout(env, JointRolloutBuffer(trainer.layout, 8, 2, "cpu"))
    env.close()
    evaluate.evaluate_policy(
        environment_config=CONFIG, bay_actor=bay_actor, row_actor=row_actor, n_episodes=1, verbose=False
    )

    assert calls["training"] == 4  # one per rollout step
    assert calls["evaluation"] > 0
