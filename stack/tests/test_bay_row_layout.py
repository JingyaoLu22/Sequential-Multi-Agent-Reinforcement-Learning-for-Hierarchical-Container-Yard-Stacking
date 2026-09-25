"""
BayRowLayout must reproduce StackEnv's own bay/row semantics exactly.

Ground truth is always StackEnv itself - _get_valid_yard_actions(),
_bay_row_to_action() and the bay-major stack_features_v3 observation -
never the layout's own formulas:

A. Bay mask: a bay is valid iff StackEnv has a valid action in it.
B. Row mask: row r of bay b is valid iff StackEnv's action for
   (physical bay 2b+1, physical row r+1) is valid.
C. Row observation: exactly bay b's stack tokens of the global
   observation, unchanged.
D. Action mapping: (bay, row) -> StackEnv action matches
   StackEnv._bay_row_to_action().
"""

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from stack.envs.stack_gym import StackEnv
from stack.training.bay_row_layout import BayRowLayout


def _make_config(seed: int, container_sizes: bool, enable_imo: bool, **overrides) -> dict:
    config = {
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
        "container_sizes": container_sizes,
        "enable_imo": enable_imo,
    }
    config.update(overrides)
    return config


def _assert_matches_stack_env(env: StackEnv, layout: BayRowLayout) -> None:
    observation = torch.as_tensor(env._create_observation()).unsqueeze(0)
    valid_actions = set(int(a) for a in env._get_valid_yard_actions())
    action_mask = torch.as_tensor(env.action_masks()).unsqueeze(0)

    bay_mask = layout.bay_masks(action_mask)[0].numpy()
    tokens = observation.numpy().reshape(layout.n_bays, layout.n_rows, -1)

    for bay_idx in range(layout.n_bays):
        expected_row_mask = np.array(
            [
                env._bay_row_to_action(2 * bay_idx + 1, row_idx + 1) in valid_actions
                for row_idx in range(layout.n_rows)
            ]
        )

        # A.
        assert bay_mask[bay_idx] == expected_row_mask.any()

        row_observation, row_mask = layout.row_inputs(
            observation, action_mask, torch.tensor([bay_idx])
        )

        # B.
        np.testing.assert_array_equal(row_mask[0].numpy(), expected_row_mask)

        # C.
        np.testing.assert_array_equal(
            row_observation[0].numpy(), tokens[bay_idx].reshape(-1)
        )


def _random_valid_action(env: StackEnv, rng: np.random.Generator) -> int:
    valid_actions = env._get_valid_yard_actions()
    assert valid_actions.size > 0
    return int(rng.choice(valid_actions))


@pytest.mark.parametrize(
    "container_sizes,enable_imo",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_layout_matches_stack_env_along_trajectories(
    container_sizes: bool, enable_imo: bool
) -> None:
    rng = np.random.default_rng(0)

    for episode_seed in range(5):
        env = StackEnv(config=_make_config(episode_seed, container_sizes, enable_imo))
        layout = BayRowLayout.from_env(env)
        env.reset(seed=episode_seed)
        _assert_matches_stack_env(env, layout)

        for _ in range(50):
            _, _, terminated, truncated, _ = env.step(_random_valid_action(env, rng))
            if terminated or truncated:
                env.reset()
            _assert_matches_stack_env(env, layout)


def test_masks_and_mapping_on_the_bay_row_grid() -> None:
    """On the 2-D view mask_2d = action_mask.reshape(n_bays, n_rows):
    bay mask = mask_2d.any(axis=1), row mask = mask_2d[selected_bay],
    global action = bay * n_rows + row - batched, each environment with
    its own selected bay."""

    layout = BayRowLayout.from_env(StackEnv(config=_make_config(0, False, False)))
    n_envs = 4
    obs_dim = layout.observation_space.shape[0]

    torch.manual_seed(0)
    observations = torch.arange(n_envs * obs_dim, dtype=torch.float32).reshape(n_envs, -1)
    action_masks = torch.rand(n_envs, layout.n_bays * layout.n_rows) > 0.5
    bay_actions = torch.tensor([2, 0, 1, 2])
    row_actions = torch.tensor([0, 2, 1, 1])

    bay_masks = layout.bay_masks(action_masks)
    row_observations, row_masks = layout.row_inputs(observations, action_masks, bay_actions)
    global_actions = layout.stack_actions(bay_actions, row_actions)

    for i, (bay, row) in enumerate(zip(bay_actions.tolist(), row_actions.tolist())):
        mask_2d = action_masks[i].numpy().reshape(layout.n_bays, layout.n_rows)
        np.testing.assert_array_equal(bay_masks[i].numpy(), mask_2d.any(axis=1))
        np.testing.assert_array_equal(row_masks[i].numpy(), mask_2d[bay])
        np.testing.assert_array_equal(
            row_observations[i].numpy(), observations[i].numpy().reshape(layout.n_bays, -1)[bay]
        )
        assert global_actions[i] == bay * layout.n_rows + row
        # The selected (bay, row) cell is the same entry of the flat mask.
        assert action_masks[i, global_actions[i]] == mask_2d[bay, row]


def test_stack_actions_match_stack_env_mapping() -> None:
    """D, for a non-square yard so bay/row cannot be swapped silently."""

    env = StackEnv(config=_make_config(5, False, False, yard_shape=(4, 5, 2)))
    layout = BayRowLayout.from_env(env)
    assert (layout.n_bays, layout.n_rows) == (4, 5)

    bays, rows = np.meshgrid(np.arange(layout.n_bays), np.arange(layout.n_rows), indexing="ij")
    actions = layout.stack_actions(bays.ravel(), rows.ravel())

    for bay_idx, row_idx, action in zip(bays.ravel(), rows.ravel(), actions):
        assert action == env._bay_row_to_action(2 * bay_idx + 1, row_idx + 1)
        assert env._action_to_bay_row(int(action)) == (2 * bay_idx + 1, row_idx + 1)


def test_from_vec_env_matches_from_stack_env() -> None:
    config = _make_config(1, True, True)
    single = BayRowLayout.from_env(StackEnv(config=config))

    vec_env = DummyVecEnv([lambda: StackEnv(config=config)] * 2)
    try:
        batched = BayRowLayout.from_env(vec_env)
    finally:
        vec_env.close()

    assert (batched.n_bays, batched.n_rows) == (single.n_bays, single.n_rows)
    assert batched.observation_space == single.observation_space
    assert batched.row_observation_space == single.row_observation_space
    # 5 base + 3 container-size + 2 IMO features per stack.
    assert single.row_observation_space.shape == (single.n_rows * 10,)


def test_rejects_observations_that_are_not_stack_tokens() -> None:
    env = StackEnv(config=_make_config(0, False, False, observation_type="flat"))

    with pytest.raises(ValueError, match="stack_features_v3"):
        BayRowLayout.from_env(env)
