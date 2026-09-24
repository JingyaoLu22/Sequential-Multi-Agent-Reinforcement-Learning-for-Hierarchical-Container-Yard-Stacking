"""
Phase 1 equivalence tests for HierarchicalEnv's StateSnapshot cache.

See docs/plan §7, tests A-E:

A. Snapshot equivalence - cached observation == freshly recomputed
   observation.
B. Mask equivalence - cached global mask == existing valid-action
   implementation.
C. Bay mask derivation - bay valid iff at least one row inside the bay
   is valid.
D. Row mask derivation - row mask exactly equals the selected bay
   slice of the global mask.
E. Action mapping - (bay, row) -> global action is correct.

BayEnv/RowEnv are left completely unmodified by Phase 1 specifically so
they can serve as the ground-truth "recompute from scratch" reference
here.
"""

import itertools

import numpy as np
import pytest

from stack.envs.hierarchical_envs.hierarchical_env import HierarchicalEnv


def _make_config(seed: int, container_sizes: bool, enable_imo: bool) -> dict:
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
        "container_sizes": container_sizes,
        "enable_imo": enable_imo,
    }


def _assert_snapshot_matches_fresh(env: HierarchicalEnv) -> None:
    """Test A + B at the current state: cached vs. freshly recomputed."""

    # A. Global observation.
    cached_global = env.get_global_state()
    fresh_global = env.bay_env.get_observation()
    np.testing.assert_array_equal(cached_global, fresh_global)

    # B. Bay mask.
    cached_bay_mask = env.get_bay_action_mask()
    fresh_bay_mask = np.asarray(env.bay_env.action_masks(), dtype=bool)
    np.testing.assert_array_equal(cached_bay_mask, fresh_bay_mask)

    # B + C. Bay mask == "any row in this bay is valid" for every bay,
    # cached or fresh.
    for bay_idx in range(env.bay_action_space.n):
        fresh_row_mask = np.asarray(env.row_env.action_masks(bay_idx), dtype=bool)
        assert fresh_bay_mask[bay_idx] == fresh_row_mask.any()

        if not cached_bay_mask[bay_idx]:
            continue

        # A + B + D. Row observation/mask for every valid bay.
        cached_row_obs = env.get_row_observation(bay_idx)
        fresh_row_obs = env.row_env.get_observation(bay_idx)
        np.testing.assert_array_equal(cached_row_obs, fresh_row_obs)

        cached_row_mask = env.get_row_action_mask(bay_idx)
        np.testing.assert_array_equal(cached_row_mask, fresh_row_mask)


def _random_valid_action(env: HierarchicalEnv, rng: np.random.Generator):
    bay_mask = env.get_bay_action_mask()
    valid_bays = np.flatnonzero(bay_mask)
    assert valid_bays.size > 0, "test picked a state with no valid bay"
    bay_idx = int(rng.choice(valid_bays))

    row_mask = env.get_row_action_mask(bay_idx)
    valid_rows = np.flatnonzero(row_mask)
    assert valid_rows.size > 0, "Agent B selected a bay with no valid row"
    row_idx = int(rng.choice(valid_rows))

    return bay_idx, row_idx


@pytest.mark.parametrize(
    "container_sizes,enable_imo",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_snapshot_matches_fresh_recompute_across_episodes(
    container_sizes: bool, enable_imo: bool
) -> None:
    rng = np.random.default_rng(0)

    for episode_seed in range(5):
        env = HierarchicalEnv(
            config=_make_config(episode_seed, container_sizes, enable_imo)
        )
        env.reset(seed=episode_seed)
        _assert_snapshot_matches_fresh(env)

        for _ in range(50):
            bay_idx, row_idx = _random_valid_action(env, rng)
            _, _, terminated, truncated, _ = env.step(bay_idx, row_idx)
            _assert_snapshot_matches_fresh(env)
            if terminated or truncated:
                env.reset()
                _assert_snapshot_matches_fresh(env)


def test_bay_mask_is_any_row_valid_in_bay() -> None:
    """Test C, directly against the cached StateSnapshot's mask_2d."""

    rng = np.random.default_rng(1)
    env = HierarchicalEnv(config=_make_config(2, False, False))
    env.reset(seed=2)

    for _ in range(30):
        snapshot = env._current_snapshot()
        np.testing.assert_array_equal(
            snapshot.bay_mask, snapshot.global_mask_2d.any(axis=1)
        )

        bay_idx, row_idx = _random_valid_action(env, rng)
        _, _, terminated, truncated, _ = env.step(bay_idx, row_idx)
        if terminated or truncated:
            env.reset()


def test_row_mask_is_selected_bay_slice_of_global_mask() -> None:
    """Test D, directly against the cached StateSnapshot's mask_2d."""

    rng = np.random.default_rng(3)
    env = HierarchicalEnv(config=_make_config(4, True, True))
    env.reset(seed=4)

    for _ in range(30):
        snapshot = env._current_snapshot()
        for bay_idx in range(env.bay_action_space.n):
            np.testing.assert_array_equal(
                env.get_row_action_mask(bay_idx),
                snapshot.global_mask_2d[bay_idx],
            )

        bay_idx, row_idx = _random_valid_action(env, rng)
        _, _, terminated, truncated, _ = env.step(bay_idx, row_idx)
        if terminated or truncated:
            env.reset()


def test_action_index_round_trip_and_to_global_action() -> None:
    """Test E: (bay, row) <-> global action index round-trips, and
    HierarchicalEnv.to_global_action matches StackEnv's own formula."""

    env = HierarchicalEnv(config=_make_config(5, False, False))
    env.reset(seed=5)

    stack_env = env.inner_env
    num_bays = env.bay_action_space.n
    num_rows = env.row_action_space.n

    physical_bays = [2 * b + 1 for b in range(num_bays)]
    physical_rows = [r + 1 for r in range(num_rows)]

    for physical_bay, physical_row in itertools.product(physical_bays, physical_rows):
        action = stack_env._bay_row_to_action(physical_bay, physical_row)
        round_tripped_bay, round_tripped_row = stack_env._action_to_bay_row(action)
        assert (round_tripped_bay, round_tripped_row) == (physical_bay, physical_row)

    # Documented example (hierarchical_env.py): n_rows=4, bay_idx=2,
    # row_idx=3 -> physical bay=5, physical row=4. Reproduce it at
    # whatever num_rows this fixture has by checking the same formula
    # HierarchicalEnv.to_global_action delegates to.
    for bay_idx in range(num_bays):
        for row_idx in range(num_rows):
            expected_physical_bay = 2 * bay_idx + 1
            expected_physical_row = row_idx + 1
            expected_action = stack_env._bay_row_to_action(
                expected_physical_bay, expected_physical_row
            )
            assert env.to_global_action(bay_idx, row_idx) == expected_action
            # action = bay_idx * num_rows + row_idx, per
            # StateSnapshot's docstring.
            assert expected_action == bay_idx * num_rows + row_idx
