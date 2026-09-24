"""
Equivalence tests between VecHierarchicalEnv (in-process for-loop) and
SubprocVecHierarchicalEnv (one OS process per environment).

Both classes expose the identical public interface (see
subproc_vec_hierarchical_env.py's module docstring) and, given the
same config/base_seed and the same sequence of actions, StackEnv's
transitions are deterministic - so driving both vec envs with the
same randomly-chosen valid actions must produce byte-identical
observations/masks/rewards/termination flags at every step. This is
the correctness guarantee for the multiprocessing rewrite: it is a
parallelism change, not a semantics change.
"""

import numpy as np
import pytest

from stack.envs.hierarchical_envs.vec_hierarchical_env import (
    VecHierarchicalEnv,
)
from stack.envs.hierarchical_envs.subproc_vec_hierarchical_env import (
    SubprocVecHierarchicalEnv,
)


def _make_config(seed: int) -> dict:
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


def _pick_actions(bay_mask: np.ndarray, row_obs_mask_fn, rng: np.random.Generator):
    """
    Choose one random valid (bay, row) pair per environment slot.

    row_obs_mask_fn(bay_actions) must return (row_obs, row_mask) for
    the given per-env bay_actions - i.e. it is
    vec_env.get_row_decision_input bound to a fixed vec_env.
    """

    num_envs = bay_mask.shape[0]
    bay_actions = np.empty(num_envs, dtype=np.int64)

    for i in range(num_envs):
        valid_bays = np.flatnonzero(bay_mask[i])
        assert valid_bays.size > 0
        bay_actions[i] = rng.choice(valid_bays)

    _, row_mask = row_obs_mask_fn(bay_actions)

    row_actions = np.empty(num_envs, dtype=np.int64)
    for i in range(num_envs):
        valid_rows = np.flatnonzero(row_mask[i])
        assert valid_rows.size > 0
        row_actions[i] = rng.choice(valid_rows)

    return bay_actions, row_actions


@pytest.mark.parametrize("num_envs", [1, 3])
def test_subproc_matches_sync_vec_env(num_envs: int) -> None:

    config = _make_config(seed=0)
    base_seed = 42

    sync_env = VecHierarchicalEnv(
        config=config,
        num_envs=num_envs,
        base_seed=base_seed,
    )
    subproc_env = SubprocVecHierarchicalEnv(
        config=config,
        num_envs=num_envs,
        base_seed=base_seed,
    )

    try:
        rng_sync = np.random.default_rng(123)
        rng_subproc = np.random.default_rng(123)

        np.testing.assert_array_equal(
            sync_env.get_global_state(), subproc_env.get_global_state()
        )
        np.testing.assert_array_equal(
            sync_env.get_bay_action_mask(), subproc_env.get_bay_action_mask()
        )

        for _ in range(40):

            bay_actions, row_actions = _pick_actions(
                sync_env.get_bay_action_mask(),
                sync_env.get_row_decision_input,
                rng_sync,
            )

            # Sanity: the same RNG stream applied to the same masks
            # must choose the same actions for both backends, since
            # masks/observations have matched at every step so far.
            subproc_bay_mask = subproc_env.get_bay_action_mask()
            np.testing.assert_array_equal(
                sync_env.get_bay_action_mask(), subproc_bay_mask
            )
            subproc_bay_actions, subproc_row_actions = _pick_actions(
                subproc_bay_mask,
                subproc_env.get_row_decision_input,
                rng_subproc,
            )
            np.testing.assert_array_equal(bay_actions, subproc_bay_actions)
            np.testing.assert_array_equal(row_actions, subproc_row_actions)

            (
                sync_next_state,
                sync_rewards,
                sync_terminated,
                sync_truncated,
                _sync_infos,
            ) = sync_env.step(bay_actions, row_actions)

            (
                subproc_next_state,
                subproc_rewards,
                subproc_terminated,
                subproc_truncated,
                _subproc_infos,
            ) = subproc_env.step(subproc_bay_actions, subproc_row_actions)

            np.testing.assert_array_equal(sync_next_state, subproc_next_state)
            np.testing.assert_allclose(sync_rewards, subproc_rewards)
            np.testing.assert_array_equal(sync_terminated, subproc_terminated)
            np.testing.assert_array_equal(sync_truncated, subproc_truncated)
            np.testing.assert_array_equal(
                sync_env.get_global_state(), subproc_env.get_global_state()
            )
            np.testing.assert_array_equal(
                sync_env.get_bay_action_mask(), subproc_env.get_bay_action_mask()
            )

    finally:
        sync_env.close()
        subproc_env.close()


def test_subproc_spaces_and_inner_env_match_reference() -> None:

    config = _make_config(seed=1)

    sync_env = VecHierarchicalEnv(config=config, num_envs=2)
    subproc_env = SubprocVecHierarchicalEnv(config=config, num_envs=2)

    try:
        assert sync_env.bay_action_space.n == subproc_env.bay_action_space.n
        assert sync_env.row_action_space.n == subproc_env.row_action_space.n
        assert (
            sync_env.bay_observation_space.shape
            == subproc_env.bay_observation_space.shape
        )
        assert sync_env.inner_env.yard_shape == subproc_env.inner_env.yard_shape
    finally:
        sync_env.close()
        subproc_env.close()
