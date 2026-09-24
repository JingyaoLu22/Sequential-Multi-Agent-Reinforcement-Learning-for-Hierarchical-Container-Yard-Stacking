"""
Synchronous vector wrapper around HierarchicalEnv.

Runs N independent HierarchicalEnv instances in a single process and
steps them together, so that Agent B / Agent R / the centralized critic
can run one batched forward pass per hierarchical decision instead of
one forward pass per environment step.

This does NOT use multiprocessing. HierarchicalEnv/StackEnv construction
and stepping is cheap, pure-numpy work, so a Python-level for-loop over
N copies in one process is sufficient - true multiprocessing would only
add IPC/pickling overhead here.

Auto-reset
----------
Whenever an inner env terminates or truncates, that env is reset
immediately inside step() (SB3 VecEnv convention) and the RESET
observation is returned in its slot of next_global_state. This is safe
for GAE bootstrapping because SequentialPPOTrainer already zeroes the
bootstrap value whenever terminated or truncated is True, regardless of
what next_global_state contains.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .hierarchical_env import HierarchicalEnv


class VecHierarchicalEnv:
    """
    Batched collection of independent HierarchicalEnv instances.

    Parameters
    ----------
    config : dict | None
        StackEnv configuration, identical for every inner environment.

    num_envs : int
        Number of parallel environment copies.

    base_seed : int | None
        If given, each inner environment is reset once at construction
        time with ``seed=base_seed + i`` so their diverging per-instance
        seed counters (see StackEnv.reset) start decorrelated instead of
        identical. If None, environments are left to their StackEnv
        default seed and diverge only via their own auto-increment.

    render_mode : str | None
        Passed directly to every HierarchicalEnv.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        num_envs: int = 1,
        base_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:

        if num_envs <= 0:
            raise ValueError(
                "num_envs must be positive."
            )

        self.num_envs = int(num_envs)

        self.envs: List[HierarchicalEnv] = [
            HierarchicalEnv(
                config=config,
                render_mode=render_mode,
            )
            for _ in range(self.num_envs)
        ]

        if base_seed is not None:
            for i, env in enumerate(self.envs):
                env.reset(seed=base_seed + i)

        # ==============================================================
        # Spaces are identical across copies (identical config).
        # ==============================================================

        self.bay_observation_space = (
            self.envs[0].bay_observation_space
        )
        self.bay_action_space = (
            self.envs[0].bay_action_space
        )

        self.row_observation_space = (
            self.envs[0].row_observation_space
        )
        self.row_action_space = (
            self.envs[0].row_action_space
        )

        self.global_observation_space = (
            self.envs[0].global_observation_space
        )

        # Exposed for diagnostics/logging code that inspects the
        # underlying StackEnv (e.g. yard_shape) - identical across every
        # copy since they share the same config.
        self.inner_env = self.envs[0].inner_env

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self,
        seeds: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Reset every inner environment.

        Parameters
        ----------
        seeds : list[int] | None
            When given, environment i is reset with seed=seeds[i],
            length must equal num_envs. When None (the default, and
            the only behavior before this parameter existed), every
            environment is reset unseeded, exactly as before - existing
            callers that never pass seeds are unaffected.

            Needed for batched evaluation (evaluate_policy_batched()),
            which reuses this same VecHierarchicalEnv across multiple
            rounds of n_episodes and must reseed each round to a
            specific base_seed+i, not just once at construction time
            (see __init__'s base_seed parameter, which only seeds the
            very first reset).

        Returns
        -------
        np.ndarray
            Stacked initial bay/global observations, shape
            (num_envs, obs_dim).
        """

        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(
                "seeds must contain exactly one seed per environment. "
                f"Expected {self.num_envs}, received {len(seeds)}."
            )

        if seeds is None:
            observations = [
                env.reset()[0]
                for env in self.envs
            ]
        else:
            observations = [
                env.reset(seed=seed)[0]
                for env, seed in zip(self.envs, seeds)
            ]

        return np.stack(observations).astype(np.float32)

    # ==================================================================
    # Global / Agent B observation
    # ==================================================================

    def get_global_state(self) -> np.ndarray:
        """
        Stacked centralized-critic global state, shape
        (num_envs, obs_dim).
        """

        return np.stack(
            [
                env.get_global_state()
                for env in self.envs
            ]
        ).astype(np.float32)

    # ==================================================================
    # Agent B action mask
    # ==================================================================

    def get_bay_action_mask(self) -> np.ndarray:
        """
        Stacked bay validity mask, shape (num_envs, n_bays).
        """

        return np.stack(
            [
                env.get_bay_action_mask()
                for env in self.envs
            ]
        ).astype(bool)

    # ==================================================================
    # Agent R observation + mask
    # ==================================================================

    def get_row_decision_input(
        self,
        bay_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-env row observation/mask for the bay each env's Agent B chose.

        Parameters
        ----------
        bay_actions : np.ndarray
            Shape (num_envs,), zero-based bay action per environment.

        Returns
        -------
        row_observations : np.ndarray
            Shape (num_envs, row_obs_dim).

        row_action_masks : np.ndarray
            Shape (num_envs, n_rows).
        """

        if len(bay_actions) != self.num_envs:
            raise ValueError(
                "bay_actions must contain exactly one action per "
                f"environment. Expected {self.num_envs}, "
                f"received {len(bay_actions)}."
            )

        row_observations = []
        row_action_masks = []

        for env, bay_action in zip(self.envs, bay_actions):

            row_observation, row_action_mask = (
                env.get_row_decision_input(
                    int(bay_action)
                )
            )

            row_observations.append(row_observation)
            row_action_masks.append(row_action_mask)

        return (
            np.stack(row_observations).astype(np.float32),
            np.stack(row_action_masks).astype(bool),
        )

    # ==================================================================
    # Environment transition
    # ==================================================================

    def step(
        self,
        bay_actions: np.ndarray,
        row_actions: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Dict],
    ]:
        """
        Step every inner environment and auto-reset any that finished.

        Parameters
        ----------
        bay_actions : np.ndarray
            Shape (num_envs,).

        row_actions : np.ndarray
            Shape (num_envs,).

        Returns
        -------
        next_global_state : np.ndarray
            Shape (num_envs, obs_dim). For any env that terminated or
            truncated this step, this is already the RESET observation
            of the next episode, not the terminal observation.

        rewards : np.ndarray
            Shape (num_envs,).

        terminated : np.ndarray (bool)
            Shape (num_envs,).

        truncated : np.ndarray (bool)
            Shape (num_envs,).

        infos : list[dict]
            One info dict per environment.
        """

        if len(bay_actions) != self.num_envs or len(row_actions) != self.num_envs:
            raise ValueError(
                "bay_actions/row_actions must contain exactly one "
                f"action per environment ({self.num_envs})."
            )

        next_states = []
        # float64, not float32: env.step() returns a native Python
        # float (full precision); every existing training call site
        # (sequential_trainer.py's collect_rollout()) immediately
        # re-casts this array to float64 anyway before use, and the
        # rollout buffer stores rewards as its own float32 array
        # regardless - so this only avoids truncating precision here,
        # at the one point (batched evaluation) that actually reports
        # this value directly instead of re-deriving it. Training
        # behavior is unaffected either way.
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict] = []

        for i, (env, bay_action, row_action) in enumerate(
            zip(self.envs, bay_actions, row_actions)
        ):

            (
                next_state,
                reward,
                env_terminated,
                env_truncated,
                info,
            ) = env.step(
                int(bay_action),
                int(row_action),
            )

            rewards[i] = reward
            terminated[i] = env_terminated
            truncated[i] = env_truncated
            infos.append(info)

            if env_terminated or env_truncated:
                next_state, _reset_info = env.reset()

            next_states.append(next_state)

        next_global_state = np.stack(next_states).astype(np.float32)

        return (
            next_global_state,
            rewards,
            terminated,
            truncated,
            infos,
        )

    # ==================================================================
    # Rendering / closing
    # ==================================================================

    def close(self) -> None:
        """
        Close every inner environment.
        """

        for env in self.envs:
            env.close()
