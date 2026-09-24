"""
Hierarchical environment coordinator for the multi-agent PPO pipeline.

This class coordinates two decision levels:

    Agent B:
        observes the original global StackEnv observation
        and selects a bay.

    Agent R:
        observes only the rows/stacks inside the bay selected
        by Agent B and selects a row.

The two decisions are then converted into the ORIGINAL StackEnv
global stack action, and StackEnv.step(...) is called exactly once.

Important
---------
There is only ONE physical StackEnv instance.

BayEnv and RowEnv do NOT own separate environments. They are only
observation/action adapters over the same StackEnv.

Decision sequence:

    state s_t
        |
        v
    Agent B observation
        |
        v
    bay_idx
        |
        |   no environment transition here
        v
    Agent R local observation(selected bay)
        |
        v
    row_idx
        |
        v
    original StackEnv global action
        |
        v
    StackEnv.step(global_action)
        |
        v
    reward, s_{t+1}

No reward, state representation, action-validity rule, or physical
environment transition is reimplemented in this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..stack_gym import StackEnv
from .bay_env import BayEnv
from .row_env import RowEnv


@dataclass(frozen=True)
class StateSnapshot:
    """
    Cached view of one physical StackEnv state, built once per state
    instead of being rebuilt independently by every hierarchical
    getter.

    global_state:
        (global_obs_dim,) float32 - identical to
        StackEnv._create_observation()'s output for this state.

    global_mask_2d:
        (num_bays, num_rows) bool - dense form of
        StackEnv._get_valid_yard_actions()'s sparse action-index output
        for this state, scattered via the same bay-major/row-minor
        indexing StackEnv._bay_row_to_action() uses:
        action = bay_idx * num_rows + row_idx.

    bay_mask:
        (num_bays,) bool = global_mask_2d.any(axis=1) - a bay is valid
        iff at least one row inside it is valid.
    """

    global_state: np.ndarray
    global_mask_2d: np.ndarray
    bay_mask: np.ndarray


class HierarchicalEnv:
    """
    Coordinator for the hierarchical multi-agent environment.

    The class owns exactly ONE StackEnv instance and gives references
    to that same environment to BayEnv and RowEnv.

    Parameters
    ----------
    config : dict | None
        Original StackEnv configuration.

    render_mode : str | None
        Passed directly to StackEnv.
    """

    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        render_mode: Optional[str] = None,
    ) -> None:

        # ==============================================================
        # ONE shared physical environment
        # ==============================================================

        self.inner_env = StackEnv(
            config=config,
            render_mode=render_mode,
        )

        # ==============================================================
        # Two adapters over the SAME StackEnv
        # ==============================================================

        self.bay_env = BayEnv(self.inner_env)
        self.row_env = RowEnv(self.inner_env)

        # Defensive sanity check:
        #
        # BayEnv and RowEnv MUST see exactly the same physical
        # environment object.
        if (
            self.bay_env.stack_env is not self.inner_env
            or self.row_env.stack_env is not self.inner_env
        ):
            raise RuntimeError(
                "BayEnv and RowEnv must share the same StackEnv instance."
            )

        # ==============================================================
        # Spaces exposed for the two agents
        # ==============================================================

        # Agent B
        self.bay_observation_space = (
            self.bay_env.observation_space
        )
        self.bay_action_space = (
            self.bay_env.action_space
        )

        # Agent R
        self.row_observation_space = (
            self.row_env.observation_space
        )
        self.row_action_space = (
            self.row_env.action_space
        )

        # The global state used later by the centralized critic is,
        # for now, exactly the original global StackEnv observation.
        #
        # Therefore it has the same space as Agent B's observation.
        self.global_observation_space = (
            self.bay_observation_space
        )

        self.render_mode = render_mode

        # ==============================================================
        # Cached StateSnapshot for the current physical state
        # ==============================================================
        #
        # Built once by reset()/step() instead of being independently
        # rebuilt by every getter below - see StateSnapshot's docstring.
        # None until the first reset().
        self._snapshot: Optional[StateSnapshot] = None

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the ONE shared StackEnv.

        Returns
        -------
        bay_observation : np.ndarray
            Initial global observation for Agent B.

        info : dict
            Original StackEnv reset info.

        Notes
        -----
        At reset time Agent R does not yet receive an observation,
        because Agent B has not selected a bay yet.
        """

        observation, info = self.inner_env.reset(
            seed=seed,
            **kwargs,
        )

        observation = self._unwrap_observation(
            observation
        )

        # StackEnv.reset() does not hand back a valid-action list the
        # way step() does (see step() below), so this is the one real
        # _get_valid_yard_actions() call per episode needed to seed the
        # snapshot. Everything else derives from it without recomputing.
        valid_actions = (
            self.inner_env._get_valid_yard_actions()
        )

        self._snapshot = self._build_snapshot(
            observation,
            valid_actions,
        )

        return observation, info

    # ==================================================================
    # Global / Agent B observation
    # ==================================================================

    def get_bay_observation(self) -> np.ndarray:
        """
        Return Agent B's current global observation.

        Read from the cached StateSnapshot instead of asking BayEnv to
        rebuild it - identical to BayEnv.get_observation()'s output for
        the current state, see StateSnapshot's docstring.
        """

        return self._current_snapshot().global_state

    def get_global_state(self) -> np.ndarray:
        """
        Return the global state for the centralized critic.

        For this project the global state is kept identical to the
        original full StackEnv observation. Read from the cached
        StateSnapshot - see get_bay_observation()/StateSnapshot.

        Later the critic will consume this state, but this environment
        class does not perform any critic computation.
        """

        return self._current_snapshot().global_state

    # ==================================================================
    # Agent B action mask
    # ==================================================================

    def get_bay_action_mask(self) -> np.ndarray:
        """
        Return the valid bay mask for Agent B.

        A bay is valid iff at least one valid original StackEnv action
        exists inside that bay. Read from the cached StateSnapshot's
        bay_mask (= global_mask_2d.any(axis=1)) instead of BayEnv
        recomputing _get_valid_yard_actions() from scratch.
        """

        return self._current_snapshot().bay_mask

    # ==================================================================
    # Agent R observation
    # ==================================================================

    def get_row_observation(
        self,
        bay_idx: int,
    ) -> np.ndarray:
        """
        Return Agent R's local observation for the bay chosen by Agent B.

        Sliced/reshaped from the cached StateSnapshot's global_state -
        the same bay-major/row-major reshape RowEnv.get_observation()
        performs, but without recomputing the underlying observation.

        Parameters
        ----------
        bay_idx : int
            Zero-based bay action produced by Agent B.

        Returns
        -------
        np.ndarray
            Original stack_features_v3 vectors belonging only to the
            selected bay.
        """

        self.row_env._validate_bay_index(bay_idx)

        snapshot = self._current_snapshot()

        stack_features = snapshot.global_state.reshape(
            self.row_env.num_bays,
            self.row_env.num_rows,
            self.row_env.features_per_stack,
        )

        return stack_features[bay_idx].reshape(-1).copy()

    # ==================================================================
    # Agent R action mask
    # ==================================================================

    def get_row_action_mask(
        self,
        bay_idx: int,
    ) -> np.ndarray:
        """
        Return Agent R's row mask inside the selected bay.

        All validity rules come from the original:

            StackEnv._get_valid_yard_actions()

        No placement constraint is recreated here. Sliced from the
        cached StateSnapshot's global_mask_2d instead of RowEnv
        recomputing _get_valid_yard_actions() from scratch.
        """

        self.row_env._validate_bay_index(bay_idx)

        return self._current_snapshot().global_mask_2d[bay_idx]

    def get_row_decision_input(
        self,
        bay_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method used during rollout.

        After Agent B selects a bay, return both:

            row_observation
            row_action_mask

        for Agent R.

        No physical environment transition occurs here.
        """

        row_observation = (
            self.get_row_observation(bay_idx)
        )

        row_action_mask = (
            self.get_row_action_mask(bay_idx)
        )

        return (
            row_observation,
            row_action_mask,
        )

    # ==================================================================
    # Hierarchical action -> original StackEnv action
    # ==================================================================

    def to_global_action(
        self,
        bay_idx: int,
        row_idx: int,
    ) -> int:
        """
        Convert the two-agent decision:

            (bay_idx, row_idx)

        into the ORIGINAL StackEnv stack action.

        Example
        -------
        If:

            n_rows = 4
            bay_idx = 2
            row_idx = 3

        then:

            physical bay = 5
            physical row = 4

        and StackEnv._bay_row_to_action(5, 4) is used.

        The original action mapping logic is therefore preserved.
        """

        return self.row_env.to_global_action(
            bay_idx,
            row_idx,
        )

    # ==================================================================
    # Environment transition
    # ==================================================================

    def step(
        self,
        bay_idx: int,
        row_idx: int,
    ) -> Tuple[
        np.ndarray,
        float,
        bool,
        bool,
        Dict,
    ]:
        """
        Execute ONE physical environment transition.

        This method is called only after BOTH agents have made their
        decisions:

            Agent B -> bay_idx
            Agent R -> row_idx

        The hierarchical action is converted into the original
        StackEnv action and passed directly to:

            StackEnv.step(global_action)

        Therefore:
            - reward stays unchanged
            - yard transition stays unchanged
            - termination stays unchanged
            - invalid-action behaviour stays unchanged

        Parameters
        ----------
        bay_idx : int
            Zero-based bay selected by Agent B.

        row_idx : int
            Zero-based row selected by Agent R.

        Returns
        -------
        next_global_state : np.ndarray
            Original StackEnv observation after placement.

        reward : float
            Original StackEnv reward.

        terminated : bool
            Original StackEnv termination flag.

        truncated : bool
            Original StackEnv truncation flag.

        info : dict
            Original StackEnv info plus hierarchical action metadata.
        """

        # --------------------------------------------------------------
        # Convert hierarchical decisions to original StackEnv action
        # --------------------------------------------------------------

        global_action = self.to_global_action(
            bay_idx,
            row_idx,
        )

        # These are only recorded for debugging / visualization.
        #
        # They do NOT affect the actual environment transition.
        physical_bay = (
            self.row_env.bay_index_to_number(
                bay_idx
            )
        )

        physical_row = (
            self.row_env.row_index_to_number(
                row_idx
            )
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # This is the ONLY physical environment transition.
        # --------------------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.inner_env.step(
            global_action
        )

        # --------------------------------------------------------------
        # Keep the returned state representation identical to the
        # original StackEnv observation.
        # --------------------------------------------------------------

        next_global_state = (
            self._unwrap_observation(
                observation
            )
        )

        # --------------------------------------------------------------
        # Refresh the StateSnapshot for s_{t+1} WITHOUT recomputing.
        #
        # StackEnv.step() already computes the valid-action set for the
        # new state as a mandatory side effect of determining
        # `truncated` (see StackEnv.step()'s own docstring/body), and
        # already returns it as info["yard_mask"] on every return path
        # (including the early-return invalid-action/termination
        # branches, where it describes the unchanged current state
        # instead - harmless here, since VecHierarchicalEnv always
        # calls reset() on a terminated/truncated env before this
        # snapshot would be read again, and reset() rebuilds it for
        # real). No extra StackEnv call is made.
        # --------------------------------------------------------------

        self._snapshot = self._build_snapshot(
            next_global_state,
            info.get(
                "yard_mask",
                [],
            ),
        )

        # --------------------------------------------------------------
        # Preserve original info and only ADD diagnostics.
        # --------------------------------------------------------------

        info = dict(info)

        info.update(
            {
                # zero-based MARL actions
                "bay_idx": int(bay_idx),
                "row_idx": int(row_idx),

                # original physical numbering
                "selected_bay": int(
                    physical_bay
                ),
                "selected_row": int(
                    physical_row
                ),

                # original StackEnv action
                "global_action": int(
                    global_action
                ),
            }
        )

        return (
            next_global_state,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # StateSnapshot helpers
    # ==================================================================

    def _current_snapshot(self) -> StateSnapshot:
        """
        Return the cached StateSnapshot for the current physical state.

        Raises if called before the first reset() - matches the
        existing convention that every getter below requires a reset()
        to have already happened.
        """

        if self._snapshot is None:
            raise RuntimeError(
                "HierarchicalEnv.reset() must be called before "
                "reading any observation/mask."
            )

        return self._snapshot

    def _build_snapshot(
        self,
        global_state: np.ndarray,
        valid_actions,
    ) -> StateSnapshot:
        """
        Build a StateSnapshot from a global observation and the sparse
        valid-action-index output of StackEnv._get_valid_yard_actions()
        (or, on step(), the equivalent info["yard_mask"] StackEnv
        already produced as a side effect of computing `truncated`).

        Densifying + reshaping this way is exactly the same bay-major/
        row-minor indexing BayEnv.action_masks()/RowEnv.action_masks()
        compute by hand (StackEnv._bay_row_to_action(bay, row) ==
        bay_idx * num_rows + row_idx), so this produces the identical
        mask values, not an approximation.
        """

        num_bays = self.row_env.num_bays
        num_rows = self.row_env.num_rows

        flat_mask = np.zeros(
            num_bays * num_rows,
            dtype=bool,
        )

        valid_actions = np.asarray(
            valid_actions,
            dtype=int,
        )

        if valid_actions.size > 0:
            flat_mask[valid_actions] = True

        global_mask_2d = flat_mask.reshape(
            num_bays,
            num_rows,
        )

        return StateSnapshot(
            global_state=np.asarray(
                global_state,
                dtype=np.float32,
            ),
            global_mask_2d=global_mask_2d,
            bay_mask=global_mask_2d.any(axis=1),
        )

    # ==================================================================
    # Observation helper
    # ==================================================================

    @staticmethod
    def _unwrap_observation(
        observation,
    ) -> np.ndarray:
        """
        Extract the original observation array.

        StackEnv optionally returns:

            {
                "observation": ...,
                "mask": ...
            }

        when action masks are embedded in observations.

        The new multi-agent rollout stores action masks separately, so
        only the original observation vector is returned here.

        No feature transformation is performed.
        """

        if isinstance(observation, dict):

            if "observation" not in observation:
                raise ValueError(
                    "HierarchicalEnv expected a StackEnv "
                    "observation dictionary containing the "
                    "'observation' key."
                )

            observation = observation[
                "observation"
            ]

        return np.asarray(
            observation,
            dtype=np.float32,
        )

    # ==================================================================
    # Rendering / closing
    # ==================================================================

    def render(self):
        """
        Delegate rendering to the original StackEnv.
        """

        return self.inner_env.render()

    def close(self) -> None:
        """
        Close the original StackEnv.
        """

        self.inner_env.close()