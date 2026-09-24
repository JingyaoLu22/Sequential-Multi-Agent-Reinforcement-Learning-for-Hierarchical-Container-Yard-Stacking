"""
Row-level observation/action adapter for the hierarchical MARL pipeline.

This class does NOT create its own StackEnv and does NOT call env.step().

It receives the same shared StackEnv used by BayEnv and provides Agent R with:

    1. The original stack_features_v3 features belonging only to the
       bay selected by Agent B.
    2. A local row-level action space.
    3. A row-level validity mask derived directly from StackEnv's
       original validity logic.
    4. Mapping utilities from:
           (bay_idx, row_idx)
       to the original StackEnv global action.

Important
---------
The feature representation itself is NOT changed.

If StackEnv produces:

    [Bay 0 Row 0 features,
     Bay 0 Row 1 features,
     ...
     Bay 1 Row 0 features,
     ...]

RowEnv simply slices out the rows belonging to the selected bay.

It does NOT recompute another feature representation and it does NOT
change the original feature ordering or stack_index values.
"""

from __future__ import annotations

from typing import List

import gymnasium as gym
import numpy as np

from ..stack_gym import StackEnv


class RowEnv:
    """
    Local selected-bay adapter used by Agent R.

    Agent B chooses a zero-based bay index:

        bay_idx = 0, 1, ..., n_bays - 1

    Agent R then chooses a zero-based row index inside that bay:

        row_idx = 0, 1, ..., n_rows - 1

    StackEnv itself uses:

        physical bay numbers = 1, 3, 5, ...
        physical row numbers = 1, 2, 3, ...

    This adapter performs the required conversions without changing
    the underlying StackEnv state.
    """

    def __init__(self, stack_env: StackEnv) -> None:
        self.stack_env = stack_env

        # --------------------------------------------------------------
        # For the new experiment we intentionally preserve the thesis
        # observation representation.
        # --------------------------------------------------------------
        if self.stack_env.observation_type != "stack_features_v3":
            raise ValueError(
                "RowEnv currently requires "
                "observation_type='stack_features_v3'. "
                f"Received {self.stack_env.observation_type!r}."
            )

        # Number of physical bays and rows.
        #
        # In StackEnv:
        #     yard_shape = (num_physical_bays, num_rows, num_tiers)
        #
        self.num_bays = self.stack_env.yard_shape[0]
        self.num_rows = self.stack_env.yard_shape[1]

        # --------------------------------------------------------------
        # Physical bay numbering
        # --------------------------------------------------------------
        #
        # StackEnv internally uses odd physical bay numbers:
        #
        #     1, 3, 5, 7, ...
        #
        # This is the same convention used by:
        #
        #     StackEnv._action_to_bay_row()
        #     StackEnv._bay_row_to_action()
        #
        all_bays = self.stack_env._generate_bay_coords(
            self.stack_env.yard_shape[0]
        )

        self.baynum_list: List[int] = [
            int(bay)
            for bay in all_bays
            if bay % 2 != 0
        ]

        if len(self.baynum_list) != self.num_bays:
            raise RuntimeError(
                "Unexpected mismatch between yard_shape and "
                "generated physical bay numbering."
            )

        # --------------------------------------------------------------
        # Number of features in ONE original stack_features_v3 token.
        # --------------------------------------------------------------
        #
        # Source StackEnv definition:
        #
        # Base:
        #   [0] majority_group
        #   [1] num_occupied
        #   [2] current_container_group
        #   [3] adj_majority_group
        #   [4] stack_index
        #
        # +3 when container_sizes=True
        # +2 when enable_imo=True
        #
        self.features_per_stack = 5

        if self.stack_env.container_sizes:
            self.features_per_stack += 3

        if self.stack_env.enable_imo:
            self.features_per_stack += 2

        self.global_num_stacks = self.num_bays * self.num_rows
        self.global_observation_dim = (
            self.global_num_stacks * self.features_per_stack
        )

        self.local_observation_dim = (
            self.num_rows * self.features_per_stack
        )

        # --------------------------------------------------------------
        # Observation space for Agent R
        # --------------------------------------------------------------
        #
        # Reuse the bounds/dtype of the original StackEnv observation
        # rather than inventing new limits.
        #
        source_space = self.stack_env.observation_space

        if isinstance(source_space, gym.spaces.Dict):
            if "observation" not in source_space.spaces:
                raise ValueError(
                    "Expected StackEnv Dict observation space to contain "
                    "an 'observation' entry."
                )

            source_space = source_space.spaces["observation"]

        if not isinstance(source_space, gym.spaces.Box):
            raise TypeError(
                "RowEnv expects the original StackEnv observation "
                "to be a gym.spaces.Box."
            )

        if source_space.shape != (self.global_observation_dim,):
            raise ValueError(
                "Unexpected StackEnv observation dimension. "
                f"Expected {(self.global_observation_dim,)}, "
                f"got {source_space.shape}."
            )

        # stack_features_v3 uses the same limits for every element,
        # therefore taking one bay-sized slice preserves the exact
        # source observation-space definition.
        self.observation_space = gym.spaces.Box(
            low=source_space.low[: self.local_observation_dim].copy(),
            high=source_space.high[: self.local_observation_dim].copy(),
            dtype=source_space.dtype,
        )

        # --------------------------------------------------------------
        # Agent R action space
        # --------------------------------------------------------------
        #
        # Agent R only chooses one row/stack INSIDE the selected bay.
        #
        self.action_space = gym.spaces.Discrete(self.num_rows)

    # ==================================================================
    # Observation
    # ==================================================================

    def get_observation(self, bay_idx: int) -> np.ndarray:
        """
        Return the original stack_features_v3 features for only the bay
        selected by Agent B.

        Parameters
        ----------
        bay_idx : int
            Zero-based bay action selected by Agent B.

        Returns
        -------
        np.ndarray
            Flattened selected-bay observation with shape:

                (num_rows * features_per_stack,)

        Important
        ---------
        No feature is recomputed here.

        We first request the ORIGINAL global StackEnv observation and
        then slice out the selected bay.
        """

        self._validate_bay_index(bay_idx)

        # --------------------------------------------------------------
        # Get EXACTLY the same observation StackEnv normally produces.
        # --------------------------------------------------------------
        global_observation = self.stack_env._create_observation()

        # StackEnv can optionally include its mask in the observation.
        # The new MARL rollout stores masks separately.
        if isinstance(global_observation, dict):
            if "observation" not in global_observation:
                raise ValueError(
                    "StackEnv returned a dict observation without "
                    "an 'observation' key."
                )

            global_observation = global_observation["observation"]

        global_observation = np.asarray(
            global_observation,
            dtype=np.float32,
        )

        if global_observation.shape != (self.global_observation_dim,):
            raise ValueError(
                "Unexpected global observation shape. "
                f"Expected {(self.global_observation_dim,)}, "
                f"got {global_observation.shape}."
            )

        # --------------------------------------------------------------
        # Original stack_features_v3 ordering is:
        #
        #   bay-major -> row-major -> features
        #
        # Therefore this reshape does NOT change semantics.
        # --------------------------------------------------------------
        stack_features = global_observation.reshape(
            self.num_bays,
            self.num_rows,
            self.features_per_stack,
        )

        # Select only the bay chosen by Agent B.
        selected_bay_features = stack_features[bay_idx]

        # Agent R's Transformer later expects a flat observation,
        # just as the original Transformer policy did.
        return selected_bay_features.reshape(-1).copy()

    # ==================================================================
    # Action masking
    # ==================================================================

    def action_masks(self, bay_idx: int) -> List[bool]:
        """
        Return Agent R's local row mask for the bay selected by Agent B.

        A row is valid iff the corresponding ORIGINAL StackEnv global
        stack action is valid.

        All placement constraints therefore continue to come from:

            StackEnv._get_valid_yard_actions()

        This includes the original:
            - capacity constraints
            - container-size constraints
            - IMO constraints

        No validity rule is duplicated here.

        Parameters
        ----------
        bay_idx : int
            Zero-based bay selected by Agent B.

        Returns
        -------
        List[bool]
            Boolean mask of length num_rows.
        """

        self._validate_bay_index(bay_idx)

        if self.stack_env.current_vessel_container is None:
            return [False] * self.num_rows

        valid_global_actions = set(
            int(action)
            for action in self.stack_env._get_valid_yard_actions()
        )

        physical_bay = self.bay_index_to_number(bay_idx)

        mask: List[bool] = []

        for row_idx in range(self.num_rows):

            # Agent R:
            #
            #     row_idx = 0, 1, 2, ...
            #
            # StackEnv:
            #
            #     row = 1, 2, 3, ...
            #
            physical_row = self.row_index_to_number(row_idx)

            global_action = self.stack_env._bay_row_to_action(
                physical_bay,
                physical_row,
            )

            mask.append(global_action in valid_global_actions)

        return mask

    # ==================================================================
    # Mapping helpers
    # ==================================================================

    def bay_index_to_number(self, bay_idx: int) -> int:
        """
        Convert Agent B's zero-based bay index to StackEnv's physical
        odd bay number.

        Examples
        --------
        0 -> 1
        1 -> 3
        2 -> 5
        """

        self._validate_bay_index(bay_idx)

        return self.baynum_list[bay_idx]

    def row_index_to_number(self, row_idx: int) -> int:
        """
        Convert Agent R's zero-based row action to StackEnv's physical
        one-based row number.

        Examples
        --------
        0 -> 1
        1 -> 2
        2 -> 3
        """

        self._validate_row_index(row_idx)

        return row_idx + 1

    def to_global_action(
        self,
        bay_idx: int,
        row_idx: int,
    ) -> int:
        """
        Convert the two-agent hierarchical action:

            (bay_idx, row_idx)

        into the ORIGINAL StackEnv global stack action.

        This reuses StackEnv._bay_row_to_action() rather than
        reimplementing its indexing formula.

        Parameters
        ----------
        bay_idx : int
            Zero-based action produced by Agent B.

        row_idx : int
            Zero-based action produced by Agent R.

        Returns
        -------
        int
            Original StackEnv action index.
        """

        physical_bay = self.bay_index_to_number(bay_idx)
        physical_row = self.row_index_to_number(row_idx)

        return int(
            self.stack_env._bay_row_to_action(
                physical_bay,
                physical_row,
            )
        )

    # ==================================================================
    # Convenience helpers
    # ==================================================================

    def get_valid_row_indices(
        self,
        bay_idx: int,
    ) -> np.ndarray:
        """
        Return the zero-based valid Agent R row actions.
        """

        mask = np.asarray(
            self.action_masks(bay_idx),
            dtype=bool,
        )

        return np.flatnonzero(mask)

    def is_row_valid(
        self,
        bay_idx: int,
        row_idx: int,
    ) -> bool:
        """
        Check whether a local row action is currently valid.
        """

        if not self._is_valid_bay_index(bay_idx):
            return False

        if not self._is_valid_row_index(row_idx):
            return False

        return bool(
            self.action_masks(bay_idx)[row_idx]
        )

    # ==================================================================
    # Validation helpers
    # ==================================================================

    def _is_valid_bay_index(self, bay_idx: int) -> bool:
        return (
            isinstance(bay_idx, (int, np.integer))
            and 0 <= int(bay_idx) < self.num_bays
        )

    def _is_valid_row_index(self, row_idx: int) -> bool:
        return (
            isinstance(row_idx, (int, np.integer))
            and 0 <= int(row_idx) < self.num_rows
        )

    def _validate_bay_index(self, bay_idx: int) -> None:
        if not self._is_valid_bay_index(bay_idx):
            raise ValueError(
                f"Invalid bay_idx={bay_idx}. "
                f"Expected 0 <= bay_idx < {self.num_bays}."
            )

    def _validate_row_index(self, row_idx: int) -> None:
        if not self._is_valid_row_index(row_idx):
            raise ValueError(
                f"Invalid row_idx={row_idx}. "
                f"Expected 0 <= row_idx < {self.num_rows}."
            )