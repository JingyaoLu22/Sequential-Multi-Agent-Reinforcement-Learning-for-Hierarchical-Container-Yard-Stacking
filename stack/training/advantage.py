"""
Generalized Advantage Estimation (GAE) utilities for the
sequential multi-agent PPO pipeline.

This module computes the BASE team advantage:

    delta_t
        =
    r_t
    + gamma * V(s_{t+1})
    - V(s_t)

and:

    A_t
        =
    delta_t
    + gamma * lambda * A_{t+1}

The resulting base advantage is shared by the hierarchical
decision made at the same environment timestep.

Later, sequential_trainer.py will use:

    Agent B:
        A_B = A_t

    Agent R:
        A_R = detached_sequence_factor * A_t

Important
---------
This module does NOT:

    - update BayPolicy
    - update RowPolicy
    - update CentralizedCritic
    - calculate PPO ratios
    - calculate the HAPPO sequence ratio
    - perform gradient descent

Those responsibilities belong to sequential_trainer.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .rollout_buffer import JointRolloutBuffer


# ======================================================================
# Core GAE computation
# ======================================================================


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : np.ndarray
        Team rewards:

            r_t

        shape:
            (T,)

    values : np.ndarray
        Old centralized critic predictions:

            V_old(s_t)

        shape:
            (T,)

    next_values : np.ndarray
        Old centralized critic predictions of next states:

            V_old(s_{t+1})

        shape:
            (T,)

    terminated : np.ndarray
        Original environment termination flags.

        shape:
            (T,)

    truncated : np.ndarray
        Original environment truncation flags.

        For the current StackEnv, truncation means that the episode
        cannot continue because no valid placement actions remain.

        shape:
            (T,)

    gamma : float
        Discount factor.

        Original project PPO default:
            0.99

    gae_lambda : float
        GAE lambda.

        Original project PPO default:
            0.95

    Returns
    -------
    advantages : np.ndarray
        Base GAE advantages:

            A_t

        shape:
            (T,)

    returns : np.ndarray
        Critic regression targets:

            R_t = A_t + V_old(s_t)

        shape:
            (T,)
    """

    # ==============================================================
    # Convert input arrays
    # ==============================================================

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    next_values = np.asarray(
        next_values,
        dtype=np.float32,
    )

    terminated = np.asarray(
        terminated,
        dtype=np.bool_,
    )

    truncated = np.asarray(
        truncated,
        dtype=np.bool_,
    )

    # ==============================================================
    # Validation
    # ==============================================================

    arrays = {
        "rewards": rewards,
        "values": values,
        "next_values": next_values,
        "terminated": terminated,
        "truncated": truncated,
    }

    for name, array in arrays.items():

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be a 1D array. "
                f"Received shape={array.shape}."
            )

    rollout_length = len(
        rewards
    )

    for name, array in arrays.items():

        if len(array) != rollout_length:
            raise ValueError(
                f"{name} length mismatch. "
                f"Expected {rollout_length}, "
                f"received {len(array)}."
            )

    if rollout_length == 0:
        raise ValueError(
            "Cannot compute GAE for an empty rollout."
        )

    if not (
        0.0 <= gamma <= 1.0
    ):
        raise ValueError(
            f"gamma must be in [0, 1], "
            f"received {gamma}."
        )

    if not (
        0.0 <= gae_lambda <= 1.0
    ):
        raise ValueError(
            f"gae_lambda must be in [0, 1], "
            f"received {gae_lambda}."
        )

    # ==============================================================
    # Episode boundaries
    # ==============================================================
    #
    # Current StackEnv semantics:
    #
    # terminated:
    #     true terminal state
    #
    # truncated:
    #     environment cannot continue because no valid actions remain
    #
    # Therefore both stop:
    #
    #     1. critic bootstrap
    #     2. recursive GAE propagation
    #
    # This is important when the rollout buffer contains transitions
    # from multiple episodes.
    # ==============================================================

    episode_ends = np.logical_or(
        terminated,
        truncated,
    )

    # ==============================================================
    # Allocate output
    # ==============================================================

    advantages = np.zeros(
        rollout_length,
        dtype=np.float32,
    )

    last_gae = 0.0

    # ==============================================================
    # Reverse-time GAE recursion
    # ==============================================================

    for step in reversed(
        range(rollout_length)
    ):

        # ----------------------------------------------------------
        # If this transition ends the episode:
        #
        #     non_terminal = 0
        #
        # therefore:
        #
        #     delta_t = r_t - V(s_t)
        #
        # and advantage propagation stops here.
        # ----------------------------------------------------------

        non_terminal = (
            0.0
            if episode_ends[step]
            else 1.0
        )

        # ----------------------------------------------------------
        # TD residual:
        #
        # delta_t
        # =
        # r_t
        # +
        # gamma * V(s_{t+1}) * non_terminal
        # -
        # V(s_t)
        # ----------------------------------------------------------

        delta = (
            rewards[step]
            + gamma
            * next_values[step]
            * non_terminal
            - values[step]
        )

        # ----------------------------------------------------------
        # GAE:
        #
        # A_t
        # =
        # delta_t
        # +
        # gamma
        # * lambda
        # * non_terminal
        # * A_{t+1}
        # ----------------------------------------------------------

        last_gae = (
            delta
            + gamma
            * gae_lambda
            * non_terminal
            * last_gae
        )

        advantages[step] = (
            last_gae
        )

    # ==============================================================
    # Critic targets
    # ==============================================================
    #
    # Standard PPO relation:
    #
    #     return_t
    #     =
    #     advantage_t
    #     +
    #     old_value_t
    #
    # ==============================================================

    returns = (
        advantages
        + values
    )

    return (
        advantages.astype(
            np.float32
        ),
        returns.astype(
            np.float32
        ),
    )


# ======================================================================
# Buffer integration
# ======================================================================


def compute_and_store_gae(
    buffer: JointRolloutBuffer,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    num_envs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE directly from JointRolloutBuffer and store the results.

    This is the function sequential_trainer.py will normally call after
    finishing rollout collection.

    Pipeline
    --------

        rollout collection
                |
                v
        JointRolloutBuffer
                |
                v
        compute_and_store_gae()
                |
                +----> advantages
                |
                +----> returns
                |
                v
        PPO updates

    Parameters
    ----------
    num_envs : int
        Number of parallel environments used during rollout collection.

        When collect_rollout() vectorizes over num_envs environments, the
        buffer's flat (T*num_envs,) arrays interleave transitions in
        env-fastest order: env0_t0, env1_t0, ..., envN-1_t0, env0_t1, ....
        compute_gae() assumes ONE contiguous chronological trajectory, so
        running it directly over that flat interleaved array would
        propagate the GAE recursion across different environments at
        adjacent buffer positions - an env-boundary contamination bug.

        When num_envs > 1, this function instead recovers each
        environment's own chronological (T,) sequence via strided slicing
        (arr[i::num_envs]), runs the unmodified compute_gae() once per
        environment, and interleaves the results back into flat output
        arrays at the same strided positions.

    Returns
    -------
    advantages : np.ndarray

    returns : np.ndarray
    """

    if len(buffer) == 0:
        raise RuntimeError(
            "Cannot compute GAE from an empty rollout buffer."
        )

    num_envs = int(num_envs)

    if num_envs <= 0:
        raise ValueError(
            "num_envs must be positive."
        )

    rollout = (
        buffer.get_rollout_arrays()
    )

    if num_envs == 1:

        advantages, returns = compute_gae(

            rewards=rollout[
                "rewards"
            ],

            values=rollout[
                "values"
            ],

            next_values=rollout[
                "next_values"
            ],

            terminated=rollout[
                "terminated"
            ],

            truncated=rollout[
                "truncated"
            ],

            gamma=gamma,

            gae_lambda=gae_lambda,
        )

    else:

        rollout_length = len(
            rollout["rewards"]
        )

        if rollout_length % num_envs != 0:
            raise ValueError(
                "Rollout buffer length must be a multiple of "
                f"num_envs. Received length={rollout_length}, "
                f"num_envs={num_envs}."
            )

        advantages = np.zeros(
            rollout_length,
            dtype=np.float32,
        )

        returns = np.zeros(
            rollout_length,
            dtype=np.float32,
        )

        for env_idx in range(num_envs):

            env_slice = slice(
                env_idx,
                None,
                num_envs,
            )

            env_advantages, env_returns = compute_gae(

                rewards=rollout[
                    "rewards"
                ][env_slice],

                values=rollout[
                    "values"
                ][env_slice],

                next_values=rollout[
                    "next_values"
                ][env_slice],

                terminated=rollout[
                    "terminated"
                ][env_slice],

                truncated=rollout[
                    "truncated"
                ][env_slice],

                gamma=gamma,

                gae_lambda=gae_lambda,
            )

            advantages[env_slice] = env_advantages
            returns[env_slice] = env_returns

    # --------------------------------------------------------------
    # Store BASE advantage + critic targets.
    #
    # No Bay/Row sequential correction is performed here.
    # --------------------------------------------------------------

    buffer.set_advantages_and_returns(
        advantages=advantages,
        returns=returns,
    )

    return (
        advantages,
        returns,
    )


# ======================================================================
# Optional PPO advantage normalization
# ======================================================================


def normalize_advantages(
    advantages: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize an advantage vector.

    PPO commonly uses:

        A_normalized
        =
        (A - mean(A))
        /
        (std(A) + eps)

    Important
    ---------
    This function is intentionally separate from compute_gae().

    GAE itself should first produce the raw mathematical advantage.

    Whether PPO advantage normalization is enabled is a TRAINING
    configuration choice and will be controlled later by
    sequential_trainer.py.
    """

    advantages = np.asarray(
        advantages,
        dtype=np.float32,
    )

    if advantages.ndim != 1:
        raise ValueError(
            "advantages must be 1D. "
            f"Received shape="
            f"{advantages.shape}."
        )

    if len(advantages) == 0:
        raise ValueError(
            "Cannot normalize an empty advantage array."
        )

    mean = float(
        advantages.mean()
    )

    std = float(
        advantages.std()
    )

    # --------------------------------------------------------------
    # Avoid unnecessary numerical changes when variance is nearly 0.
    # --------------------------------------------------------------

    if std < eps:
        return (
            advantages - mean
        ).astype(
            np.float32
        )

    normalized = (
        advantages - mean
    ) / (
        std + eps
    )

    return normalized.astype(
        np.float32
    )