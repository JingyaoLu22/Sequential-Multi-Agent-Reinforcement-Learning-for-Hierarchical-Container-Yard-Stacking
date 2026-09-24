"""
Evaluation metrics for the sequential hierarchical multi-agent policy.

This module contains data structures and aggregation utilities only.

It does NOT:
    - run the environment
    - select actions
    - update models
    - calculate PPO losses
    - modify rewards

The actual evaluation loop is implemented in:

    evaluation/evaluate.py

One episode follows:

    Agent B chooses bay
        ->
    Agent R chooses row
        ->
    ONE StackEnv.step()
        ->
    reward

For visualization we keep both:

    step rewards:
        r_0, r_1, ..., r_T

and cumulative rewards:

        C_t = sum_{k=0}^{t} r_k

so evaluation can later plot the reward accumulated through an episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ======================================================================
# One evaluation episode
# ======================================================================


@dataclass
class EpisodeMetrics:
    """
    Metrics recorded for ONE deterministic evaluation episode.

    Parameters
    ----------
    episode_index
        Index of the evaluation episode.

    total_reward
        Sum of all original StackEnv rewards in this episode.

    episode_length
        Number of complete hierarchical decisions / physical
        StackEnv transitions.

    step_rewards
        Original StackEnv reward at every step.

    cumulative_rewards
        Running sum of step rewards.

    bay_actions
        Zero-based Agent B bay actions.

    row_actions
        Zero-based Agent R row actions.

    selected_bays
        Physical StackEnv bay numbers:

            1, 3, 5, ...

    selected_rows
        Physical StackEnv row numbers:

            1, 2, 3, ...

    global_actions
        Original StackEnv stack actions.

    terminated
        Final original environment terminated flag.

    truncated
        Final original environment truncated flag.

    containers_retrieved
        Final containers_retrieved value when provided by StackEnv.

    containers_remaining
        Final containers_remaining value when provided by StackEnv.

    imo_violation
        Whether an IMO violation was reported.

    all_actions_masked
        Whether StackEnv reported that no valid actions remained.
    """

    episode_index: int

    total_reward: float
    episode_length: int

    step_rewards: List[float]
    cumulative_rewards: List[float]

    bay_actions: List[int]
    row_actions: List[int]

    selected_bays: List[int]
    selected_rows: List[int]

    global_actions: List[int]

    terminated: bool
    truncated: bool

    containers_retrieved: Optional[int]
    containers_remaining: Optional[int]

    imo_violation: bool
    all_actions_masked: bool

    # ==================================================================
    # Convenience properties
    # ==================================================================

    @property
    def completed_successfully(
        self,
    ) -> bool:
        """
        True only when the environment reports that zero containers
        remain.

        We deliberately do NOT define:

            terminated == True

        as success because StackEnv can also terminate following
        invalid/unsafe actions.
        """

        return (
            self.containers_remaining
            is not None
            and
            self.containers_remaining == 0
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert episode metrics to a normal dictionary.
        """

        result = asdict(
            self
        )

        result[
            "completed_successfully"
        ] = self.completed_successfully

        return result


# ======================================================================
# Complete evaluation summary
# ======================================================================


@dataclass
class EvaluationSummary:
    """
    Aggregated statistics across multiple evaluation episodes.
    """

    n_episodes: int

    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float

    mean_episode_length: float
    std_episode_length: float

    successful_episodes: int
    completion_rate: float

    truncated_episodes: int
    truncation_rate: float

    imo_violation_episodes: int
    all_actions_masked_episodes: int

    episode_rewards: List[float]
    episode_lengths: List[int]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert summary into a dictionary suitable for printing,
        JSON output or W&B logging.
        """

        return asdict(
            self
        )


# ======================================================================
# Build one EpisodeMetrics object
# ======================================================================


def build_episode_metrics(
    *,
    episode_index: int,
    step_rewards: Sequence[float],
    bay_actions: Sequence[int],
    row_actions: Sequence[int],
    selected_bays: Sequence[int],
    selected_rows: Sequence[int],
    global_actions: Sequence[int],
    terminated: bool,
    truncated: bool,
    final_info: Dict[str, Any],
) -> EpisodeMetrics:
    """
    Construct metrics for one completed evaluation episode.

    The function does not alter any source reward or environment
    information.
    """

    # ==============================================================
    # Convert sequences to concrete Python lists
    # ==============================================================

    step_rewards = [
        float(value)
        for value in step_rewards
    ]

    bay_actions = [
        int(value)
        for value in bay_actions
    ]

    row_actions = [
        int(value)
        for value in row_actions
    ]

    selected_bays = [
        int(value)
        for value in selected_bays
    ]

    selected_rows = [
        int(value)
        for value in selected_rows
    ]

    global_actions = [
        int(value)
        for value in global_actions
    ]

    # ==============================================================
    # All action traces should describe the same number of physical
    # environment steps.
    # ==============================================================

    episode_length = len(
        step_rewards
    )

    trace_lengths = {
        "bay_actions":
            len(bay_actions),

        "row_actions":
            len(row_actions),

        "selected_bays":
            len(selected_bays),

        "selected_rows":
            len(selected_rows),

        "global_actions":
            len(global_actions),
    }

    for name, length in trace_lengths.items():

        if length != episode_length:

            raise ValueError(
                f"{name} length mismatch. "
                f"Expected {episode_length}, "
                f"received {length}."
            )

    # ==============================================================
    # Cumulative reward
    # ==============================================================
    #
    # C_t =
    #
    #     r_0
    #     + r_1
    #     + ...
    #     + r_t
    #
    # This is the curve that we will visualize in evaluate.py.
    # ==============================================================

    if episode_length > 0:

        cumulative_rewards = (
            np.cumsum(
                np.asarray(
                    step_rewards,
                    dtype=np.float64,
                )
            )
            .astype(float)
            .tolist()
        )

        total_reward = float(
            cumulative_rewards[-1]
        )

    else:

        cumulative_rewards = []

        total_reward = 0.0

    # ==============================================================
    # Diagnostics already exposed by the original StackEnv
    # ==============================================================

    containers_retrieved = (
        final_info.get(
            "containers_retrieved"
        )
    )

    containers_remaining = (
        final_info.get(
            "containers_remaining"
        )
    )

    if containers_retrieved is not None:

        containers_retrieved = int(
            containers_retrieved
        )

    if containers_remaining is not None:

        containers_remaining = int(
            containers_remaining
        )

    return EpisodeMetrics(

        episode_index=int(
            episode_index
        ),

        total_reward=(
            total_reward
        ),

        episode_length=(
            episode_length
        ),

        step_rewards=(
            step_rewards
        ),

        cumulative_rewards=(
            cumulative_rewards
        ),

        bay_actions=(
            bay_actions
        ),

        row_actions=(
            row_actions
        ),

        selected_bays=(
            selected_bays
        ),

        selected_rows=(
            selected_rows
        ),

        global_actions=(
            global_actions
        ),

        terminated=bool(
            terminated
        ),

        truncated=bool(
            truncated
        ),

        containers_retrieved=(
            containers_retrieved
        ),

        containers_remaining=(
            containers_remaining
        ),

        imo_violation=bool(
            final_info.get(
                "imo_violation",
                False,
            )
        ),

        all_actions_masked=bool(
            final_info.get(
                "all_actions_masked",
                False,
            )
        ),
    )


# ======================================================================
# Aggregate episodes
# ======================================================================


def summarize_episodes(
    episodes: Sequence[
        EpisodeMetrics
    ],
) -> EvaluationSummary:
    """
    Aggregate multiple deterministic evaluation episodes.

    Reward statistics intentionally include the same core quantities
    commonly used by the original project:

        mean reward
        std reward
        min reward
        max reward

    Additional diagnostics are computed without altering the
    environment or reward definition.
    """

    if len(episodes) == 0:

        raise ValueError(
            "Cannot summarize zero evaluation episodes."
        )

    # ==============================================================
    # Reward
    # ==============================================================

    rewards = np.asarray(
        [
            episode.total_reward
            for episode in episodes
        ],
        dtype=np.float64,
    )

    # ==============================================================
    # Episode lengths
    # ==============================================================

    lengths = np.asarray(
        [
            episode.episode_length
            for episode in episodes
        ],
        dtype=np.float64,
    )

    # ==============================================================
    # Completion
    # ==============================================================

    successful_episodes = sum(
        episode.completed_successfully
        for episode in episodes
    )

    truncated_episodes = sum(
        episode.truncated
        for episode in episodes
    )

    imo_violation_episodes = sum(
        episode.imo_violation
        for episode in episodes
    )

    all_actions_masked_episodes = sum(
        episode.all_actions_masked
        for episode in episodes
    )

    n_episodes = len(
        episodes
    )

    # ==============================================================
    # Summary
    # ==============================================================

    return EvaluationSummary(

        n_episodes=(
            n_episodes
        ),

        mean_reward=float(
            rewards.mean()
        ),

        std_reward=float(
            rewards.std()
        ),

        min_reward=float(
            rewards.min()
        ),

        max_reward=float(
            rewards.max()
        ),

        mean_episode_length=float(
            lengths.mean()
        ),

        std_episode_length=float(
            lengths.std()
        ),

        successful_episodes=int(
            successful_episodes
        ),

        completion_rate=float(
            successful_episodes
            / n_episodes
        ),

        truncated_episodes=int(
            truncated_episodes
        ),

        truncation_rate=float(
            truncated_episodes
            / n_episodes
        ),

        imo_violation_episodes=int(
            imo_violation_episodes
        ),

        all_actions_masked_episodes=int(
            all_actions_masked_episodes
        ),

        episode_rewards=[
            float(value)
            for value in rewards
        ],

        episode_lengths=[
            int(value)
            for value in lengths
        ],
    )