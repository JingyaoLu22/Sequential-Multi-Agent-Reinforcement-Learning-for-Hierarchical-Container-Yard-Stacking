"""Episode-level metrics for Sequential HPPO evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _floats(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _ints(values: Sequence[int]) -> list[int]:
    return [int(value) for value in values]


@dataclass
class EpisodeMetrics:
    episode_index: int
    total_reward: float
    episode_length: int
    step_rewards: list[float]
    cumulative_rewards: list[float]
    bay_actions: list[int]
    row_actions: list[int]
    selected_bays: list[int]
    selected_rows: list[int]
    global_actions: list[int]
    terminated: bool
    truncated: bool
    containers_retrieved: Optional[int]
    containers_remaining: Optional[int]
    imo_violation: bool
    all_actions_masked: bool

    @property
    def completed_successfully(self) -> bool:
        return self.containers_remaining == 0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["completed_successfully"] = self.completed_successfully
        return result


@dataclass
class EvaluationSummary:
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
    episode_rewards: list[float]
    episode_lengths: list[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    rewards = _floats(step_rewards)
    traces = {
        "bay_actions": _ints(bay_actions),
        "row_actions": _ints(row_actions),
        "selected_bays": _ints(selected_bays),
        "selected_rows": _ints(selected_rows),
        "global_actions": _ints(global_actions),
    }
    expected = len(rewards)
    for name, values in traces.items():
        if len(values) != expected:
            raise ValueError(
                f"{name} length mismatch. Expected {expected}, received {len(values)}."
            )

    cumulative = np.cumsum(rewards, dtype=np.float64).astype(float).tolist()
    return EpisodeMetrics(
        episode_index=int(episode_index),
        total_reward=float(cumulative[-1]) if cumulative else 0.0,
        episode_length=expected,
        step_rewards=rewards,
        cumulative_rewards=cumulative,
        bay_actions=traces["bay_actions"],
        row_actions=traces["row_actions"],
        selected_bays=traces["selected_bays"],
        selected_rows=traces["selected_rows"],
        global_actions=traces["global_actions"],
        terminated=bool(terminated),
        truncated=bool(truncated),
        containers_retrieved=_optional_int(final_info.get("containers_retrieved")),
        containers_remaining=_optional_int(final_info.get("containers_remaining")),
        imo_violation=bool(final_info.get("imo_violation", False)),
        all_actions_masked=bool(final_info.get("all_actions_masked", False)),
    )


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def summarize_episodes(episodes: Sequence[EpisodeMetrics]) -> EvaluationSummary:
    if not episodes:
        raise ValueError("Cannot summarize zero evaluation episodes.")

    rewards = np.asarray([episode.total_reward for episode in episodes], dtype=np.float64)
    lengths = np.asarray([episode.episode_length for episode in episodes], dtype=np.float64)
    n_episodes = len(episodes)
    successful = sum(episode.completed_successfully for episode in episodes)
    truncated = sum(episode.truncated for episode in episodes)
    violations = sum(episode.imo_violation for episode in episodes)
    masked = sum(episode.all_actions_masked for episode in episodes)

    return EvaluationSummary(
        n_episodes=n_episodes,
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
        min_reward=float(rewards.min()),
        max_reward=float(rewards.max()),
        mean_episode_length=float(lengths.mean()),
        std_episode_length=float(lengths.std()),
        successful_episodes=int(successful),
        completion_rate=float(successful / n_episodes),
        truncated_episodes=int(truncated),
        truncation_rate=float(truncated / n_episodes),
        imo_violation_episodes=int(violations),
        all_actions_masked_episodes=int(masked),
        episode_rewards=_floats(rewards),
        episode_lengths=_ints(lengths),
    )
