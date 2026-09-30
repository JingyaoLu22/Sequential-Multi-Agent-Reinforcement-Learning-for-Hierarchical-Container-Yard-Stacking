"""Episode-level metrics for Sequential HPPO evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class EpisodeMetrics:
    episode_index: int
    total_reward: float
    episode_length: int
    step_rewards: List[float]
    bay_actions: List[int]
    row_actions: List[int]
    global_actions: List[int]
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
        return {**asdict(self), "completed_successfully": self.completed_successfully}


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
    episode_rewards: List[float]
    episode_lengths: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_episode_metrics(
    *,
    episode_index: int,
    step_rewards: List[float],
    bay_actions: List[int],
    row_actions: List[int],
    global_actions: List[int],
    terminated: bool,
    truncated: bool,
    final_info: Dict[str, Any],
) -> EpisodeMetrics:
    """One finished episode; final_info is StackEnv's info of its last step."""

    def optional_int(key: str) -> Optional[int]:
        return None if final_info.get(key) is None else int(final_info[key])

    return EpisodeMetrics(
        episode_index=episode_index,
        total_reward=float(sum(step_rewards)),
        episode_length=len(step_rewards),
        step_rewards=step_rewards,
        bay_actions=bay_actions,
        row_actions=row_actions,
        global_actions=global_actions,
        terminated=terminated,
        truncated=truncated,
        containers_retrieved=optional_int("containers_retrieved"),
        containers_remaining=optional_int("containers_remaining"),
        imo_violation=bool(final_info.get("imo_violation", False)),
        all_actions_masked=bool(final_info.get("all_actions_masked", False)),
    )


def summarize_episodes(episodes: Sequence[EpisodeMetrics]) -> EvaluationSummary:
    rewards = np.array([episode.total_reward for episode in episodes])
    lengths = np.array([episode.episode_length for episode in episodes])
    n = len(episodes)
    successful = sum(episode.completed_successfully for episode in episodes)
    truncated = sum(episode.truncated for episode in episodes)

    return EvaluationSummary(
        n_episodes=n,
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
        min_reward=float(rewards.min()),
        max_reward=float(rewards.max()),
        mean_episode_length=float(lengths.mean()),
        std_episode_length=float(lengths.std()),
        successful_episodes=successful,
        completion_rate=successful / n,
        truncated_episodes=truncated,
        truncation_rate=truncated / n,
        imo_violation_episodes=sum(episode.imo_violation for episode in episodes),
        all_actions_masked_episodes=sum(episode.all_actions_masked for episode in episodes),
        episode_rewards=rewards.tolist(),
        episode_lengths=lengths.tolist(),
    )
