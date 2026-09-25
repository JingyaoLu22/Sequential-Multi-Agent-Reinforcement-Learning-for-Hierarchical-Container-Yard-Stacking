"""
The environment sizes and their StackEnv configurations.

This is the single definition that every entry point imports: the
baselines (run.py, plots.py) and Sequential HPPO training and
evaluation. It only uses the standard library, so both the
stack package (``stack.configs.environments``) and the script-style
baseline modules under stack/ (``configs.environments``) import it
directly.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

ENVIRONMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small": {
        "vessel_shape": (3, 3, 3),
        "yard_shape": (3, 3, 3),
        "num_containers": 27,
        "group_num": 3,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    },
    "small_with_margin": {
        "vessel_shape": (3, 3, 3),
        "yard_shape": (3, 4, 3),
        "num_containers": 27,
        "group_num": 3,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": False,
        "reward_clip": False,
        "stack_fill_penalty": True,
        "container_sizes": True,
        "enable_imo": True,
    },
    "medium": {
        "vessel_shape": (4, 4, 4),
        "yard_shape": (4, 4, 4),
        "num_containers": 64,
        "group_num": 4,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    },
    "medium_with_margin": {
        "vessel_shape": (4, 4, 4),
        "yard_shape": (4, 5, 4),
        "num_containers": 64,
        "group_num": 4,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": True,
        "enable_imo": True,
    },
    "large": {
        "vessel_shape": (6, 6, 5),
        "yard_shape": (6, 6, 5),
        "num_containers": 180,
        "group_num": 6,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    },
    "large_with_margin": {
        "vessel_shape": (6, 6, 5),
        "yard_shape": (6, 7, 5),
        "num_containers": 180,
        "group_num": 6,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": True,
        "enable_imo": True,
    },
    "large_v2": {
        "vessel_shape": (8, 5, 5),
        "yard_shape": (8, 5, 5),
        "num_containers": 200,
        "group_num": 8,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    },
    "large_v2_with_margin": {
        "vessel_shape": (8, 5, 5),
        "yard_shape": (8, 7, 5),
        "num_containers": 200,
        "group_num": 8,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": True,
        "enable_imo": True,
    },
    "large_v3": {
        "vessel_shape": (10, 6, 5),
        "yard_shape": (10, 6, 5),
        "num_containers": 300,
        "group_num": 10,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    },
    "large_v3_with_margin": {
        "vessel_shape": (10, 6, 5),
        "yard_shape": (10, 7, 5),
        "num_containers": 300,
        "group_num": 10,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": True,
        "enable_imo": True,
    },
    "large_v4": {
        "vessel_shape": (10, 8, 5),
        "yard_shape": (10, 8, 5),
        "num_containers": 400,
        "group_num": 10,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    },
    "large_v4_with_margin": {
        "vessel_shape": (10, 8, 5),
        "yard_shape": (10, 9, 5),
        "num_containers": 400,
        "group_num": 10,
        "group_placement": "random",
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": True,
        "enable_imo": True,
    },
}

ENVIRONMENT_SIZES = tuple(ENVIRONMENT_CONFIGS)


def set_config(
    size: str = "small",
    seed: int = 42,
    reward_norm: Optional[bool] = None,
    reward_clip: Optional[bool] = None,
) -> Dict[str, Any]:
    """StackEnv configuration for size.

    reward_norm / reward_clip override the size's default only when
    explicitly given (see add_reward_arguments()).
    """

    if size not in ENVIRONMENT_CONFIGS:
        raise ValueError(f"Invalid size {size!r}. Choose one of: {', '.join(ENVIRONMENT_SIZES)}.")

    config = {**ENVIRONMENT_CONFIGS[size], "seed": seed}

    if reward_norm is not None:
        config["reward_norm"] = reward_norm
    if reward_clip is not None:
        config["reward_clip"] = reward_clip

    return config


def add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    """Explicit --[no-]reward_norm / --[no-]reward_clip overrides of the
    size's set_config() default, shared by every training entry point."""

    for name in ("reward_norm", "reward_clip"):
        parser.add_argument(
            f"--{name}",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"Override set_config()'s {name} for --size (default: keep it).",
        )
