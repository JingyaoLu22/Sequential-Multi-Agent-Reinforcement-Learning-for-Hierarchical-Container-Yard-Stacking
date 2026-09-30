"""Small environment and network settings shared by the tests."""

from dataclasses import replace

import torch

from stack.configs.environments import set_config
from stack.configs.hierarchical_config import get_hierarchical_config
from stack.run_sequential_hppo import build_training_system


def environment(seed: int = 0, **overrides) -> dict:
    """small_with_margin (40ft containers and IMO) on a 3 x 3 x 2 yard:
    12-step episodes, so rollouts cross episode boundaries."""
    return {**set_config("small_with_margin", seed), "vessel_shape": (3, 3, 2), "yard_shape": (3, 3, 2),
            "num_containers": 12, **overrides}


def algorithm(**overrides):
    small = dict(embed_dim=16, n_heads=2, n_layers=1, vf_dim=16, dropout=0.0)
    return replace(get_hierarchical_config("small"), **{**small, **overrides})


def training_system(environment_config: dict, algorithm_config, num_envs: int = 2, vec_backend: str = "sync"):
    """(env, trainer), with weights fixed by torch.manual_seed(0)."""
    torch.manual_seed(0)
    env, _bay, _row, _critic, trainer = build_training_system(
        environment_config, algorithm_config, torch.device("cpu"), num_envs=num_envs, vec_backend=vec_backend
    )
    return env, trainer
