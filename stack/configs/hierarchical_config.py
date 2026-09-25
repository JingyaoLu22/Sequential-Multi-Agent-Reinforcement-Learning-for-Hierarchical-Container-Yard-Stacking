"""
Sequential HPPO hyperparameters and the per-size training profiles.

HierarchicalConfig holds every hyperparameter. Those that scale with the
problem size have no default: they come only from TRAINING_PROFILES, via
get_hierarchical_config(size), or from a saved checkpoint's config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .environments import ENVIRONMENT_SIZES


@dataclass(frozen=True)
class HierarchicalConfig:

    # Set by the size's training profile (TRAINING_PROFILES).
    total_timesteps: int
    learning_rate: float
    # Complete Bay -> Row -> StackEnv.step() transitions per PPO update.
    buffer_size: int
    batch_size: int
    n_epochs: int
    clip_range: float
    ent_coef: float
    checkpoint_freq: int

    # Pointer-network actors and centralized critic.
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    vf_dim: int = 128
    tanh_clipping: float = 10.0

    # The current-container group is feature 2 of every stack_features_v3
    # token, and it stays inside the tokens the encoder sees.
    container_start: int = 2
    container_dim: int = 1
    include_container_in_encoder: bool = True

    # PPO / GAE settings shared by every size.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # Periodic evaluation during training.
    eval_freq: int = 25_000
    n_eval_episodes: int = 10

    # Profile the config was built from; "custom" for checkpoints saved
    # before this field existed.
    training_profile: str = "custom"

    def actor_kwargs(self) -> Dict[str, Any]:
        """PointerActor arguments shared by the Bay and Row actors."""

        return dict(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            tanh_clipping=self.tanh_clipping,
            include_container_in_encoder=self.include_container_in_encoder,
            container_start=self.container_start,
            container_dim=self.container_dim,
        )

    def critic_kwargs(self) -> Dict[str, Any]:
        """CentralizedCritic arguments."""

        return dict(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            vf_dim=self.vf_dim,
            include_container_in_encoder=self.include_container_in_encoder,
            container_start=self.container_start,
            container_dim=self.container_dim,
        )

    def trainer_kwargs(self) -> Dict[str, Any]:
        """SequentialPPOTrainer hyperparameters."""

        return dict(
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            normalize_advantage=self.normalize_advantage,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# checkpoint_freq is tiered too: a 28M-step "massive" run would otherwise
# write its full training state far too often.
TRAINING_PROFILES: Dict[str, Dict[str, Any]] = {
    "small": {
        "total_timesteps": 1_000_000,
        "learning_rate": 1e-4,
        "buffer_size": 2_048,
        "batch_size": 64,
        "n_epochs": 3,
        "clip_range": 0.20,
        "ent_coef": 0.15,
        "checkpoint_freq": 100_000,
    },
    "medium": {
        "total_timesteps": 2_000_000,
        "learning_rate": 1e-4,
        "buffer_size": 4_096,
        "batch_size": 128,
        "n_epochs": 3,
        "clip_range": 0.20,
        "ent_coef": 0.15,
        "checkpoint_freq": 100_000,
    },
    "large": {
        "total_timesteps": 10_000_000,
        "learning_rate": 1e-4,
        "buffer_size": 8_192,
        "batch_size": 256,
        "n_epochs": 3,
        "clip_range": 0.15,
        "ent_coef": 0.15,
        "checkpoint_freq": 250_000,
    },
    "massive": {
        "total_timesteps": 28_000_000,
        "learning_rate": 1e-4,
        "buffer_size": 8_192,
        "batch_size": 256,
        "n_epochs": 5,
        "clip_range": 0.15,
        "ent_coef": 0.05,
        "checkpoint_freq": 250_000,
    },
}


# A size and its "_with_margin" variant share a training profile.
_BASE_SIZE_TO_TRAINING_PROFILE: Dict[str, str] = {
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large_v2": "large",
    "large_v3": "massive",
    "large_v4": "massive",
}

# Every size of configs/environments.py; a size without a profile fails
# here at import time.
SIZE_TO_TRAINING_PROFILE: Dict[str, str] = {
    size: _BASE_SIZE_TO_TRAINING_PROFILE[size.removesuffix("_with_margin")]
    for size in ENVIRONMENT_SIZES
}


def get_hierarchical_config(
    size: str,
    training_profile: Optional[str] = None,
) -> HierarchicalConfig:
    """Config for ``size``: its training profile, or the explicitly named
    one when training_profile is neither None nor "auto"."""

    if training_profile in (None, "auto"):
        training_profile = SIZE_TO_TRAINING_PROFILE[size]

    return HierarchicalConfig(
        training_profile=training_profile,
        **TRAINING_PROFILES[training_profile],
    )
