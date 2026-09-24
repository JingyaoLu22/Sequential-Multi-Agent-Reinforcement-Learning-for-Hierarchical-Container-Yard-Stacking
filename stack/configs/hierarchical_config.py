"""
Configuration for the separated hierarchical multi-agent PPO pipeline.

This file centralizes model and training hyperparameters.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class HierarchicalConfig:
    """
    Hyperparameters for the separated hierarchical PPO pipeline.

    Model architecture defaults are independent of environment size. Training
    hyperparameters are selected by :func:`get_hierarchical_config` from the
    requested environment size and may then be overridden from the CLI.
    """

    # Name of the size-dependent training profile used to create this config.
    # ``custom`` keeps old checkpoints (which do not contain this field)
    # backwards compatible when they are loaded for evaluation.
    training_profile: str = "custom"

    # ==================================================================
    # Transformer architecture
    # ==================================================================

    # Original Transformer embedding dimension.
    embed_dim: int = 128

    # Original number of attention heads.
    n_heads: int = 4

    # Original number of TransformerEncoder layers.
    n_layers: int = 2

    # Original policy-level Transformer dropout.
    dropout: float = 0.1

    # Hidden dimension of centralized critic MLP.
    vf_dim: int = 128

    # Original PointerDecoder tanh clipping.
    tanh_clipping: float = 10.0

    # ==================================================================
    # stack_features_v3 container context
    # ==================================================================
    #
    # Original stack_features_v3:
    #
    #   [0] majority_group
    #   [1] num_occupied
    #   [2] current_container_group
    #   [3] adj_majority_group
    #   [4] stack_index
    #   ...
    #
    # Therefore:
    #
    #   container_start = 2
    #   container_dim   = 1
    #
    # These are NOT new features.
    # They describe the existing StackEnv representation.
    # ==================================================================

    container_start: int = 2
    container_dim: int = 1

    # Keep current-container information inside Transformer tokens,
    # consistent with the original implementation.
    include_container_in_encoder: bool = True

    # ==================================================================
    # PPO / GAE
    # ==================================================================

    learning_rate: float = 3e-4

    n_epochs: int = 10

    batch_size: int = 64

    # Number of COMPLETE hierarchical transitions collected before
    # one PPO update:
    #
    #   Bay decision
    #       ->
    #   Row decision
    #       ->
    #   ONE StackEnv.step()
    #
    buffer_size: int = 2048

    gamma: float = 0.99

    gae_lambda: float = 0.95

    clip_range: float = 0.2

    # The old joint_hierarchical configuration used 0.15 by default.
    ent_coef: float = 0.15

    vf_coef: float = 0.5

    # Standard PPO/SB3 gradient clipping value.
    max_grad_norm: float = 0.5

    # Preserve standard PPO advantage normalization.
    normalize_advantage: bool = True

    # ==================================================================
    # Training / evaluation runtime defaults
    # ==================================================================

    total_timesteps: int = 1_500_000

    eval_freq: int = 25_000

    n_eval_episodes: int = 10

    checkpoint_freq: int = 1_000_000

    # ==================================================================
    # Validation
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:
        """
        Validate configuration values immediately.

        Because this dataclass is frozen, configuration mistakes are
        detected at construction time and parameters cannot later be
        accidentally modified during training.
        """

        # --------------------------------------------------------------
        # Transformer
        # --------------------------------------------------------------

        if not self.training_profile:
            raise ValueError(
                "training_profile cannot be empty."
            )

        if self.embed_dim <= 0:
            raise ValueError(
                "embed_dim must be positive."
            )

        if self.n_heads <= 0:
            raise ValueError(
                "n_heads must be positive."
            )

        if (
            self.embed_dim
            % self.n_heads
            != 0
        ):
            raise ValueError(
                "embed_dim must be divisible "
                "by n_heads."
            )

        if self.n_layers <= 0:
            raise ValueError(
                "n_layers must be positive."
            )

        if not (
            0.0
            <= self.dropout
            < 1.0
        ):
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if self.vf_dim <= 0:
            raise ValueError(
                "vf_dim must be positive."
            )

        if self.tanh_clipping < 0:
            raise ValueError(
                "tanh_clipping cannot be negative."
            )

        # --------------------------------------------------------------
        # Observation metadata
        # --------------------------------------------------------------

        if self.container_start < 0:
            raise ValueError(
                "container_start cannot be negative."
            )

        if self.container_dim <= 0:
            raise ValueError(
                "container_dim must be positive."
            )

        # --------------------------------------------------------------
        # PPO
        # --------------------------------------------------------------

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.n_epochs <= 0:
            raise ValueError(
                "n_epochs must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.buffer_size <= 0:
            raise ValueError(
                "buffer_size must be positive."
            )

        if not (
            0.0
            <= self.gamma
            <= 1.0
        ):
            raise ValueError(
                "gamma must be in [0, 1]."
            )

        if not (
            0.0
            <= self.gae_lambda
            <= 1.0
        ):
            raise ValueError(
                "gae_lambda must be in [0, 1]."
            )

        if self.clip_range <= 0:
            raise ValueError(
                "clip_range must be positive."
            )

        if self.ent_coef < 0:
            raise ValueError(
                "ent_coef cannot be negative."
            )

        if self.vf_coef < 0:
            raise ValueError(
                "vf_coef cannot be negative."
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive."
            )

        # --------------------------------------------------------------
        # Runtime
        # --------------------------------------------------------------

        if self.total_timesteps <= 0:
            raise ValueError(
                "total_timesteps must be positive."
            )

        if self.eval_freq <= 0:
            raise ValueError(
                "eval_freq must be positive."
            )

        if self.n_eval_episodes <= 0:
            raise ValueError(
                "n_eval_episodes must be positive."
            )

        if self.checkpoint_freq <= 0:
            raise ValueError(
                "checkpoint_freq must be positive."
            )

    # ==================================================================
    # Convenience dictionaries
    # ==================================================================

    def agent_kwargs(
        self,
    ) -> Dict[str, Any]:
        """
        Parameters shared by AgentB and AgentR.

        Environment-specific dimensions such as n_bays and n_rows are
        deliberately NOT included here.
        """

        return {
            "embed_dim":
                self.embed_dim,

            "n_heads":
                self.n_heads,

            "n_layers":
                self.n_layers,

            "dropout":
                self.dropout,

            "tanh_clipping":
                self.tanh_clipping,

            "include_container_in_encoder":
                self.include_container_in_encoder,

            "container_start":
                self.container_start,

            "container_dim":
                self.container_dim,
        }

    def critic_kwargs(
        self,
    ) -> Dict[str, Any]:
        """
        Parameters used to construct CentralizedCritic.

        n_stacks is derived later from the actual StackEnv.
        """

        return {
            "embed_dim":
                self.embed_dim,

            "n_heads":
                self.n_heads,

            "n_layers":
                self.n_layers,

            "dropout":
                self.dropout,

            "vf_dim":
                self.vf_dim,

            "include_container_in_encoder":
                self.include_container_in_encoder,

            "container_start":
                self.container_start,

            "container_dim":
                self.container_dim,
        }

    def trainer_kwargs(
        self,
    ) -> Dict[str, Any]:
        """
        Parameters used to construct SequentialPPOTrainer.
        """

        return {
            "learning_rate":
                self.learning_rate,

            "n_epochs":
                self.n_epochs,

            "batch_size":
                self.batch_size,

            "gamma":
                self.gamma,

            "gae_lambda":
                self.gae_lambda,

            "clip_range":
                self.clip_range,

            "ent_coef":
                self.ent_coef,

            "vf_coef":
                self.vf_coef,

            "max_grad_norm":
                self.max_grad_norm,

            "normalize_advantage":
                self.normalize_advantage,
        }

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Return complete configuration as a dictionary.

        Useful later for:

            logging
            W&B
            experiment metadata
            saving configuration
        """

        return asdict(
            self
        )


# ======================================================================
# Default configuration
# ======================================================================


DEFAULT_HIERARCHICAL_CONFIG = (
    HierarchicalConfig()
)


# ======================================================================
# Size-dependent training profiles
# ======================================================================
#
# Only parameters that should scale with the problem size are repeated
# here. Architecture, GAE, and evaluation defaults continue to come from
# HierarchicalConfig above. checkpoint_freq is tiered here too, since a
# 28M-step "massive" run at the default resume-checkpoint cadence would
# otherwise write the full training state after every single rollout.
# ======================================================================

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


SIZE_TO_TRAINING_PROFILE: Dict[str, str] = {
    "small": "small",
    "small_with_margin": "small",
    "medium": "medium",
    "medium_with_margin": "medium",
    "large": "large",
    "large_with_margin": "large",
    "large_v2": "large",
    "large_v2_with_margin": "large",
    "large_v3": "massive",
    "large_v3_with_margin": "massive",
    "large_v4": "massive",
    "large_v4_with_margin": "massive",
}


def get_training_profile_name(size: str) -> str:
    """Return the automatic training profile for an environment size."""

    try:
        return SIZE_TO_TRAINING_PROFILE[size]
    except KeyError as error:
        valid_sizes = ", ".join(sorted(SIZE_TO_TRAINING_PROFILE))
        raise ValueError(
            f"Unknown environment size: {size}. Valid sizes: {valid_sizes}"
        ) from error


def get_hierarchical_config(
    size: str,
    training_profile: Optional[str] = None,
) -> HierarchicalConfig:
    """Build the algorithm config for ``size``.

    Parameters
    ----------
    size:
        Environment size passed to ``stack.run_sequential_hppo --size``.
    training_profile:
        Optional explicit profile name. ``None`` or ``"auto"`` selects the
        profile through :data:`SIZE_TO_TRAINING_PROFILE`.
    """

    if training_profile in (None, "auto"):
        profile_name = get_training_profile_name(size)
    else:
        profile_name = training_profile

    if profile_name not in TRAINING_PROFILES:
        valid_profiles = ", ".join(sorted(TRAINING_PROFILES))
        raise ValueError(
            f"Unknown training profile: {profile_name}. "
            f"Valid profiles: {valid_profiles}"
        )

    return replace(
        DEFAULT_HIERARCHICAL_CONFIG,
        training_profile=profile_name,
        **TRAINING_PROFILES[profile_name],
    )
