"""
train.py

High-level training utilities for the stowage stack environment using
Maskable PPO. The functions used are :

    create_env       - Build training and evaluation envs.
    create_model     - Instantiate a MaskablePPO agent model (Transformer Pointer Net or Flat MLP).
    create_callbacks - Assemble the SB3 callback chain used during `.learn()`.
    train            - Function to train the model.
"""

from __future__ import annotations
from typing import Any
from sb3_contrib.ppo_mask import MaskablePPO
from envs.stack_gym import StackEnv
from envs.hierarchical_envs.hierarchical_low_level_env import HierarchicalLowLevelEnv
from envs.hierarchical_envs.hierarchical_high_level_env import HierarchicalHighLevelEnv
from sb3_contrib.common.wrappers import ActionMasker
from utils import (
    MaskedEvalCallback,
    mask_fn,
    create_parallel_envs,
    create_parallel_hierarchical_envs,
    create_parallel_high_level_envs,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from models.transformer_policy import MaskableTransformerPolicy
from models.transformer_bay_policy import MaskableBayTransformerPolicy
from models.joint_hierarchical_policy import (
    MaskableJointMlpPolicy,
    MaskableJointTransformerPolicy,
)
from models.joint_diffobs_policy import MaskableDiffObsJointTransformerPolicy


def create_env(
    config: dict,
    seed: int | None,
    render_mode: str | None,
    parallel: bool,
    n_parallel_envs: int,
    hierarchical: bool = False,
    high_level_policy_type: str = "rule_based_grouped",
    hierarchical_high_level: bool = False,
    low_level_agent_type: str = "rule_based_grouped",
    low_level_model_path: str | None = None,
    joint_hierarchical: bool = False,
    hierarchical_diffobs: bool = False,
) -> tuple[SubprocVecEnv | StackEnv | HierarchicalLowLevelEnv | HierarchicalHighLevelEnv, ActionMasker]:
    """
    Build training and evaluation environments for the stowage stack task.

    Parallel training is enabled using parallel parameters.

    Reward normalisation and clipping are disabled for the evaluation env so
    that reported scores reflect the true (un-scaled) reward signal.

    When ``hierarchical=True``, environments are wrapped with
    ``HierarchicalLowLevelEnv`` which embeds a fixed high-level bay-selection
    agent.  The RL agent only learns the low-level stack selection.

    Args:
        config (dict): Environment configuration dict.
        seed (int | None): Random seed forwarded to the environment constructors.
        render_mode (str | None): Render mode for the non-parallel env
        parallel (bool): If True, create a parallelized training env.
        n_parallel_envs (int): Number of parallel worker processes when
            parallel=True.  Ignored when parallel=False.
        hierarchical (bool): Wrap envs with HierarchicalLowLevelEnv.
        high_level_policy_type (str): Policy for the high-level agent
            ("rule_based_grouped", "rule_based", "random").  Only used
            when hierarchical=True.

    Returns:
        tuple with training environment and evaluation environment.
    """
    # Build a separate evaluation config to turn off any reward shaping that would distort the evaluation score.
    eval_config = config.copy()

    eval_config["reward_norm"] = False  # Disable reward normalization for evaluation
    eval_config["reward_clip"] = False  # Disable reward clipping for evaluation

    if hierarchical_diffobs:
        # Differentiated-observation hierarchical: bay head sees bay-level
        # features, stack head sees per-bay stack features.
        config["observation_type"] = "hierarchical_diff_obs"
        eval_config["observation_type"] = "hierarchical_diff_obs"
        if parallel:
            train_env = create_parallel_envs(
                config, n_envs=n_parallel_envs, vec_env_cls=SubprocVecEnv
            )
        else:
            train_env = StackEnv(config=config, render_mode=render_mode)

        eval_env = StackEnv(config=eval_config, render_mode=None)
        eval_env = ActionMasker(eval_env, mask_fn)

    elif joint_hierarchical:
        # Joint hierarchical: RL learns bay+stack simultaneously via
        # autoregressive policy.  Uses standard StackEnv directly.
        if parallel:
            train_env = create_parallel_envs(
                config, n_envs=n_parallel_envs, vec_env_cls=SubprocVecEnv
            )
        else:
            train_env = StackEnv(config=config, render_mode=render_mode)

        eval_env = StackEnv(config=eval_config, render_mode=None)
        eval_env = ActionMasker(eval_env, mask_fn)

    elif hierarchical_high_level:
        # High-level training: RL picks bay, low-level agent is fixed
        ll_model_path = low_level_model_path if low_level_agent_type == "trained_model" else None
        ll_policy_type = low_level_agent_type if low_level_agent_type != "trained_model" else "rule_based_grouped"

        if parallel:
            train_env = create_parallel_high_level_envs(
                config,
                low_level_policy_type=ll_policy_type,
                low_level_model_path=ll_model_path,
                n_envs=n_parallel_envs,
                vec_env_cls=SubprocVecEnv,
            )
        else:
            train_env = HierarchicalHighLevelEnv(
                config=config,
                low_level_policy_type=ll_policy_type,
                low_level_model_path=ll_model_path,
                render_mode=render_mode,
            )

        eval_env = HierarchicalHighLevelEnv(
            config=eval_config,
            low_level_policy_type=ll_policy_type,
            low_level_model_path=ll_model_path,
            render_mode=None,
        )
        eval_env = ActionMasker(eval_env, mask_fn)

    elif hierarchical:
        if parallel:
            train_env = create_parallel_hierarchical_envs(
                config,
                high_level_policy_type=high_level_policy_type,
                n_envs=n_parallel_envs,
                vec_env_cls=SubprocVecEnv,
            )
        else:
            train_env = HierarchicalLowLevelEnv(
                config=config,
                high_level_policy_type=high_level_policy_type,
                render_mode=render_mode,
            )

        eval_env = HierarchicalLowLevelEnv(
            config=eval_config,
            high_level_policy_type=high_level_policy_type,
            render_mode=None,
        )
        eval_env = ActionMasker(eval_env, mask_fn)
    else:
        if parallel:
            # Create parallel training envs
            train_env = create_parallel_envs(
                config, n_envs=n_parallel_envs, vec_env_cls=SubprocVecEnv
            )
            # Evaluation always runs in a single process.
            eval_env = StackEnv(config=eval_config, render_mode=None)
            # in sb3, the eval env must also be masked otherwise model chooses incorrect actions.
            eval_env = ActionMasker(eval_env, mask_fn)
        else:
            # No parallel training env.
            train_env = StackEnv(config=config, render_mode=render_mode)

            # Single-process evaluation env with the same render mode.
            eval_env = StackEnv(config=eval_config, render_mode=render_mode)
            eval_env = ActionMasker(eval_env, mask_fn)

    return train_env, eval_env


def create_model(
    train_env: SubprocVecEnv | StackEnv,
    config: dict,
    device: str,
    parallel: bool,
    n_parallel_envs: int,
    use_transformer: bool = False,
    embed_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    vf_dim: int = 128,
    tanh_clipping: float = 10.0,
    n_epochs: int = 10,
    lr: float = 3e-4,
    vf_coef: float = 0.5,
    ent_coef: float | None = None,
    batch_size: int = 64,
    n_steps_total: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr_decay: bool = False,
    run: Any = None,
    hierarchical_high_level: bool = False,
    n_stacks: int | None = None,
    n_rows_per_bay: int | None = None,
    joint_hierarchical: bool = False,
    hierarchical_diffobs: bool = False,
    group_num: int | None = None,
) -> MaskablePPO:
    """
    Instantiate a MaskablePPO agent for the stowage stack environment.

    Two policy architectures are supported:

    Transformer (use_transformer=True): uses the Pointer Network architecture.
    MLP (default): a standard two-hidden-layer network [256, 256, 32].

    The number of rollout steps per update (n_steps) is automatically
    updated based on the number of parallel envs to make sure number of PPO updates are
    consistent across parallel and non-parallel training runs.

    Args:
        train_env: Training environment.
        eval_env: Evaluation environment (unused here but kept for API symmetry).
        config (dict): Environment configuration (currently unused inside this
            function but available for future policy-specific settings).
        device (str): PyTorch device string, e.g. "cpu" or "cuda".
        parallel (bool): Whether the training env is vectorised.
        n_parallel_envs (int): Number of parallel workers (used to scale n_steps so the total rollout size stays near 2048)
        use_transformer (bool): Use the Transformer policy instead of MLP.
        embed_dim (int): Embedding dimension for the Transformer encoder.
        n_heads (int): Number of attention heads in the Transformer encoder.
        n_layers (int): Number of Transformer encoder layers.
        vf_dim (int): Hidden dimension of the value-function MLP head.
        tanh_clipping (float): Clip logits to [-tanh_clipping, tanh_clipping]
                         before computing action probabilities (Pointer Network trick).
        n_epochs (int): Number of PPO gradient update epochs per rollout.
        lr (float): Learning rate.
        vf_coef (float): Value-function loss coefficient in the PPO objective.
        run: Active Wandb run object, or None to disable TensorBoard logging.

    Returns:
        Model : MaskablePPO agent model ready for training.
    """
    # Scale n_steps inversely with the number of parallel envs so the total
    # number of transitions collected per update remains constant.
    if parallel:
        n_steps = n_steps_total // n_parallel_envs
    else:
        n_steps = n_steps_total

    # Determine container feature layout for transformer policies
    # when using stack_features_v3 (5 scalar features per stack).
    _container_kwargs = {}
    if config.get("observation_type") == "stack_features_v3":
        _container_kwargs = dict(container_start=2, container_dim=1)

    # Default entropy coefficient
    if ent_coef is None:
        ent_coef = 0.15 if (joint_hierarchical or hierarchical_diffobs) else 0.3

    if hierarchical_diffobs:
        n_bays = n_stacks // n_rows_per_bay
        bay_f_dim = group_num + 4
        stack_f_dim = 5 * group_num + 5
        policy = MaskableDiffObsJointTransformerPolicy
        policy_kwargs = dict(
            n_bays=n_bays,
            n_rows_per_bay=n_rows_per_bay,
            bay_f_dim=bay_f_dim,
            stack_f_dim=stack_f_dim,
            group_num=group_num,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            vf_dim=vf_dim,
            tanh_clipping=tanh_clipping,
        )
    elif joint_hierarchical:
        n_bays = n_stacks // n_rows_per_bay
        if use_transformer:
            policy = MaskableJointTransformerPolicy
            policy_kwargs = dict(
                n_stacks=n_stacks,
                n_bays=n_bays,
                n_rows_per_bay=n_rows_per_bay,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                vf_dim=vf_dim,
                tanh_clipping=tanh_clipping,
                **_container_kwargs,
            )
        else:
            policy = MaskableJointMlpPolicy
            policy_kwargs = dict(
                n_stacks=n_stacks,
                n_bays=n_bays,
                n_rows_per_bay=n_rows_per_bay,
                vf_dim=vf_dim,
            )
    elif use_transformer:
        if hierarchical_high_level:
            # Bay-level pointer network for high-level agent
            policy = MaskableBayTransformerPolicy
            policy_kwargs = dict(
                n_stacks=n_stacks,
                n_rows_per_bay=n_rows_per_bay,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                vf_dim=vf_dim,
                tanh_clipping=tanh_clipping,
                **_container_kwargs,
            )
        else:
            # Stack-level pointer network (original)
            policy = MaskableTransformerPolicy
            policy_kwargs = dict(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                vf_dim=vf_dim,
                tanh_clipping=tanh_clipping,
                **_container_kwargs,
            )
    else:
        # MLP policy. Critic and actor networks share same backbonne architecture
        # but have separate heads.
        policy = "MlpPolicy"
        policy_kwargs = dict(net_arch=[256, 256, 32])

    # Optional linear LR decay: ramps from `lr` down to 0 over training.
    if lr_decay:
        from stable_baselines3.common.utils import get_linear_fn
        learning_rate = get_linear_fn(lr, 0.0, 1.0)
    else:
        learning_rate = lr

    model = MaskablePPO(
        policy=policy,
        env=train_env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=0.2,
        verbose=1,
        device=device,
        tensorboard_log=f"runs/{run.id}" if run is not None else None,
    )

    return model


def create_callbacks(
    eval_env: ActionMasker,
    eval_freq: int,
    n_eval_episodes: int,
    save_model_flag: bool,
    save_dir: str,
    save_filename: str,
    max_reward_threshold: float | None = None,
    run: Any = None,
) -> CallbackList:
    """
    Build the SB3 CallbackList used during model training for evaluation metrics
    and model saving.

    Model is saved whenever a new best mean evaluation reward is achieved,

    The MaskedEvalCallback periodically rolls out the
    current policy on the evaluation environment and saves the
    best checkpoint.

    Args:
        eval_env: Action-masked evaluation environment.
        eval_freq (int): How often (in environment steps) to run evaluation.
        n_eval_episodes (int): Number of episodes per evaluation run.
        save_model_flag (bool): Whether to save the best model to disk.
        save_dir (str): Directory in which to save the best model checkpoint.
        save_filename (str): Base filename for the checkpoint (without extension).
        max_reward_threshold (float | None): Used during evaluation to
                 compute the percentage of episodes that achieve perfect episode for given yard.
        run: Active W&B run object, or None to skip W&B logging.

    Returns:
        CallbackList: List of sb3 callbacks.
    """
    # Core evaluation callback: tracks best reward and saves checkpoints.
    eval_callback = MaskedEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        save_best_model=save_model_flag,
        save_dir=save_dir,
        save_filename=save_filename,
        max_reward_threshold=max_reward_threshold,
    )

    callback_list = [eval_callback]
    if run is not None:
        # Stream scalars (reward, loss, entropy, etc) to Wandb.
        callback_list.append(WandbCallback(verbose=0))

    callbacks = CallbackList(callback_list)

    return callbacks


def train(
    config: dict,
    device: str,
    render_mode: str | None,
    parallel: bool,
    n_parallel_envs: int,
    eval_freq: int,
    n_eval_episodes: int,
    timesteps: int,
    save_model_flag: bool,
    save_dir: str,
    save_filename: str,
    max_reward_threshold: float | None = None,
    use_transformer: bool = False,
    embed_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    vf_dim: int = 128,
    tanh_clipping: float = 10.0,
    n_epochs: int = 10,
    lr: float = 3e-4,
    vf_coef: float = 0.5,
    ent_coef: float | None = None,
    batch_size: int = 64,
    n_steps_total: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr_decay: bool = False,
    run: Any = None,
    hierarchical: bool = False,
    high_level_policy_type: str = "rule_based_grouped",
    hierarchical_high_level: bool = False,
    low_level_agent_type: str = "rule_based_grouped",
    low_level_model_path: str | None = None,
    joint_hierarchical: bool = False,
    hierarchical_diffobs: bool = False,
) -> None:
    """
    Training function.

    Args:
        config (dict): Environment configuration dict.
        device (str): PyTorch device string, e.g. "cpu" or "cuda".
        render_mode (str | None): Render mode for single-process environments.
        parallel (bool): Enable parallel training.
        n_parallel_envs (int): Number of parallel worker processes.
        eval_freq (int): Evaluation frequency in environment steps.
        n_eval_episodes (int): Number of rollout episodes per evaluation.
        timesteps (int): Total environment interaction steps to train for.
        save_model_flag (bool): Save the model or not.
        save_dir (str): Directory for model checkpoints.
        save_filename (str): Base filename for saved checkpoints.
        max_reward_threshold (float | None): Used during evaluation to
                 compute the percentage of episodes that achieve perfect episode for given yard.
        use_transformer (bool): Use Transformer Pointer Network architecture.
        embed_dim (int): Transformer encoder embedding dimension.
        n_heads (int): Number of attention heads in the Transformer encoder.
        n_layers (int): Number of Transformer encoder layers.
        vf_dim (int): Hidden dimension of the value-function head.
        tanh_clipping (float): Logit clipping scale for the Pointer Network head.
        n_epochs (int): PPO gradient update epochs per rollout batch.
        lr (float): Learning rate.
        vf_coef (float): Value-function loss coefficient in the PPO objective.
        run: Active W&B run object, or None to skip W&B logging.
        hierarchical (bool): Use HierarchicalLowLevelEnv wrapper with a fixed
            high-level agent for bay selection.
        high_level_policy_type (str): Policy for the high-level agent
            ("rule_based_grouped", "rule_based", "random").  Only used
            when hierarchical=True.
    """
    seed = config.get("seed", None)

    # Build environments
    train_env, eval_env = create_env(
        config,
        seed=seed,
        render_mode=render_mode,
        parallel=parallel,
        n_parallel_envs=n_parallel_envs,
        hierarchical=hierarchical,
        high_level_policy_type=high_level_policy_type,
        hierarchical_high_level=hierarchical_high_level,
        low_level_agent_type=low_level_agent_type,
        low_level_model_path=low_level_model_path,
        joint_hierarchical=joint_hierarchical,
        hierarchical_diffobs=hierarchical_diffobs,
    )

    # Compute n_stacks and n_rows_per_bay for the bay-level transformer
    n_stacks = config.get("yard_shape", (2, 2, 2))[0] * config.get("yard_shape", (2, 2, 2))[1]
    n_rows_per_bay = config.get("yard_shape", (2, 2, 2))[1]

    # Build model
    model = create_model(
        train_env,
        config=config,
        device=device,
        parallel=parallel,
        n_parallel_envs=n_parallel_envs,
        use_transformer=use_transformer,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        vf_dim=vf_dim,
        tanh_clipping=tanh_clipping,
        n_epochs=n_epochs,
        lr=lr,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps_total=n_steps_total,
        gamma=gamma,
        gae_lambda=gae_lambda,
        lr_decay=lr_decay,
        run=run,
        hierarchical_high_level=hierarchical_high_level,
        n_stacks=n_stacks,
        n_rows_per_bay=n_rows_per_bay,
        joint_hierarchical=joint_hierarchical,
        hierarchical_diffobs=hierarchical_diffobs,
        group_num=config.get("group_num", 3),
    )

    # Initialize callbacks
    callbacks = create_callbacks(
        eval_env,
        eval_freq,
        n_eval_episodes,
        save_model_flag,
        save_dir,
        save_filename,
        max_reward_threshold=max_reward_threshold,
        run=run,
    )

    # Train
    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Close envs
    train_env.close()
    eval_env.close()

    return
