"""
train.py

High-level training utilities for the stack environment using
Maskable PPO. The functions used are :

    create_env       - Build training and evaluation envs.
    create_model     - Instantiate a MaskablePPO agent model (Transformer Pointer Net or Flat MLP).
    create_callbacks - Assemble the SB3 callback chains.
    train            - Function to train the model.
"""

from __future__ import annotations
import os
from typing import Any
from sb3_contrib.ppo_mask import MaskablePPO
from envs.stack_gym import StackEnv
from envs.hierarchical_envs.hierarchical_low_level_env import HierarchicalLowLevelEnv
from envs.hierarchical_envs.hierarchical_high_level_env import HierarchicalHighLevelEnv
from sb3_contrib.common.wrappers import ActionMasker
from utils import (
    MaskedEvalCallback,
    SaveModelCallback,
    mask_fn,
    create_parallel_envs,
    create_parallel_hierarchical_envs,
    create_parallel_high_level_envs,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_linear_fn
from wandb.integration.sb3 import WandbCallback
from models.transformer_policy import MaskableTransformerPolicy
from models.transformer_bay_policy import MaskableBayTransformerPolicy
from models.joint_hierarchical_policy import (
    MaskableJointMlpPolicy,
    MaskableJointTransformerPolicy,
)
from models.joint_diffobs_policy import MaskableDiffObsJointTransformerPolicy
from models.sequential_hierarchical_policy import MaskableSequentialTransformerPolicy
from models.sequential_hppo import SequentialHPPO


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

    When hierarchical=True, environments are wrapped with
    HierarchicalLowLevelEnv which embeds a fixed high-level bay-selection
    agent. The RL agent only learns the low-level stack selection.

    Args:
        config (dict): Environment configuration dict.
        seed (int | None): Random seed forwarded to the environment constructors.
        render_mode (str | None): Render mode for the non-parallel env
        parallel (bool): If True, create a parallelized training env.
        n_parallel_envs (int): Number of parallel worker processes when
            parallel=True.  Ignored when parallel=False.
        hierarchical (bool): Wrap envs with HierarchicalLowLevelEnv, using a
            fixed high-level bay-selection agent while the RL agent learns
            only the low-level stack selection.
        high_level_policy_type (str): Policy for the high-level agent
            ("rule_based_grouped", "rule_based", "random").  Only used
            when hierarchical=True.
        hierarchical_high_level (bool): Wrap envs with HierarchicalHighLevelEnv,
            using a fixed (or trained) low-level stack-selection agent while
            the RL agent learns only the high-level bay selection.
        low_level_agent_type (str): Policy for the low-level agent when
            hierarchical_high_level=True ("rule_based_grouped", "rule_based",
            "random", "trained_model"). If "trained_model", low_level_model_path
            must be provided; otherwise falls back to "rule_based_grouped".
        low_level_model_path (str | None): Path to a trained low-level policy
            model, used when hierarchical_high_level=True and
            low_level_agent_type="trained_model".
        joint_hierarchical (bool): Use plain StackEnv instances but signal
            that training uses separate joint high-level/low-level policy
            networks (bay selection and stack selection).
        hierarchical_diffobs (bool): Differentiated-observation hierarchical
            mode, where the bay head sees bay-level features and the stack
            head sees per-bay stack features. Sets
            config["observation_type"] = "hierarchical_diff_obs" on both the
            training and evaluation configs.

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
        # Learns suboptimal policy.
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
        # Joint hierarchical: For Training separate policy netowrks for selecting a bay (high-level policy)
        # and then a stack in the selected bay (low-level policy)
        if parallel:
            train_env = create_parallel_envs(
                config, n_envs=n_parallel_envs, vec_env_cls=SubprocVecEnv
            )
        else:
            train_env = StackEnv(config=config, render_mode=render_mode)

        eval_env = StackEnv(config=eval_config, render_mode=None)
        eval_env = ActionMasker(eval_env, mask_fn)

    elif hierarchical_high_level:
        # For sequential Hierarchical Training where one-level is fixed while the other is trained.
        if low_level_agent_type == "trained_model" and (
            not low_level_model_path or not os.path.isfile(low_level_model_path)
        ):
            raise ValueError(
                "low_level_agent_type='trained_model' requires low_level_model_path "
                f"to point to an existing model file, got {low_level_model_path!r}."
            )

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
            # In sb3 the eval env must also be masked otherwise model chooses incorrect actions.
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
    sequential_hppo: bool = False,
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
        config (dict): Environment configuration.
        device (str): PyTorch device string, e.g. "cpu" or "cuda".
        parallel (bool): Whether the training env is vectorised.
        n_parallel_envs (int): Number of parallel workers.
        use_transformer (bool): Use the Transformer policy instead of MLP.
        embed_dim (int): Embedding dimension for the Transformer encoder.
        n_heads (int): Number of attention heads in the Transformer encoder.
        n_layers (int): Number of Transformer encoder layers.
        vf_dim (int): Hidden dimension of the value-function MLP head.
        tanh_clipping (float): Clip logits to [-tanh_clipping, tanh_clipping] before computing action probabilities).
        n_epochs (int): Number of PPO gradient update epochs per rollout.
        lr (float): Learning rate.
        vf_coef (float): Value-function loss coefficient in the PPO objective.
        ent_coef (float | None): Entropy loss coefficient in the PPO objective.
        batch_size (int): Batch size for PPO updates.
        n_steps_total (int): Total number of rollout steps per update (before scaling for parallel envs).
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): GAE lambda parameter for advantage estimation.
        lr_decay (bool): Whether to use linear learning rate decay.
        max_grad_norm (float): Maximum gradient norm for clipping.
        run (Any): Active Wandb run object, or None to disable TensorBoard logging.
        hierarchical_high_level (bool): Whether to use a hierarchical high-level policy for sequntial training.
        n_stacks (int | None): Number of stacks in the yard.
        n_rows_per_bay (int | None): Number of rows per bay in the yard.
        joint_hierarchical (bool): Whether to use a joint hierarchical policy.
        hierarchical_diffobs (bool): Whether to use a differentiated observation hierarchical policy (currently not working).
        group_num (int | None): Number of container groups in the yard.
        sequential_hppo (bool): Whether to use Sequential HPPO (separate bay actor, row actor and critic).
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
    # when using stack_features_v3 (5 or 8 scalar features per stack).
    # Container group is always at index 2 with dim 1, regardless of container_sizes.
    _container_kwargs = {}
    if config.get("observation_type") == "stack_features_v3":
        _container_kwargs = dict(container_start=2, container_dim=1)

    # Default entropy coefficient
    if ent_coef is None:
        ent_coef = 0.15 if (joint_hierarchical or hierarchical_diffobs or sequential_hppo) else 0.3

    
    if hierarchical_diffobs:
        # High level and low level policies receive separate observations (does not work currently)
        if n_stacks is None or n_rows_per_bay is None or group_num is None:
            raise ValueError("hierarchical_diffobs=True requires n_stacks, n_rows_per_bay, and group_num")
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
    elif sequential_hppo:
        # Sequential HPPO: separate bay actor, row actor and critic, updated one after another by SequentialHPPO.
        n_bays = n_stacks // n_rows_per_bay
        policy = MaskableSequentialTransformerPolicy
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
    elif joint_hierarchical:
        # Joint hierarchical policy: separate high-level and low-level policies for bay selection and stack selection.
        # Same observation is passed to both policies.
        n_bays = n_stacks // n_rows_per_bay
        if use_transformer:
            # Use Transformer based Pointer Networks
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
            # Use MLP based policy
            policy = MaskableJointMlpPolicy
            policy_kwargs = dict(
                n_stacks=n_stacks,
                n_bays=n_bays,
                n_rows_per_bay=n_rows_per_bay,
                vf_dim=vf_dim,
            )
    # Mainly used for sequential hierarchical training.
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
            # Stack-level pointer network for low-level agent
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
        learning_rate = get_linear_fn(lr, 0.0, 1.0)
    else:
        learning_rate = lr

    # SequentialHPPO is MaskablePPO with the Sequential HPPO train().
    model = (SequentialHPPO if sequential_hppo else MaskablePPO)(
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
    checkpoint_freq: int = 1_000_000,
    run: Any = None,
) -> CallbackList:
    """
    Build the SB3 CallbackList used during model training for evaluation metrics
    and model saving.

    Model is saved in three ways:
    1. Best eval model: Saved when a new best mean evaluation reward is achieved
    2. Checkpoints: Saved at checkpoint_freq step intervals if save_model_flag=True
    3. Final model: Saved at the end of training if save_model_flag=True

    All models are saved in : {save_dir}/{save_filename}/


    Args:
        eval_env: Action-masked evaluation environment.
        eval_freq (int): How often (in environment steps) to run evaluation.
        n_eval_episodes (int): Number of episodes per evaluation run.
        save_model_flag (bool): Whether to save the model.
        save_dir (str): Directory in which to save model checkpoints.
        save_filename (str): Base filename for the checkpoints (also used for subfolder name).
        max_reward_threshold (float | None): Used during evaluation to
                 compute the percentage of episodes that achieve near perfect episode for given yard.
        checkpoint_freq (int): Frequency (in timesteps) for saving model checkpoints.
        run: Active W&B run object, or None to skip W&B logging.

    Returns:
        CallbackList: List of sb3 callbacks.
    """
    import os
    
    # Create subfolder for all model saves
    base_model_dir = os.path.join(save_dir, save_filename)
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Core evaluation callback: tracks best reward and saves best checkpoint.
    eval_callback = MaskedEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        save_best_model=save_model_flag,
        save_dir=base_model_dir,
        save_filename="best_model",
        max_reward_threshold=max_reward_threshold,
    )

    callback_list = [eval_callback]
    
    # Add periodic checkpoint callback for periodically saving model regardless of evaluation performance.
    if save_model_flag:
        checkpoint_callback = SaveModelCallback(
            save_freq=checkpoint_freq,
            save_dir=save_dir,
            save_filename=save_filename,
            verbose=1,
        )
        callback_list.append(checkpoint_callback)
    
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
    checkpoint_freq: int = 1_000_000,
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
    sequential_hppo: bool = False,
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
        ent_coef (float | None): Entropy loss coefficient in the PPO objective.
        batch_size (int): Batch size for PPO updates.
        n_steps_total (int): Total rollout steps per update (before scaling for parallel envs).
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): GAE lambda parameter for advantage estimation.
        lr_decay (bool): Whether to use linear learning rate decay.
        run: Active W&B run object, or None to skip W&B logging.
        hierarchical (bool): Use HierarchicalLowLevelEnv wrapper for serial hierarchical training.
        high_level_policy_type (str): Fixed Policy for the high-level agent
            ("rule_based_grouped", "rule_based", "random").  Only used when hierarchical=True for
            sequential training.
        hierarchical_high_level (bool), low_level_agent_type (str), low_level_model_path (str | None): Used for sequential hierarchical training.
        joint_hierarchical (bool): Used for joint hierarchical policy for bay-level and stack-level selection.
        hierarchical_diffobs (bool): Used for differentiated observation hierarchical policy (currently not working).
        sequential_hppo (bool): Train separate bay and row actors with sequential (HAPPO-style) PPO updates.
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
        sequential_hppo=sequential_hppo,
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
        checkpoint_freq=checkpoint_freq,
        run=run,
    )

    # Train
    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Save final model after training completes
    if save_model_flag:
        import os
        base_model_dir = os.path.join(save_dir, save_filename)
        os.makedirs(base_model_dir, exist_ok=True)
        final_path = os.path.join(base_model_dir, "final_model")
        model.save(final_path)
        print(f"\nFinal model saved to {final_path}")

    # Close envs
    train_env.close()
    eval_env.close()

    return
