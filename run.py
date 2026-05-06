import argparse

import wandb

from train import train
from utils import set_config, get_device


def main(args):

    # Get environment config (based on args.size which is small, medium, or large)
    config = set_config(size=args.size, seed=args.seed)

    # Get device (GPU or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Initialize wandb run (unless disabled)
    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            tags=args.wandb_tags or None,
            config={**vars(args), **config},
            sync_tensorboard=True,
        )

    # Train the model
    train(
        config=config,
        device=device,
        render_mode=args.render_mode,
        parallel=args.parallel,
        n_parallel_envs=args.n_parallel_envs,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        timesteps=args.timesteps,
        save_model_flag=args.save_model,
        save_dir=args.save_dir,
        save_filename=args.save_filename,
        max_reward_threshold=args.max_reward_threshold,
        checkpoint_freq=args.checkpoint_freq,
        use_transformer=args.use_transformer,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vf_dim=args.vf_dim,
        tanh_clipping=args.tanh_clipping,
        n_epochs=args.n_epochs,
        lr=args.lr,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_steps_total=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr_decay=args.lr_decay,
        run=run,
        hierarchical=args.hierarchical,
        high_level_policy_type=args.high_level_policy,
        hierarchical_high_level=args.hierarchical_high_level,
        low_level_agent_type=args.low_level_agent,
        low_level_model_path=args.low_level_model_path,
        joint_hierarchical=args.joint_hierarchical,
        hierarchical_diffobs=args.hierarchical_diffobs,
    )

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on StackEnv with various configurations."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "small_with_margin", "medium", "medium_with_margin", "large", "large_with_margin", "large_v2", "large_v2_with_margin", "large_v3", "large_v3_with_margin", "large_v4", "large_v4_with_margin"],
        help="Size of the environment configuration.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--render_mode", type=str, default=None, help="Render mode for the environment."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Whether to use parallel environments for training.",
    )
    parser.add_argument(
        "--n_parallel_envs",
        type=int,
        default=4,
        help="Number of parallel environments to use if --parallel is set.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=25_000,
        help="Frequency (in timesteps) for evaluation during training.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=30,
        help="Number of episodes to evaluate at each evaluation step.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_500_000,
        help="Total number of training timesteps.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Whether to save the trained model after training.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Directory to save the trained model if --save_model is set.",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default="ppo_stack_env_model",
        help="Filename for the saved model (without extension).",
    )
    parser.add_argument(
        "--max_reward_threshold",
        type=float,
        default=None,
        help="Reward threshold for computing percent of episodes achieving max reward during evaluation.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1_000_000,
        help="Frequency (in timesteps) for saving model checkpoints (e.g., 1000000 = every 1M steps).",
    )
    # wandb args
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb logging."
    )
    parser.add_argument(
        "--wandb_project", type=str, default="stack-rl", help="wandb project name."
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity (username or team). Defaults to your wandb login default.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional display name for the wandb run.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="Optional tags for the wandb run (space-separated).",
    )
    # transformer args
    parser.add_argument(
        "--use_transformer",
        action="store_true",
        help="Use transformer attention policy instead of MLP.",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Transformer embedding dimension."
    )
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads."
    )
    parser.add_argument(
        "--n_layers", type=int, default=2, help="Number of transformer encoder layers."
    )
    parser.add_argument(
        "--vf_dim", type=int, default=128, help="Hidden dim for the critic MLP head."
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="Tanh clipping for pointer logits (0 disables).",
    )
    # training args
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to optimize policy at each update.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.5,
        help="Value function coefficient for the loss.",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=None,
        help="Entropy coefficient for the loss. Defaults to 0.15 for joint_hierarchical, 0.3 otherwise.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size for PPO gradient updates.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Total number of rollout transitions per update (divided across parallel envs).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards.",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation.",
    )
    parser.add_argument(
        "--lr_decay",
        action="store_true",
        help="Use linear learning rate decay to 0 over training.",
    )
    # hierarchical args
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Use hierarchical mode: fixed high-level agent selects bay, RL learns low-level stack selection.",
    )
    parser.add_argument(
        "--high_level_policy",
        type=str,
        default="rule_based_grouped",
        choices=["rule_based_grouped", "rule_based", "random"],
        help="Policy for the high-level bay-selection agent (only used with --hierarchical).",
    )
    # hierarchical high-level training args
    parser.add_argument(
        "--hierarchical_high_level",
        action="store_true",
        help="Train the high-level bay-selection agent with a fixed low-level agent.",
    )
    parser.add_argument(
        "--low_level_agent",
        type=str,
        default="rule_based_grouped",
        choices=["rule_based_grouped", "rule_based", "random", "trained_model"],
        help="Policy for the fixed low-level agent (only used with --hierarchical_high_level).",
    )
    parser.add_argument(
        "--low_level_model_path",
        type=str,
        default=None,
        help="Path to saved MaskablePPO low-level model (required when --low_level_agent trained_model).",
    )
    # joint hierarchical args
    parser.add_argument(
        "--joint_hierarchical",
        action="store_true",
        help="Train bay and stack selection jointly with an autoregressive policy.",
    )
    parser.add_argument(
        "--hierarchical_diffobs",
        action="store_true",
        help="Train with differentiated observations: bay head sees bay-level features, stack head sees per-bay stack features.",
    )
    args = parser.parse_args()

    main(args)
