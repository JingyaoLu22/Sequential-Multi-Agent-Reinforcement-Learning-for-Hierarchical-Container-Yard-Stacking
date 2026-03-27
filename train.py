from sb3_contrib.ppo_mask import MaskablePPO
from envs.stack_gym import StackEnv
from sb3_contrib.common.wrappers import ActionMasker
from utils import MaskedEvalCallback, mask_fn, create_parallel_envs, save_model
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from models.transformer_policy import MaskableTransformerPolicy


def create_env(config, seed, render_mode, parallel, n_parallel_envs):
    eval_config = config.copy()

    eval_config['reward_norm'] = False  # Disable reward normalization for evaluation
    eval_config['reward_clip'] = False   # Disable reward clipping for evaluation
    
    if parallel:
        train_env = create_parallel_envs(config, n_envs=n_parallel_envs, vec_env_cls=SubprocVecEnv)
        eval_env = StackEnv(config=eval_config, render_mode=None)
        eval_env = ActionMasker(eval_env, mask_fn)
    else:
        train_env = StackEnv(config=config, render_mode=render_mode)

        eval_env = StackEnv(config=eval_config, render_mode=render_mode)
        eval_env = ActionMasker(eval_env, mask_fn)
    return train_env, eval_env

def create_model(train_env, eval_env, config, device, eval_freq, n_eval_episodes, parallel, n_parallel_envs,
                 use_transformer=False, embed_dim=128, n_heads=4, n_layers=2, vf_dim=128,
                 tanh_clipping=10.0, n_epochs=10, lr=3e-4, vf_coef=0.5, run=None):

    if parallel:
        n_steps = 2048 // n_parallel_envs
    else:
        n_steps = 2048

    if use_transformer:
        policy = MaskableTransformerPolicy
        policy_kwargs = dict(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            vf_dim=vf_dim,
            tanh_clipping=tanh_clipping,
        )
    else:
        policy = "MlpPolicy"
        policy_kwargs = dict(net_arch=[256, 256, 32])

    model = MaskablePPO(
        policy=policy,
        env=train_env,
        learning_rate=lr,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.3,
        vf_coef=vf_coef,
        clip_range=0.2,
        verbose=1,
        device=device,
        tensorboard_log=f"runs/{run.id}" if run is not None else None,
    )

    return model

def create_callbacks(eval_env, eval_freq, n_eval_episodes, save_model_flag, save_dir, save_filename,
                     max_reward_threshold=None, run=None):

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
        callback_list.append(WandbCallback(verbose=0))

    callbacks = CallbackList(callback_list)
    
    return callbacks

def train(config, device, render_mode, parallel, n_parallel_envs, eval_freq, n_eval_episodes,
           timesteps, save_model_flag, save_dir, save_filename, max_reward_threshold=None,
           use_transformer=False, embed_dim=128, n_heads=4, n_layers=2, vf_dim=128,
           tanh_clipping=10.0, n_epochs=10, lr=3e-4, vf_coef=0.5, run=None):
    # Create environments
    seed = config.get("seed", None)
    train_env, eval_env = create_env(config, seed=seed, render_mode=render_mode, parallel=parallel, n_parallel_envs=n_parallel_envs)

    # Create model and evaluation callback
    model = create_model(train_env, eval_env, config, device, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                         parallel=parallel, n_parallel_envs=n_parallel_envs,
                         use_transformer=use_transformer, embed_dim=embed_dim, n_heads=n_heads,
                         n_layers=n_layers, vf_dim=vf_dim, tanh_clipping=tanh_clipping, n_epochs=n_epochs, lr=lr, vf_coef=vf_coef, run=run)
    callbacks = create_callbacks(eval_env, eval_freq, n_eval_episodes, save_model_flag, save_dir, save_filename,
                                  max_reward_threshold=max_reward_threshold, run=run)

    # TRAIN
    model.learn(total_timesteps=timesteps, callback=callbacks)

    # SAVE MODEL
    if save_model_flag:
        save_model(model, save_dir, save_filename)

    train_env.close()
    eval_env.close()

    return
