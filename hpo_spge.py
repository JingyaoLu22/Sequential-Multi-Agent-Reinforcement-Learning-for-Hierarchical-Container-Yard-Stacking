"""
This script is used to perform hyperparameter optimization (HPO) for reinforcement learning algorithms using Optuna.
It allows for the tuning of various hyperparameters for different algorithms such as PPO, A2C, TRPO, DQN, and QRDQN.
To use this script, you need to provide a configuration file.
e.g. `python hpo_spge.py --config spge.sb3hpo.yaml --env_name SPGE --algo ppo`
"""
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
import gymnasium as gym
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from algorithms.A2C import MaskableA2C
from algorithms.DQN import MaskableDQN
from algorithms.QRDQN import MaskableQRDQN
from algorithms.TRPO import MaskableTRPO
from envs.stowage_gym import StowageEnv
from envs.stowage_aec import StowageAEC
from envs.stowage_crane_gym import MultiCraneStowageEnv
import os
import yaml
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}

NAME_TO_ALGO = {
    "ppo": MaskablePPO,
    "a2c": MaskableA2C,
    "trpo": MaskableTRPO,
    "dqn": MaskableDQN,
    "qrdqn": MaskableQRDQN,
}



def sample_ac_params(trial: optuna.Trial) -> Dict[str, Any]:
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    net_arch = {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)
    return {
        "gae_lambda": gae_lambda,
        "n_steps": n_steps,
        "policy_kwargs": {
            "net_arch": net_arch,
        },
    }


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    kwargs = sample_ac_params(trial)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    kwargs["policy_kwargs"].update({"ortho_init": ortho_init})
    kwargs.update(
        {   
            "gae_lambda": 1,
            "n_steps": 5,
            "learning_rate": 7e-4,
        }
    )
    trial.set_user_attr("gae_lambda_", 1) # A2C does not use gae_lambda
    trial.set_user_attr("n_steps", 5)
    trial.set_user_attr("learning_rate", 7e-4)
    return kwargs

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    kwargs = sample_ac_params(trial)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)

    kwargs.update(
        {
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
        }
    )
    return kwargs


def sample_trpo_params(trial: optuna.Trial) -> Dict[str, Any]:
    kwargs = sample_ac_params(trial)
    cg_damping = trial.suggest_float("cg_damping", 0.001, 0.1, log=True)
    target_kl = trial.suggest_float("target_kl", 0.005, 0.01, log=True)
    kwargs.update(
        {
            "cg_damping": cg_damping,
            "target_kl": target_kl,
        }
    )
    return kwargs


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for DQN hyperparameters."""
    learning_starts = trial.suggest_int("learning_starts", 20, 200)
    update_type = trial.suggest_categorical("update_type", ["soft", "hard"])
    if update_type == "hard":
        tau = 1.0
        target_update_interval = trial.suggest_int("target_update_interval", 100, 10000, log=True)
    else:
        tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
        target_update_interval = 1
    trial.set_user_attr("tau", tau)
    trial.set_user_attr("target_update_interval", target_update_interval)

    return {
        "tau": tau,
        "learning_starts": learning_starts,
        "target_update_interval": target_update_interval,
    }


def sample_qrdqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for QRDQN hyperparameters."""
    kwargs = sample_dqn_params(trial)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.0001, 0.1, log=True)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.0001, 0.05, log=True)
    kwargs.update(
        {
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
        }
    )
    return kwargs


NAME_TO_HYPERPARAMS = {
    "ppo": sample_ppo_params,
    "a2c": sample_a2c_params,
    "trpo": sample_trpo_params,
    "dqn": sample_dqn_params,
    "qrdqn": sample_qrdqn_params,
}
NAME_TO_ENV = {
    "SPGE" : StowageEnv,
    "SPAEC" : StowageAEC,
    "SPGE-MC" : MultiCraneStowageEnv,
}


class TrialEvalCallback(MaskableEvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample global hyperparameters.
    gamma = trial.suggest_float("gamma", 0.2, 0.9999, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]
    policy_kwargs = {"activation_fn": activation_fn}
    kwargs.update(
        {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "policy_kwargs": policy_kwargs,
        }
    )

    # Sample algorithm specific hyperparameters.
    algo_params = algo_hyperparams(trial)
    if "policy_kwargs" in algo_params:
        kwargs["policy_kwargs"].update(algo_params["policy_kwargs"])
        del algo_params["policy_kwargs"]
    kwargs.update(algo_params)

    # Create the RL model.
    train_env = NAME_TO_ENV[env_name](config = config["environment"])
    model = algorithm(env=train_env, **kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(NAME_TO_ENV[env_name](config["environment"]))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

def load_config(config_name: str) -> dict:
    """
    Load a configuration file from the train_configs directory. The yaml file should contain all the parameters relevant to training

    Args:
        config_name (str): The name of the configuration file to load.

    Returns:
        dict: The loaded configuration dictionary.
    """
    path_to_config = os.path.join(CONFIGS_DIR, config_name)
    with open(path_to_config, "r") as file:
        config = yaml.safe_load(file)
    config ['environment']['vessel_shape'] = tuple(config['environment']['vessel_shape'])
    config ['environment']['yard_shape'] = tuple(config['environment']['yard_shape'])
    return config


def get_argparser():
    """Define all the Argparse arguments for the training script."""
    parser = argparse.ArgumentParser(description="Training arguments for PPO with configurable YAML setup.")

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="train.sb3hpo.yaml",
        help="Path to the configuration YAML file. Example: train.default.yaml (placed in StowAI/train_configs/).",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="environment name to be used in the training. Example: SPGE.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="Algorithm to be used in the training. Example: ppo.",
    )
    parser.add_argument(
        "--storage_file",
        type=str,
        help="Storage file for Optuna. Example: ./hpo_studies/study.log",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes for the study. Example: 'First trial with PPO'.",
    )


    return parser


if __name__ == "__main__":
    """ Main function for running training.
    The script depends on the yaml configs in the train_configs directory. All arguments can be overridden through the command line.
    via the argparse module. Only run_name uses a default value and is set only via the argparser.
    """
    np.seterr(all="raise")  # Raise all numpy errors

    parser = get_argparser()
    args = parser.parse_args()
    config_str = args.config
    config = load_config(config_str)
    env_name = args.env_name if args.env_name else config["general"]['env_name']
    config["general"]["algorithm"] = args.algo if args.algo else config["general"]["algorithm"]
    notes = args.notes if args.notes else config["general"]["notes"]

    # Load the environment
    environment_config = config["environment"]
    environment = NAME_TO_ENV[env_name]


    # Load HPO related parameters
    N_EVAL_EPISODES = config["general"]["n_eval_episodes"]
    N_TIMESTEPS = config["general"]["n_timesteps"]
    N_STARTUP_TRIALS = config["general"]["n_startup_trials"]
    N_EVALUATIONS = config["general"]["n_evaluations"]
    N_TRIALS = config["general"]["n_trials"]
    EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
    TIMEOUT = config["general"]["timeout"]
    storage_file = config["general"]["storage_file"]

    # Load the algorithm
    algorithm = NAME_TO_ALGO[config["general"]["algorithm"]]
    algo_hyperparams = NAME_TO_HYPERPARAMS[config["general"]["algorithm"]]
    print("Algorithm: ", config["general"]["algorithm"], "Scenario: ", env_name)

    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)
    study_name = "{}-{}-{}".format(config["general"]["algorithm"], env_name, notes)

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(storage_file),
    )
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    try:
        with np.errstate(under="ignore"):  # Ignore underflow warnings
            study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial for {}:".format(config["general"]["algorithm"]))
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
