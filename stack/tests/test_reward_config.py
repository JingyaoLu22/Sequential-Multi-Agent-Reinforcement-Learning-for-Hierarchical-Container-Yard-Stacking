"""
Reward settings come from the shared set_config(); only the explicit
--[no-]reward_norm / --[no-]reward_clip flags change them, and they only
shape the training reward: every evaluation reports raw rewards.
"""

import torch

from stack.configs.environments import set_config
from stack.envs.stack_gym import StackEnv
from stack.evaluation.evaluate import evaluate_policy, make_evaluation_config
from stack.run_sequential_hppo import build_parser
from stack.training.bay_row_layout import BayRowLayout

from .common import environment


def test_reward_settings_change_only_through_explicit_flags() -> None:
    def config(*flags):
        args = build_parser().parse_args(["--size", "small_with_margin", *flags])
        return set_config(args.size, 1, args.reward_norm, args.reward_clip)

    # Without flags: exactly set_config(), i.e. what the baselines train on.
    assert config() == set_config("small_with_margin", 1)
    assert (config()["reward_norm"], config()["reward_clip"]) == (False, False)
    assert (config("--reward_norm")["reward_norm"], config("--reward_norm")["reward_clip"]) == (True, False)


def test_evaluation_always_uses_raw_rewards() -> None:
    training_config = environment(reward_norm=True, reward_clip=True)
    assert make_evaluation_config(training_config) == environment(reward_norm=False, reward_clip=False)
    assert training_config["reward_norm"] and training_config["reward_clip"]  # never modified

    # evaluate_policy reports raw rewards even when handed the training config.
    torch.manual_seed(0)
    actors = BayRowLayout.from_env(StackEnv(config=training_config)).build_actors(
        embed_dim=16, n_heads=2, n_layers=1, dropout=0.0)
    totals = [[e.total_reward for e in evaluate_policy(environment_config=config, bay_actor=actors[0],
                                                       row_actor=actors[1], n_episodes=4, verbose=False)[0]]
              for config in (training_config, environment(reward_norm=False, reward_clip=False))]
    assert totals[0] == totals[1]
