"""
Reward settings come from the shared set_config() for every algorithm;
only the explicit --[no-]reward_norm / --[no-]reward_clip flags change them.
"""

import argparse

import pytest

from stack.configs.environments import add_reward_arguments, set_config
from stack.run_sequential_hppo import build_parser


def _parse(*argv):
    parser = argparse.ArgumentParser()
    add_reward_arguments(parser)
    return parser.parse_args(list(argv))


def test_small_with_margin_default_is_unnormalized_and_unclipped() -> None:
    config = set_config("small_with_margin", seed=1)
    assert (config["reward_norm"], config["reward_clip"]) == (False, False)


def test_flags_default_to_the_size_config() -> None:
    args = _parse()
    assert (args.reward_norm, args.reward_clip) == (None, None)
    assert set_config("small_with_margin", 1, args.reward_norm, args.reward_clip) == set_config(
        "small_with_margin", 1
    )


@pytest.mark.parametrize(
    "argv,expected",
    [
        (("--reward_norm", "--reward_clip"), (True, True)),
        (("--no-reward_norm",), (False, None)),
        (("--reward_clip",), (None, True)),
    ],
)
def test_explicit_flags_override_either_way(argv, expected) -> None:
    args = _parse(*argv)
    assert (args.reward_norm, args.reward_clip) == expected

    config = set_config("medium_with_margin", 1, args.reward_norm, args.reward_clip)
    default = set_config("medium_with_margin", 1)
    for name, value in zip(("reward_norm", "reward_clip"), expected):
        assert config[name] == (default[name] if value is None else value)


def test_sequential_hppo_cli_exposes_the_shared_flags() -> None:
    args = build_parser().parse_args(["--size", "small_with_margin", "--reward_norm"])
    assert (args.reward_norm, args.reward_clip) == (True, None)


def test_training_and_standalone_evaluation_share_reward_scale() -> None:
    from stack.evaluation.evaluate import build_parser as build_evaluation_parser
    from stack.evaluation.evaluate import make_evaluation_config

    training_config = set_config("medium_with_margin", seed=1)
    assert training_config["reward_norm"] and training_config["reward_clip"]

    for extra, expected in (([], (False, False)), (["--eval_use_training_rewards"], (True, True))):
        training_args = build_parser().parse_args(["--size", "medium_with_margin", *extra])
        standalone_args = build_evaluation_parser().parse_args(["--model_dir", "unused", *extra])
        assert training_args.eval_use_training_rewards == standalone_args.eval_use_training_rewards

        config = make_evaluation_config(training_config, standalone_args.eval_use_training_rewards)
        assert (config["reward_norm"], config["reward_clip"]) == expected
        # The training run's own config is never modified.
        assert training_config["reward_norm"] and training_config["reward_clip"]
