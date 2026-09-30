"""
Deterministic evaluation: episode i is StackEnv's episode reset(seed=
base_seed + i), whatever the batch size, and evaluating during training
does not change training.
"""

import numpy as np
import pytest
import torch

from stack.envs.stack_gym import StackEnv
from stack.evaluation.evaluate import evaluate_policy
from stack.training.bay_row_layout import BayRowLayout
from stack.training.rollout_buffer import JointRolloutBuffer

from .common import algorithm, environment, training_system


def _evaluate(bay_actor, row_actor, n_episodes, eval_batch_size, base_seed=1000):
    episodes, _ = evaluate_policy(environment_config=environment(), bay_actor=bay_actor, row_actor=row_actor,
                                  n_episodes=n_episodes, base_seed=base_seed, eval_batch_size=eval_batch_size,
                                  verbose=False)
    return episodes


@pytest.fixture
def actors():
    layout = BayRowLayout.from_env(StackEnv(config=environment()))
    torch.manual_seed(0)
    return layout.build_actors(embed_dim=16, n_heads=2, n_layers=1, dropout=0.0)


def test_episode_i_is_the_stack_env_episode_of_seed_base_plus_i(actors) -> None:
    """The contract plots.py relies on to compare agents on the same episodes."""

    # 5 episodes in rounds of 2: two full rounds and a padded partial one.
    batched = _evaluate(*actors, n_episodes=5, eval_batch_size=2, base_seed=50)
    one_at_a_time = _evaluate(*actors, n_episodes=5, eval_batch_size=1, base_seed=50)
    assert [e.to_dict() for e in batched] == [e.to_dict() for e in one_at_a_time]

    for i, episode in enumerate(batched):
        env = StackEnv(config=environment())
        env.reset(seed=50 + i)
        rewards = [env.step(action)[1] for action in episode.global_actions]
        np.testing.assert_allclose(rewards, episode.step_rewards, rtol=1e-6)


def test_periodic_evaluation_does_not_change_training() -> None:
    config = algorithm(buffer_size=24, batch_size=8, n_epochs=2, dropout=0.1)

    def train(evaluate: bool):
        env, trainer = training_system(environment(11), config)

        def step_callback(_step):
            if evaluate:
                _evaluate(trainer.bay_actor, trainer.row_actor, n_episodes=2, eval_batch_size=2)

        for _ in range(2):
            stats = trainer.train_iteration(env, JointRolloutBuffer(trainer.layout, 24, 2, "cpu"), step_callback)
        env.close()
        return repr(stats), trainer.row_actor.state_dict()

    (stats_without, params_without), (stats_with, params_with) = train(False), train(True)
    assert stats_with == stats_without  # repr: NaN-safe exact comparison
    assert all(torch.equal(params_with[k], params_without[k]) for k in params_without)
