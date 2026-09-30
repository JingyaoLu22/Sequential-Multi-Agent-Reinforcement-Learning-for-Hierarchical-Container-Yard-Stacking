"""
DummyVecEnv (--vec_backend sync) and SubprocVecEnv (--vec_backend subproc)
give the trainer identical environment semantics.
"""

import numpy as np
import torch
from sb3_contrib.common.maskable.utils import get_action_masks

from stack.training.rollout_buffer import JointRolloutBuffer

from .common import algorithm, environment, training_system

ALGORITHM = algorithm(buffer_size=48, batch_size=12, n_epochs=1)
NUM_ENVS = 3


def _system(backend: str):
    return training_system(environment(5), ALGORITHM, num_envs=NUM_ENVS, vec_backend=backend)


def test_both_backends_step_identically_across_episode_boundaries() -> None:
    envs = {backend: _system(backend)[0] for backend in ("sync", "subproc")}
    rng = np.random.default_rng(0)
    try:
        sync, subproc = envs.values()
        np.testing.assert_array_equal(sync.reset(), subproc.reset())
        episodes_ended = 0

        for _step in range(40):
            masks = get_action_masks(sync)
            np.testing.assert_array_equal(masks, get_action_masks(subproc))
            actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])

            (obs_a, rewards_a, dones_a, infos_a), (obs_b, rewards_b, dones_b, infos_b) = (
                sync.step(actions), subproc.step(actions))
            np.testing.assert_array_equal(obs_a, obs_b)  # after a done: the auto-reset observation
            np.testing.assert_allclose(rewards_a, rewards_b)
            np.testing.assert_array_equal(dones_a, dones_b)
            assert [i.get("TimeLimit.truncated") for i in infos_a] == [i.get("TimeLimit.truncated") for i in infos_b]
            episodes_ended += int(dones_a.sum())

        assert episodes_ended > 0, "the test never crossed an episode boundary"
    finally:
        for env in envs.values():
            env.close()


def test_training_is_independent_of_the_backend() -> None:
    runs = []
    for backend in ("sync", "subproc"):
        env, trainer = _system(backend)
        for _ in range(2):
            buffer = JointRolloutBuffer(trainer.layout, ALGORITHM.buffer_size, NUM_ENVS, "cpu")
            stats = trainer.train_iteration(env, buffer)
        env.close()
        runs.append((stats, buffer, trainer.row_actor.state_dict()))

    (stats_a, buffer_a, params_a), (stats_b, buffer_b, params_b) = runs
    assert repr(stats_a) == repr(stats_b)  # repr: NaN-safe exact comparison
    for field in ("global_states", "bay_actions", "row_actions", "rewards", "dones", "values", "advantages"):
        assert torch.equal(getattr(buffer_a, field), getattr(buffer_b, field)), field
    assert all(torch.equal(params_a[k], params_b[k]) for k in params_a)
