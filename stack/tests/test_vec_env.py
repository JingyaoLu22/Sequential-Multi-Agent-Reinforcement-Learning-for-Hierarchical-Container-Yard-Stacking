"""
DummyVecEnv (--vec_backend sync) and SubprocVecEnv (--vec_backend subproc)
give the trainer identical environment semantics.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from sb3_contrib.common.maskable.utils import get_action_masks

from stack.configs.environments import set_config
from stack.configs.hierarchical_config import get_hierarchical_config
from stack.run_sequential_hppo import build_training_system
from stack.training.rollout_buffer import JointRolloutBuffer

ENVIRONMENT = dict(set_config("small_with_margin", seed=5), vessel_shape=(3, 3, 2), yard_shape=(3, 3, 2),
                   num_containers=12)
ALGORITHM = replace(get_hierarchical_config("small"), embed_dim=16, n_heads=2, n_layers=1, vf_dim=16,
                    dropout=0.0, buffer_size=48, batch_size=12, n_epochs=1)
NUM_ENVS = 3


def _system(backend: str):
    torch.manual_seed(0)
    return build_training_system(ENVIRONMENT, ALGORITHM, torch.device("cpu"), num_envs=NUM_ENVS,
                                 vec_backend=backend)


def test_both_backends_step_identically_across_episode_boundaries() -> None:
    envs = {backend: _system(backend)[0] for backend in ("sync", "subproc")}
    rng = np.random.default_rng(0)
    try:
        observations = {backend: env.reset() for backend, env in envs.items()}
        np.testing.assert_array_equal(observations["sync"], observations["subproc"])
        episodes_ended = 0

        for _step in range(40):
            masks = {backend: get_action_masks(env) for backend, env in envs.items()}
            np.testing.assert_array_equal(masks["sync"], masks["subproc"])

            actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in masks["sync"]])
            results = {backend: env.step(actions) for backend, env in envs.items()}

            (obs_a, rewards_a, dones_a, infos_a), (obs_b, rewards_b, dones_b, infos_b) = results.values()
            np.testing.assert_array_equal(obs_a, obs_b)  # after a done: the auto-reset observation
            np.testing.assert_allclose(rewards_a, rewards_b)
            np.testing.assert_array_equal(dones_a, dones_b)
            assert [i.get("TimeLimit.truncated") for i in infos_a] == [i.get("TimeLimit.truncated") for i in infos_b]
            episodes_ended += int(dones_a.sum())

        assert episodes_ended > 0, "the test never crossed an episode boundary"
    finally:
        for env in envs.values():
            env.close()


@pytest.mark.parametrize("num_iterations", [2])
def test_training_is_independent_of_the_backend(num_iterations: int) -> None:
    runs = {}
    for backend in ("sync", "subproc"):
        env, _bay, _row, _critic, trainer = _system(backend)
        stats = []
        for _ in range(num_iterations):
            buffer = JointRolloutBuffer(trainer.layout, ALGORITHM.buffer_size, NUM_ENVS, "cpu")
            iteration = trainer.train_iteration(env, buffer)
            stats.append({k: v for k, v in iteration.items() if not k.endswith("_seconds")})
        env.close()
        runs[backend] = (stats, buffer, trainer.row_actor.state_dict())

    (stats_a, buffer_a, params_a), (stats_b, buffer_b, params_b) = runs.values()
    assert repr(stats_a) == repr(stats_b)  # repr: NaN-safe exact comparison
    for field in ("global_states", "bay_actions", "row_actions", "rewards", "dones", "values", "advantages"):
        assert torch.equal(getattr(buffer_a, field), getattr(buffer_b, field)), field
    assert all(torch.equal(params_a[k], params_b[k]) for k in params_a)
