"""JointRolloutBuffer: (T, N) storage, vectorized GAE, step-major flattening."""

import numpy as np
import torch
from gymnasium import spaces

from stack.training.bay_row_layout import BayRowLayout
from stack.training.rollout_buffer import JointRolloutBuffer


def test_gae_and_flattening_on_a_hand_computed_rollout() -> None:
    layout = BayRowLayout(n_bays=1, n_rows=2, observation_space=spaces.Box(-1, 1, (2,)),
                          row_observation_space=spaces.Box(-1, 1, (2,)))
    buffer = JointRolloutBuffer(layout, buffer_size=6, num_envs=2, device="cpu")

    # Environment 0 never finishes; environment 1's episode ends at t = 1.
    values = np.array([0.0, 0.5])
    dones = [[False, False], [False, True], [False, False]]
    for t in range(3):
        buffer.add_batch(
            global_states=np.full((2, 2), t), bay_action_masks=np.ones((2, 1), bool), bay_actions=np.zeros(2),
            bay_log_probs=np.zeros(2), row_observations=np.zeros((2, 2)), row_action_masks=np.ones((2, 2), bool),
            row_actions=np.zeros(2), row_log_probs=np.zeros(2), rewards=np.ones(2), dones=dones[t],
            values=values, next_values=np.ones(2),
        )

    advantages, returns = buffer.compute_gae(gamma=0.5, gae_lambda=0.5)

    # delta_t = r + 0.5 V(s_{t+1}) (1 - done) - V(s_t);  A_t = delta_t + 0.25 (1 - done) A_{t+1}
    #   env 0: delta = 1.5, 1.5, 1.5  ->  A = 1.96875, 1.875, 1.5
    #   env 1: delta = 1.0, 0.5, 1.0  ->  A = 1.125,   0.5,   1.0
    expected = torch.tensor([[1.96875, 1.125], [1.875, 0.5], [1.5, 1.0]])
    # Flattened step-major: sample t * N + n.
    torch.testing.assert_close(advantages, expected.reshape(-1))
    torch.testing.assert_close(returns, (expected + torch.tensor(values, dtype=torch.float32)).reshape(-1))
    assert buffer.rollout_batch().global_states[:, 0].tolist() == [0, 0, 1, 1, 2, 2]
