"""
Phase 3 equivalence tests for the precomputed Bay sequence-correction
ratio.

See docs/plan §7:

H. Bay correction equivalence - precomputed correction equals
   minibatch recomputation.
L. Agent update isolation - updating Agent B does not change Agent R;
   updating Agent R does not change Agent B; critic update does not
   change either actor. Also used here as a regression guard that
   Phase 3's changes did not reintroduce a CPU<->GPU round trip into
   the (already good) PPO update phase.
"""

import copy
from dataclasses import replace

import torch

from stack.configs.hierarchical_config import DEFAULT_HIERARCHICAL_CONFIG
from stack.run_sequential_hppo import build_training_system, create_rollout_buffer
from stack.training.advantage import compute_and_store_gae


def _small_environment_config(seed: int) -> dict:
    return {
        "vessel_shape": (3, 3, 2),
        "yard_shape": (3, 3, 2),
        "num_containers": 12,
        "group_num": 3,
        "group_placement": "random",
        "seed": seed,
        "observation_type": "stack_features_v3",
        "reward_norm": True,
        "reward_clip": True,
        "stack_fill_penalty": True,
        "container_sizes": False,
        "enable_imo": False,
    }


def _small_algorithm_config(buffer_size: int, n_epochs: int):
    return replace(
        DEFAULT_HIERARCHICAL_CONFIG,
        embed_dim=16,
        n_heads=2,
        n_layers=1,
        vf_dim=16,
        dropout=0.0,
        buffer_size=buffer_size,
        batch_size=buffer_size // 4,
        n_epochs=n_epochs,
    )


def _build(num_envs: int, buffer_size: int, n_epochs: int, seed: int = 0):
    environment_config = _small_environment_config(seed)
    algorithm_config = _small_algorithm_config(buffer_size, n_epochs)
    device = torch.device("cpu")

    env, agent_b, agent_r, critic, trainer = build_training_system(
        environment_config, algorithm_config, device=device, num_envs=num_envs
    )
    buffer = create_rollout_buffer(env, buffer_size=buffer_size, device=device)
    return env, trainer, buffer


def _filled_and_updated_buffer(env, trainer, buffer):
    """Run one rollout + GAE + critic + Agent B update, matching
    train_iteration()'s state right before precompute_bay_correction()
    would run."""

    trainer.collect_rollout(env, buffer)
    compute_and_store_gae(
        buffer=buffer, gamma=trainer.gamma, gae_lambda=trainer.gae_lambda, num_envs=trainer.num_envs
    )
    trainer.update_critic(buffer)
    trainer.update_bay_actor(buffer)


def test_precomputed_ratio_matches_live_minibatch_recomputation() -> None:
    num_envs = 2
    buffer_size = 16
    n_epochs = 3
    env, trainer, buffer = _build(num_envs=num_envs, buffer_size=buffer_size, n_epochs=n_epochs, seed=2)

    _filled_and_updated_buffer(env, trainer, buffer)

    # Precompute once, in buffer order.
    trainer.precompute_bay_correction(buffer)
    full_batch = next(buffer.get_batches(batch_size=buffer.size, shuffle=False))
    precomputed_ratio_in_order = buffer._bay_correction_cache.clone()

    # Live recomputation directly on the same full, unshuffled batch -
    # the ground truth _compute_bay_sequence_ratio() call every
    # minibatch used to make.
    live_ratio_in_order = trainer._compute_bay_sequence_ratio(full_batch)

    torch.testing.assert_close(
        precomputed_ratio_in_order, live_ratio_in_order, atol=0.0, rtol=0.0
    )

    # Now replay update_row_actor()'s actual shuffled-minibatch pattern
    # across multiple epochs, and for every minibatch compare the
    # precomputed-and-gathered ratio (what get_row_batches() now hands
    # update_row_actor()) against a live per-minibatch recomputation
    # (what the old code path used to do), for every sample regardless
    # of which minibatch/epoch it lands in.
    #
    # Tolerance, not bit-exactness, here: this recomputes Agent B's
    # forward pass on a DIFFERENT batch shape/composition (a shuffled
    # minibatch of trainer.batch_size samples) than the precomputed
    # pass above (the full, unshuffled buffer). Batched matmul on both
    # CPU and GPU is not perfectly invariant to batch shape at float32
    # precision - summation order inside a kernel can differ - so a
    # ~1e-7 (a few ULP at float32) difference here is expected float32
    # noise, not a logic bug: the old, unoptimized code already
    # recomputed a fresh, slightly different value every epoch purely
    # because shuffling changes each sample's minibatch composition
    # epoch to epoch. Precomputing once (this phase's change) removes
    # that pre-existing epoch-to-epoch jitter rather than introducing
    # new error.
    for _epoch in range(n_epochs):
        for batch, precomputed_ratio in buffer.get_row_batches(
            batch_size=trainer.batch_size, shuffle=True
        ):
            live_ratio = trainer._compute_bay_sequence_ratio(batch)
            torch.testing.assert_close(
                precomputed_ratio, live_ratio, atol=1e-5, rtol=1e-4
            )


def test_update_row_actor_uses_precomputed_ratio_without_extra_bay_forward_pass() -> None:
    """update_row_actor() must not call Agent B's network at all -
    _compute_bay_sequence_ratio() (the only thing that does) should be
    invoked exactly once total, by precompute_bay_correction(), not
    once per minibatch inside update_row_actor()."""

    num_envs = 2
    buffer_size = 16
    n_epochs = 3
    env, trainer, buffer = _build(num_envs=num_envs, buffer_size=buffer_size, n_epochs=n_epochs, seed=3)

    _filled_and_updated_buffer(env, trainer, buffer)

    call_count = 0
    original = trainer._compute_bay_sequence_ratio

    def counting_wrapper(batch):
        nonlocal call_count
        call_count += 1
        return original(batch)

    trainer._compute_bay_sequence_ratio = counting_wrapper
    try:
        trainer.precompute_bay_correction(buffer)
        trainer.update_row_actor(buffer)
    finally:
        trainer._compute_bay_sequence_ratio = original

    assert call_count == 1


def test_agent_update_isolation_and_no_cpu_gpu_roundtrip() -> None:
    """L: updating one actor/critic does not perturb the others'
    parameters, and none of the three update_* methods perform a
    .cpu()/.numpy() call (a regression guard for the PPO-update phase
    Phase 3 touched)."""

    num_envs = 2
    buffer_size = 16
    env, trainer, buffer = _build(num_envs=num_envs, buffer_size=buffer_size, n_epochs=2, seed=4)

    trainer.collect_rollout(env, buffer)
    compute_and_store_gae(
        buffer=buffer, gamma=trainer.gamma, gae_lambda=trainer.gae_lambda, num_envs=trainer.num_envs
    )

    agent_b_before = copy.deepcopy(trainer.agent_b.state_dict())
    agent_r_before = copy.deepcopy(trainer.agent_r.state_dict())
    critic_before = copy.deepcopy(trainer.critic.state_dict())

    def assert_unchanged(state_dict, reference, label):
        for key, value in reference.items():
            torch.testing.assert_close(
                state_dict[key], value, msg=f"{label}.{key} changed unexpectedly"
            )

    def assert_changed(state_dict, reference, label):
        assert any(
            not torch.equal(state_dict[key], value) for key, value in reference.items()
        ), f"{label} did not change after its own update"

    original_cpu = torch.Tensor.cpu
    original_numpy = torch.Tensor.numpy
    offending_calls = []

    def tracking_cpu(self, *args, **kwargs):
        offending_calls.append("cpu")
        return original_cpu(self, *args, **kwargs)

    def tracking_numpy(self, *args, **kwargs):
        offending_calls.append("numpy")
        return original_numpy(self, *args, **kwargs)

    torch.Tensor.cpu = tracking_cpu
    torch.Tensor.numpy = tracking_numpy
    try:
        trainer.update_critic(buffer)
        assert_changed(trainer.critic.state_dict(), critic_before, "critic")
        assert_unchanged(trainer.agent_b.state_dict(), agent_b_before, "agent_b")
        assert_unchanged(trainer.agent_r.state_dict(), agent_r_before, "agent_r")

        agent_b_before = copy.deepcopy(trainer.agent_b.state_dict())
        trainer.update_bay_actor(buffer)
        assert_changed(trainer.agent_b.state_dict(), agent_b_before, "agent_b")
        assert_unchanged(trainer.agent_r.state_dict(), agent_r_before, "agent_r")

        trainer.precompute_bay_correction(buffer)

        agent_r_before = copy.deepcopy(trainer.agent_r.state_dict())
        agent_b_before = copy.deepcopy(trainer.agent_b.state_dict())
        trainer.update_row_actor(buffer)
        assert_changed(trainer.agent_r.state_dict(), agent_r_before, "agent_r")
        assert_unchanged(trainer.agent_b.state_dict(), agent_b_before, "agent_b")
    finally:
        torch.Tensor.cpu = original_cpu
        torch.Tensor.numpy = original_numpy

    assert offending_calls == [], (
        "update_critic/update_bay_actor/precompute_bay_correction/"
        f"update_row_actor performed unexpected CPU<->GPU calls: {offending_calls}"
    )
