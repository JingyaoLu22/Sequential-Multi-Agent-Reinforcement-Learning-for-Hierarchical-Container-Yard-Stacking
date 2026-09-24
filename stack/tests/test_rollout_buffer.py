"""
Phase 4 equivalence test: JointRolloutBuffer.add_batch() vs. the
per-environment add() loop it replaces.

See docs/plan §7, test I ("Batched buffer insertion"):
add_batch() results must match repeated add() calls exactly, including
the storage arrays, buffer position/full state, the partial-write
behaviour when the buffer has less remaining capacity than num_envs,
and the ValueErrors raised on malformed input.
"""

import numpy as np
import pytest
from gymnasium import spaces

from stack.training.rollout_buffer import JointRolloutBuffer


GLOBAL_OBS_DIM = 12
BAY_OBS_DIM = 12
ROW_OBS_DIM = 4
N_BAYS = 3
N_ROWS = 4


def _make_buffer(buffer_size: int) -> JointRolloutBuffer:
    return JointRolloutBuffer(
        buffer_size=buffer_size,
        global_observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(GLOBAL_OBS_DIM,), dtype=np.float32
        ),
        bay_observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(BAY_OBS_DIM,), dtype=np.float32
        ),
        row_observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(ROW_OBS_DIM,), dtype=np.float32
        ),
        n_bays=N_BAYS,
        n_rows=N_ROWS,
        device="cpu",
    )


def _random_transition_batch(num_envs: int, rng: np.random.Generator) -> dict:
    bay_action_masks = np.zeros((num_envs, N_BAYS), dtype=bool)
    row_action_masks = np.zeros((num_envs, N_ROWS), dtype=bool)
    bay_actions = np.zeros(num_envs, dtype=np.int64)
    row_actions = np.zeros(num_envs, dtype=np.int64)

    for i in range(num_envs):
        # Ensure at least one valid bay/row, and that the sampled
        # action always falls inside the valid set, matching what a
        # real rollout (masked sampling) guarantees.
        valid_bays = rng.choice(N_BAYS, size=rng.integers(1, N_BAYS + 1), replace=False)
        bay_action_masks[i, valid_bays] = True
        bay_actions[i] = rng.choice(valid_bays)

        valid_rows = rng.choice(N_ROWS, size=rng.integers(1, N_ROWS + 1), replace=False)
        row_action_masks[i, valid_rows] = True
        row_actions[i] = rng.choice(valid_rows)

    return {
        "global_states": rng.standard_normal((num_envs, GLOBAL_OBS_DIM)).astype(np.float32),
        "bay_observations": rng.standard_normal((num_envs, BAY_OBS_DIM)).astype(np.float32),
        "bay_action_masks": bay_action_masks,
        "bay_actions": bay_actions,
        "bay_log_probs": rng.standard_normal(num_envs).astype(np.float32),
        "row_observations": rng.standard_normal((num_envs, ROW_OBS_DIM)).astype(np.float32),
        "row_action_masks": row_action_masks,
        "row_actions": row_actions,
        "row_log_probs": rng.standard_normal(num_envs).astype(np.float32),
        "rewards": rng.standard_normal(num_envs).astype(np.float32),
        "values": rng.standard_normal(num_envs).astype(np.float32),
        "next_values": rng.standard_normal(num_envs).astype(np.float32),
        "terminated": rng.random(num_envs) < 0.2,
        "truncated": rng.random(num_envs) < 0.1,
    }


def _add_via_loop(buffer: JointRolloutBuffer, batch: dict) -> int:
    num_envs = batch["global_states"].shape[0]
    written = 0
    for i in range(num_envs):
        if buffer.is_full():
            break
        buffer.add(
            global_state=batch["global_states"][i],
            bay_observation=batch["bay_observations"][i],
            bay_action_mask=batch["bay_action_masks"][i],
            bay_action=int(batch["bay_actions"][i]),
            bay_log_prob=float(batch["bay_log_probs"][i]),
            row_observation=batch["row_observations"][i],
            row_action_mask=batch["row_action_masks"][i],
            row_action=int(batch["row_actions"][i]),
            row_log_prob=float(batch["row_log_probs"][i]),
            reward=float(batch["rewards"][i]),
            value=float(batch["values"][i]),
            next_value=float(batch["next_values"][i]),
            terminated=bool(batch["terminated"][i]),
            truncated=bool(batch["truncated"][i]),
        )
        written += 1
    return written


def _assert_buffers_equal(buffer_a: JointRolloutBuffer, buffer_b: JointRolloutBuffer) -> None:
    assert buffer_a.pos == buffer_b.pos
    assert buffer_a.full == buffer_b.full

    for field in (
        "global_states",
        "bay_observations",
        "bay_action_masks",
        "bay_actions",
        "old_bay_log_probs",
        "row_observations",
        "row_action_masks",
        "row_actions",
        "old_row_log_probs",
        "rewards",
        "terminated",
        "truncated",
        "values",
        "next_values",
    ):
        np.testing.assert_array_equal(
            getattr(buffer_a, field), getattr(buffer_b, field), err_msg=f"field={field}"
        )


@pytest.mark.parametrize("num_envs", [1, 4, 16])
def test_add_batch_matches_looped_add_single_round(num_envs: int) -> None:
    rng = np.random.default_rng(0)
    buffer_size = num_envs * 3

    buffer_loop = _make_buffer(buffer_size)
    buffer_batch = _make_buffer(buffer_size)

    batch = _random_transition_batch(num_envs, rng)

    written_loop = _add_via_loop(buffer_loop, batch)
    written_batch = buffer_batch.add_batch(**batch)

    assert written_loop == written_batch == num_envs
    _assert_buffers_equal(buffer_loop, buffer_batch)


def test_add_batch_matches_looped_add_across_multiple_rollout_steps() -> None:
    rng = np.random.default_rng(1)
    num_envs = 8
    buffer_size = num_envs * 5

    buffer_loop = _make_buffer(buffer_size)
    buffer_batch = _make_buffer(buffer_size)

    for _ in range(5):
        batch = _random_transition_batch(num_envs, rng)
        _add_via_loop(buffer_loop, batch)
        buffer_batch.add_batch(**batch)
        _assert_buffers_equal(buffer_loop, buffer_batch)

    assert buffer_loop.full and buffer_batch.full


def test_add_batch_partial_write_matches_is_full_break() -> None:
    """buffer_size not a multiple of num_envs: both paths must silently
    drop the same trailing environments once the buffer fills."""
    rng = np.random.default_rng(2)
    num_envs = 8
    buffer_size = 5  # only 5 of 8 envs fit

    buffer_loop = _make_buffer(buffer_size)
    buffer_batch = _make_buffer(buffer_size)

    batch = _random_transition_batch(num_envs, rng)

    written_loop = _add_via_loop(buffer_loop, batch)
    written_batch = buffer_batch.add_batch(**batch)

    assert written_loop == written_batch == buffer_size
    assert buffer_loop.full and buffer_batch.full
    _assert_buffers_equal(buffer_loop, buffer_batch)


def test_add_batch_raises_when_already_full() -> None:
    rng = np.random.default_rng(3)
    num_envs = 4
    buffer = _make_buffer(num_envs)
    batch = _random_transition_batch(num_envs, rng)
    buffer.add_batch(**batch)
    assert buffer.is_full()

    with pytest.raises(RuntimeError):
        buffer.add_batch(**_random_transition_batch(num_envs, rng))

    with pytest.raises(RuntimeError):
        buffer.add(
            global_state=batch["global_states"][0],
            bay_observation=batch["bay_observations"][0],
            bay_action_mask=batch["bay_action_masks"][0],
            bay_action=int(batch["bay_actions"][0]),
            bay_log_prob=float(batch["bay_log_probs"][0]),
            row_observation=batch["row_observations"][0],
            row_action_mask=batch["row_action_masks"][0],
            row_action=int(batch["row_actions"][0]),
            row_log_prob=float(batch["row_log_probs"][0]),
            reward=float(batch["rewards"][0]),
            value=float(batch["values"][0]),
            next_value=float(batch["next_values"][0]),
            terminated=bool(batch["terminated"][0]),
            truncated=bool(batch["truncated"][0]),
        )


@pytest.mark.parametrize(
    "field,bad_shape",
    [
        ("global_states", (3, GLOBAL_OBS_DIM + 1)),
        ("bay_action_masks", (3, N_BAYS + 1)),
        ("row_actions", (2,)),
    ],
)
def test_add_batch_raises_on_shape_mismatch(field: str, bad_shape) -> None:
    rng = np.random.default_rng(4)
    num_envs = 3
    buffer = _make_buffer(num_envs)
    batch = _random_transition_batch(num_envs, rng)
    batch[field] = np.zeros(bad_shape, dtype=batch[field].dtype)

    with pytest.raises(ValueError):
        buffer.add_batch(**batch)


def test_add_batch_raises_on_invalid_action_under_mask() -> None:
    rng = np.random.default_rng(5)
    num_envs = 3
    buffer_add = _make_buffer(num_envs)
    buffer_batch = _make_buffer(num_envs)
    batch = _random_transition_batch(num_envs, rng)

    # Corrupt env 1's bay action to point at a masked-out bay.
    batch["bay_action_masks"][1, :] = False
    batch["bay_action_masks"][1, 0] = True
    batch["bay_actions"][1] = (batch["bay_actions"][1] + 1) % N_BAYS
    if batch["bay_action_masks"][1, batch["bay_actions"][1]]:
        batch["bay_actions"][1] = (batch["bay_actions"][1] + 1) % N_BAYS

    with pytest.raises(ValueError):
        buffer_add.add(
            global_state=batch["global_states"][1],
            bay_observation=batch["bay_observations"][1],
            bay_action_mask=batch["bay_action_masks"][1],
            bay_action=int(batch["bay_actions"][1]),
            bay_log_prob=float(batch["bay_log_probs"][1]),
            row_observation=batch["row_observations"][1],
            row_action_mask=batch["row_action_masks"][1],
            row_action=int(batch["row_actions"][1]),
            row_log_prob=float(batch["row_log_probs"][1]),
            reward=float(batch["rewards"][1]),
            value=float(batch["values"][1]),
            next_value=float(batch["next_values"][1]),
            terminated=bool(batch["terminated"][1]),
            truncated=bool(batch["truncated"][1]),
        )

    with pytest.raises(ValueError):
        buffer_batch.add_batch(**batch)


def test_add_batch_does_not_validate_rows_beyond_capacity() -> None:
    """A malformed row (out-of-range action) beyond the buffer's
    remaining capacity must not raise, exactly as the per-row
    `if buffer.is_full(): break; buffer.add(...)` loop this replaces
    would never even call add() - and thus never validate anything -
    for that row."""

    rng = np.random.default_rng(6)
    num_envs = 8
    buffer_size = 5  # only rows 0..4 fit; rows 5..7 are never written

    buffer = _make_buffer(buffer_size)
    batch = _random_transition_batch(num_envs, rng)

    # Corrupt a row past the writable prefix with an out-of-range bay
    # action - this row should simply be dropped, not raise.
    batch["bay_actions"][6] = N_BAYS  # out of range: valid is [0, N_BAYS)

    written = buffer.add_batch(**batch)

    assert written == buffer_size
    assert buffer.full
    # The written rows' actions came from the (valid) first `written`
    # entries, unaffected by the corrupted row beyond them.
    np.testing.assert_array_equal(
        buffer.bay_actions[:written], batch["bay_actions"][:written]
    )


def test_add_batch_raises_for_out_of_range_action_within_capacity() -> None:
    """The mirror case: an out-of-range action WITHIN the writable
    prefix must still raise, exactly as add() would for that row."""

    rng = np.random.default_rng(7)
    num_envs = 4
    buffer = _make_buffer(num_envs)
    batch = _random_transition_batch(num_envs, rng)

    batch["row_actions"][1] = N_ROWS  # out of range: valid is [0, N_ROWS)

    with pytest.raises(ValueError):
        buffer.add_batch(**batch)
