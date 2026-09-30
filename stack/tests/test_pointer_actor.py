"""
PointerActor is the existing pointer architecture, not a re-implementation.

With the same weights, the bay actor (tokens_per_action = n_rows) must
produce exactly the logits of transformer_bay_policy.BayTransformerActorCritic
and the row actor (tokens_per_action = 1) exactly those of
transformer_policy.TransformerActorCritic - the actor heads of the SB3
policies these agents were derived from.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from stack.models.pointer_actor import PointerActor
from stack.models.transformer_bay_policy import BayTransformerActorCritic
from stack.models.transformer_policy import TransformerActorCritic

N_BAYS, N_ROWS, FEATURES = 3, 4, 10
EMBED_DIM, N_HEADS = 16, 2


def _space(n_stacks: int) -> spaces.Box:
    return spaces.Box(-np.inf, np.inf, (n_stacks * FEATURES,), np.float32)


def _actor(n_stacks: int, tokens_per_action: int) -> PointerActor:
    torch.manual_seed(0)
    return PointerActor(
        _space(n_stacks),
        n_stacks=n_stacks,
        tokens_per_action=tokens_per_action,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=1,
        dropout=0.0,
    ).eval()


def _observations(n_stacks: int, batch: int = 5) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, n_stacks * FEATURES)


def test_bay_actor_matches_existing_bay_pointer_policy() -> None:
    n_stacks = N_BAYS * N_ROWS
    actor = _actor(n_stacks, tokens_per_action=N_ROWS)

    reference = BayTransformerActorCritic(
        feature_dim=n_stacks * EMBED_DIM + 1,
        n_stacks=n_stacks,
        n_bays=N_BAYS,
        n_rows_per_bay=N_ROWS,
        embed_dim=EMBED_DIM,
        group_num=1,
        n_heads=N_HEADS,
    )
    reference.decoder.load_state_dict(actor.decoder.state_dict())

    observations = _observations(n_stacks)
    with torch.no_grad():
        expected = reference.forward_actor(actor.encoder(observations))
        torch.testing.assert_close(actor(observations), expected, rtol=0, atol=0)


def test_row_actor_matches_existing_stack_pointer_policy() -> None:
    actor = _actor(N_ROWS, tokens_per_action=1)

    reference = TransformerActorCritic(
        feature_dim=N_ROWS * EMBED_DIM + 1,
        n_stacks=N_ROWS,
        embed_dim=EMBED_DIM,
        group_num=1,
        n_heads=N_HEADS,
    )
    reference.decoder.load_state_dict(actor.decoder.state_dict())

    observations = _observations(N_ROWS)
    with torch.no_grad():
        expected = reference.forward_actor(actor.encoder(observations))
        torch.testing.assert_close(actor(observations), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens_per_action", [1, N_ROWS])
def test_masked_actions_are_never_chosen(tokens_per_action: int) -> None:
    n_stacks = N_BAYS * N_ROWS if tokens_per_action > 1 else N_ROWS
    actor = _actor(n_stacks, tokens_per_action)
    observations = _observations(n_stacks, batch=64)

    torch.manual_seed(2)
    masks = torch.rand(64, actor.n_actions) > 0.5
    masks[torch.arange(64), torch.randint(actor.n_actions, (64,))] = True

    unmasked_logits = actor(observations)
    greedy, greedy_log_probs = actor.act(observations, masks, deterministic=True)
    sampled, sampled_log_probs = actor.act(observations, masks)

    assert masks.gather(1, greedy[:, None]).all()
    assert masks.gather(1, sampled[:, None]).all()
    expected_greedy = unmasked_logits.masked_fill(~masks, float("-inf")).argmax(dim=-1)
    torch.testing.assert_close(greedy, expected_greedy)

    # evaluate_actions scores stored actions exactly as act() did.
    log_probs, entropy = actor.evaluate_actions(observations, sampled, masks)
    torch.testing.assert_close(log_probs, sampled_log_probs)
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(
        actor.evaluate_actions(observations, greedy, masks)[0], greedy_log_probs
    )
