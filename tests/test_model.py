"""Tests for the full model forward pass.

These exercise the ASSEMBLED model (embedding -> N decoder layers ->
final norm -> output projection), using random weights of the correct
shape (see testutil/random_weights.py for why that's a legitimate way to
test architecture correctness without needing real downloaded weights).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import LLAMA_3_2_1B, LlamaConfig
from model.model import forward, next_token_greedy
from testutil.random_weights import random_model_weights


def test_forward_output_shape():
    config = LLAMA_3_2_1B
    weights = random_model_weights(config, num_layers=2)
    input_ids = torch.randint(0, config.vocab_size, (2, 7))

    logits = forward(input_ids, weights, config)

    assert logits.shape == (2, 7, config.vocab_size)


def test_forward_produces_no_nan_or_inf():
    config = LLAMA_3_2_1B
    weights = random_model_weights(config, num_layers=4)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    logits = forward(input_ids, weights, config)

    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_forward_is_deterministic_given_fixed_weights():
    config = LLAMA_3_2_1B
    weights = random_model_weights(config, num_layers=2)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    logits_1 = forward(input_ids, weights, config)
    logits_2 = forward(input_ids, weights, config)

    assert torch.equal(logits_1, logits_2), "same weights + same input must give identical output"


def test_causal_masking_holds_across_the_full_stack():
    # The single most important end-to-end correctness property: a
    # token's logits must depend ONLY on itself and earlier tokens,
    # never on later ones — checked here by changing a LATER token and
    # confirming an EARLIER position's logits don't change at all, after
    # going through the full embedding -> N layers -> output stack, not
    # just a single attention call in isolation (which test_attention.py
    # already covers). Bugs in residual wiring or layer-stacking could
    # in principle leak future information even if each individual
    # attention call is causally correct, so this is worth checking at
    # the assembled-model level too.
    config = LLAMA_3_2_1B
    weights = random_model_weights(config, num_layers=3)

    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    logits_original = forward(input_ids, weights, config)

    modified_ids = input_ids.clone()
    modified_ids[0, -1] = (modified_ids[0, -1] + 1) % config.vocab_size  # change only the LAST token
    logits_modified = forward(modified_ids, weights, config)

    # Every position EXCEPT the last must be byte-for-byte identical —
    # changing the last token must not affect anything earlier.
    assert torch.equal(logits_original[:, :-1, :], logits_modified[:, :-1, :]), (
        "changing the last token leaked into earlier positions' logits — causal masking broken somewhere in the full stack"
    )
    # And the last position's logits SHOULD differ (sanity check that
    # the test setup is meaningful — if they didn't differ, the model
    # might just be ignoring its own input entirely).
    assert not torch.equal(logits_original[:, -1, :], logits_modified[:, -1, :]), (
        "test setup error: changing the last token should change its own logits"
    )


def test_tied_embeddings_output_projection_is_embed_tokens_transposed():
    config = LlamaConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                          intermediate_size=32)
    weights = random_model_weights(config, num_layers=1)

    assert weights.lm_head_weight is None  # tied-embeddings case
    proj = weights.output_projection()

    assert torch.equal(proj, weights.embed_tokens.T)


def test_untied_lm_head_takes_precedence_over_tied_fallback():
    config = LlamaConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                          intermediate_size=32)
    weights = random_model_weights(config, num_layers=1)
    weights.lm_head_weight = torch.randn(config.vocab_size, config.hidden_size)

    proj = weights.output_projection()

    assert torch.equal(proj, weights.lm_head_weight.T)
    assert not torch.equal(proj, weights.embed_tokens.T)


def test_next_token_greedy_picks_argmax_at_last_position():
    batch, seq_len, vocab_size = 2, 4, 10
    logits = torch.zeros(batch, seq_len, vocab_size)
    # Make the answer at the LAST position unambiguous and different
    # per batch element, so this also checks the right axis is used.
    logits[0, -1, 7] = 100.0
    logits[1, -1, 3] = 100.0
    # Plant a decoy at an EARLIER position to make sure it's ignored.
    logits[0, 0, 9] = 999.0

    next_tokens = next_token_greedy(logits)

    assert next_tokens.tolist() == [7, 3]


def test_smaller_config_runs_end_to_end_quickly():
    # A deliberately tiny config, for fast iteration during development
    # — this isn't testing anything DIFFERENT from the full-size tests
    # above, just confirming the model code has no hidden dependency on
    # Llama-3.2-1B's specific dimensions (e.g. an accidentally hardcoded
    # head count) that would make it break on a different-sized config.
    config = LlamaConfig(
        vocab_size=50, hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        intermediate_size=16, rms_norm_eps=1e-5, rope_theta=10000.0,
    )
    weights = random_model_weights(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 3))

    logits = forward(input_ids, weights, config)

    assert logits.shape == (1, 3, config.vocab_size)
    assert not torch.isnan(logits).any()
