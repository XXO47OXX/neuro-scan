"""Tests for circuit detection."""

import pytest
import torch

from neuro_scan.circuit import (
    CircuitConfig,
    CircuitReport,
    InteractionResult,
    _generate_candidate_pairs,
    _select_top_k_layers,
)
from neuro_scan.config import AblationResult
from neuro_scan.similarity import compute_block_influence, compute_layer_similarity


class TestSimilarity:
    def test_compute_layer_similarity_shape(self):
        hidden_states = [torch.randn(5, 64) for _ in range(4)]
        sim = compute_layer_similarity(hidden_states)
        assert sim.shape == (4, 4)

    def test_similarity_diagonal_is_one(self):
        hidden_states = [torch.randn(5, 64) for _ in range(4)]
        sim = compute_layer_similarity(hidden_states)
        for i in range(4):
            assert abs(sim[i, i] - 1.0) < 1e-5

    def test_similarity_is_symmetric(self):
        hidden_states = [torch.randn(5, 64) for _ in range(4)]
        sim = compute_layer_similarity(hidden_states)
        for i in range(4):
            for j in range(4):
                assert abs(sim[i, j] - sim[j, i]) < 1e-6

    def test_block_influence_length(self):
        hidden_states = [torch.randn(5, 64) for _ in range(4)]
        bi = compute_block_influence(hidden_states)
        assert len(bi) == 3  # num_layers - 1

    def test_block_influence_range(self):
        hidden_states = [torch.randn(5, 64) for _ in range(4)]
        bi = compute_block_influence(hidden_states)
        for score in bi:
            assert 0.0 <= score <= 2.0  # cos_sim in [-1,1] so BI in [0,2]

    def test_identical_layers_zero_bi(self):
        h = torch.randn(5, 64)
        hidden_states = [h.clone() for _ in range(3)]
        bi = compute_block_influence(hidden_states)
        for score in bi:
            assert abs(score) < 1e-5


class TestCandidatePairs:
    def test_fast_strategy(self):
        top_k = [5, 10, 15, 20]
        pairs = _generate_candidate_pairs(top_k, 32, "fast")
        # Should have K(K-1)/2 + adjacent pairs
        assert len(pairs) >= 6  # C(4,2) = 6
        assert all(a < b for a, b in pairs)

    def test_exhaustive_strategy(self):
        pairs = _generate_candidate_pairs([0, 1], 4, "exhaustive")
        assert len(pairs) == 6  # C(4,2) = 6

    def test_no_duplicates(self):
        top_k = [5, 6, 7, 8]
        pairs = _generate_candidate_pairs(top_k, 32, "fast")
        assert len(pairs) == len(set(pairs))

    def test_no_self_pairs(self):
        top_k = [5, 10]
        pairs = _generate_candidate_pairs(top_k, 32, "fast")
        for a, b in pairs:
            assert a != b


class TestSelectTopK:
    def test_selects_by_delta(self):
        results = [
            AblationResult(layer_idx=0, score=4.0, score_delta=0.1, uncertainty=0.1),
            AblationResult(layer_idx=1, score=3.0, score_delta=1.5, uncertainty=0.1),
            AblationResult(layer_idx=2, score=4.0, score_delta=0.5, uncertainty=0.1),
            AblationResult(layer_idx=3, score=2.0, score_delta=2.0, uncertainty=0.1),
        ]
        top = _select_top_k_layers(results, k=2)
        assert top == [3, 1]


class TestInteractionResult:
    def test_frozen(self):
        r = InteractionResult(
            layer_i=5,
            layer_j=10,
            individual_delta_i=0.3,
            individual_delta_j=0.2,
            joint_delta=0.8,
            interaction_effect=0.3,
            interaction_type="synergistic",
        )
        assert r.interaction_effect == 0.3
        with pytest.raises(AttributeError):
            r.layer_i = 99
