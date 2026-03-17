from __future__ import annotations

import logging

import numpy as np
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def compute_layer_similarity(
    hidden_states: list[torch.Tensor],
) -> np.ndarray:
    """Compute cosine similarity matrix between all layer pairs.

    Args:
        hidden_states: Per-layer hidden states, each (seq_len, d_model).
            Uses last token position for comparison.

    Returns:
        Similarity matrix of shape (num_layers, num_layers).
    """
    vecs = [hs[-1].float() for hs in hidden_states]
    n = len(vecs)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            cos = F.cosine_similarity(
                vecs[i].unsqueeze(0),
                vecs[j].unsqueeze(0),
            ).item()
            sim_matrix[i, j] = cos
            sim_matrix[j, i] = cos

    return sim_matrix


def compute_block_influence(
    hidden_states: list[torch.Tensor],
) -> list[float]:
    """Compute Block Influence score for each layer.

    BI(layer_i) = 1 - cos_sim(h_{i-1}, h_i)
    where h_{i-1} is the residual stream input and h_i is the output.

    Low BI = layer contributes little (redundant/removable).
    High BI = layer contributes substantially (critical).

    Reference: ShortGPT (ACL 2025).

    Args:
        hidden_states: Per-layer hidden states, each (seq_len, d_model).
            Uses last token position.

    Returns:
        List of BI scores, length = num_layers - 1 (no BI for first layer).
        Index i corresponds to BI of layer i+1 (comparing layer i to i+1).
    """
    bi_scores = []
    for i in range(1, len(hidden_states)):
        cos = F.cosine_similarity(
            hidden_states[i - 1][-1].unsqueeze(0).float(),
            hidden_states[i][-1].unsqueeze(0).float(),
        ).item()
        bi_scores.append(1.0 - cos)
    return bi_scores
