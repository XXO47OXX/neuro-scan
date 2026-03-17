from __future__ import annotations

import json

import pytest
import torch

from neuro_scan.config import (
    AblationResult,
    LogitLensStep,
    NeuroReport,
    NeuroScanConfig,
)


# --- Mock Backend ---


class MockBackend:
    """Mock backend returning deterministic results for testing."""

    def __init__(self, total_layers: int = 32, vocab_size: int = 32000):
        self._total_layers = total_layers
        self._vocab_size = vocab_size
        self._loaded = False
        self._calls: list[str] = []

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True
        self._calls.append(f"load:{model_path}")

    def get_total_layers(self) -> int:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._total_layers

    def get_tokenizer(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return MockTokenizer()

    def forward(self, text: str) -> torch.Tensor:
        self._calls.append(f"forward:{text[:30]}")
        torch.manual_seed(42)
        return torch.randn(self._vocab_size)

    def forward_with_ablation(
        self, text: str, ablated_layers: list[int]
    ) -> torch.Tensor:
        self._calls.append(f"ablate:{ablated_layers}")
        # Simulate: ablating a middle layer slightly changes the output
        torch.manual_seed(42 + sum(ablated_layers))
        return torch.randn(self._vocab_size)

    def forward_with_hidden_states(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self._calls.append("hidden_states")
        torch.manual_seed(42)
        logits = torch.randn(self._vocab_size)
        hidden_states = [
            torch.randn(5, 768) for _ in range(self._total_layers)
        ]
        return logits, hidden_states

    def forward_with_attention(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self._calls.append("attention")
        torch.manual_seed(42)
        logits = torch.randn(self._vocab_size)
        # Simulate attention: (num_heads, seq_len, seq_len)
        attention = [
            torch.softmax(torch.randn(8, 5, 5), dim=-1)
            for _ in range(self._total_layers)
        ]
        return logits, attention

    def get_norm_and_head(self):
        def mock_norm(x):
            return x

        def mock_head(x):
            # Project to vocab_size
            torch.manual_seed(99)
            w = torch.randn(x.shape[-1], self._vocab_size)
            return x @ w

        return mock_norm, mock_head

    def cleanup(self) -> None:
        self._loaded = False
        self._calls.append("cleanup")


class MockTokenizer:
    """Mock tokenizer that maps digits 0-9 to token IDs 48-57."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if len(text) == 1 and text.isdigit():
            return [48 + int(text)]
        return [ord(c) for c in text[:10]]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(min(t, 127)) for t in token_ids)


# --- Fixtures ---


@pytest.fixture
def mock_backend():
    """Provide a pre-loaded MockBackend."""
    b = MockBackend()
    b.load("mock-model")
    return b


@pytest.fixture
def mock_backend_small():
    """Provide a pre-loaded MockBackend with 8 layers."""
    b = MockBackend(total_layers=8)
    b.load("mock-small")
    return b


@pytest.fixture
def mock_backend_large():
    """Provide a pre-loaded MockBackend with 80 layers (Qwen2 style)."""
    b = MockBackend(total_layers=80)
    b.load("mock-qwen2-80")
    return b


@pytest.fixture
def mock_tokenizer():
    """Provide a MockTokenizer."""
    return MockTokenizer()


@pytest.fixture
def sample_config():
    """Provide a standard NeuroScanConfig."""
    return NeuroScanConfig(
        model_path="mock-model",
        probe_name="math",
        batch_size=4,
        output_dir="./test-results",
        top_k_layers=5,
    )


@pytest.fixture
def sample_ablation_results():
    """Provide sample ablation results for a 32-layer model."""
    results = []
    for i in range(32):
        # Simulate: middle layers are most important
        if 10 <= i <= 20:
            delta = 0.5 + (i - 10) * 0.1
        else:
            delta = 0.05 + i * 0.01
        results.append(
            AblationResult(
                layer_idx=i,
                score=4.5 - delta,
                score_delta=delta,
                uncertainty=0.1,
            )
        )
    return results


@pytest.fixture
def sample_logit_lens_trajectory():
    """Provide sample logit lens trajectories for 3 samples, 32 layers."""
    trajectories = []
    for s in range(3):
        trajectory = []
        for layer_idx in range(32):
            # Simulate: target token emerges gradually
            target_prob = min(0.8, layer_idx / 32.0)
            trajectory.append(
                LogitLensStep(
                    layer_idx=layer_idx,
                    top_token=f"tok_{layer_idx}",
                    top_token_prob=0.3 + layer_idx * 0.02,
                    target_token_prob=target_prob,
                    entropy=10.0 - layer_idx * 0.2,
                )
            )
        trajectories.append(trajectory)
    return trajectories


@pytest.fixture
def sample_report(sample_ablation_results, sample_logit_lens_trajectory):
    """Provide a complete sample NeuroReport."""
    from neuro_scan.labeler import label_layers

    labels = label_layers(
        total_layers=32,
        ablation_results=sample_ablation_results,
        logit_lens_trajectory=sample_logit_lens_trajectory,
    )

    sorted_by_impact = sorted(
        sample_ablation_results,
        key=lambda r: abs(r.score_delta),
        reverse=True,
    )
    top_layers = [r.layer_idx for r in sorted_by_impact[:5]]

    return NeuroReport(
        model_path="mock-model",
        probe_name="math",
        total_layers=32,
        baseline_score=4.5,
        baseline_uncertainty=0.1,
        ablation_results=sample_ablation_results,
        logit_lens_trajectory=sample_logit_lens_trajectory,
        attention_entropy=[[1.5, 2.0, 1.8] for _ in range(32)],
        layer_labels=labels,
        top_important_layers=top_layers,
        total_time_seconds=10.5,
    )


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide a temporary output directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def sample_probe_json(tmp_path):
    """Provide a temporary custom probe JSON file."""
    probe_data = {
        "name": "test_custom",
        "description": "Test custom probe",
        "scoring": "digits",
        "samples": [
            {
                "prompt": "Rate from 0-9: is 2+2=4?\nAnswer: ",
                "expected_score": 9.0,
                "metadata": {"category": "basic"},
            },
            {
                "prompt": "Rate from 0-9: is 1+1=3?\nAnswer: ",
                "expected_score": 1.0,
                "metadata": {"category": "basic"},
            },
        ],
    }
    path = tmp_path / "custom_probe.json"
    path.write_text(json.dumps(probe_data))
    return path
