import pytest
import torch

from neuro_scan.tuned_lens import AffineTranslator, TunedLens


class TestAffineTranslator:
    def test_identity_init(self):
        t = AffineTranslator(64)
        x = torch.randn(64)
        out = t(x)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_forward_shape(self):
        t = AffineTranslator(128)
        x = torch.randn(10, 128)
        out = t(x)
        assert out.shape == (10, 128)

    def test_gradient_flow(self):
        t = AffineTranslator(32)
        x = torch.randn(32)
        y = t(x).sum()
        y.backward()
        assert t.weight.grad is not None
        assert t.bias.grad is not None


class TestTunedLens:
    def test_save_load_roundtrip(self, tmp_path):
        translators = [AffineTranslator(64) for _ in range(4)]
        # Perturb one translator
        translators[1].weight.data += 0.1
        translators[1].bias.data += 0.05

        lens = TunedLens(translators)
        save_path = tmp_path / "lens.safetensors"
        lens.save(save_path)

        loaded = TunedLens.load(save_path)
        assert len(loaded.translators) == 4
        torch.testing.assert_close(
            loaded.translators[1].weight.data,
            translators[1].weight.data,
        )
        torch.testing.assert_close(
            loaded.translators[1].bias.data,
            translators[1].bias.data,
        )

    def test_project_uses_translator(self):
        t = AffineTranslator(64)
        t.weight.data = torch.eye(64) * 2.0  # scale by 2
        lens = TunedLens([t])

        h = torch.randn(1, 64)
        norm_fn = lambda x: x
        head_fn = lambda x: x  # identity

        result = lens.project(h, 0, norm_fn, head_fn)
        expected = h.float() * 2.0  # scaled by translator
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
