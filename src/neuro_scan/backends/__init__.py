"""Inference backends for neuro-scan.

Each backend wraps a specific inference framework and implements
the neuroanatomy analysis interface (ablation, hidden states, attention).
"""

from neuro_scan.backends.base import Backend

__all__ = ["Backend"]
