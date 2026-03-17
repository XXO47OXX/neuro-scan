"""Evaluation probes for neuro-scan.

Probes provide standardized test inputs and scoring criteria for evaluating
layer-level behavior in transformer models.
"""

from neuro_scan.probes.base import Probe, ProbeSample

__all__ = ["Probe", "ProbeSample"]
