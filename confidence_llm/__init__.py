"""
Confidence LLM - A package for measuring language model confidence using KL divergence.

This package implements model confidence measurement using the self-certainty method
from the Intuitor paper, measuring how confident a model is in its predictions.
"""

from .confidence_estimator import ConfidenceEstimator
from .utils import (
    self_certainty_from_logits,
    entropy_from_logits,
    kl_divergence_from_uniform,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "ConfidenceEstimator",
    "self_certainty_from_logits",
    "entropy_from_logits", 
    "kl_divergence_from_uniform",
] 