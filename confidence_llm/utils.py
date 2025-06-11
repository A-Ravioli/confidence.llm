"""
Utility functions for computing confidence metrics from model logits.

This module contains the core mathematical functions for computing various
confidence metrics including self-certainty, entropy, and KL divergence.
"""

import torch
import torch.nn.functional as F
from typing import Union, Optional


def self_certainty_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculate self-certainty from logits using the Intuitor method.
    
    Self-certainty measures how far the model's output distribution is from uniform.
    Higher values indicate higher confidence.
    
    Formula: self_certainty = logsumexp(logits) - mean(logits)
    
    Args:
        logits (torch.Tensor): Model logits of shape (..., vocab_size)
        dim (int): Dimension along which to compute the self-certainty
        
    Returns:
        torch.Tensor: Self-certainty scores of shape (...,)
        
    Examples:
        >>> logits = torch.randn(2, 1000)  # batch_size=2, vocab_size=1000
        >>> certainty = self_certainty_from_logits(logits)
        >>> print(certainty.shape)  # torch.Size([2])
    """
    return torch.logsumexp(logits, dim=dim) - logits.mean(dim=dim)


def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculate entropy from logits.
    
    Entropy measures the uncertainty in the probability distribution.
    Lower entropy indicates higher confidence.
    
    Args:
        logits (torch.Tensor): Model logits of shape (..., vocab_size)
        dim (int): Dimension along which to compute entropy
        
    Returns:
        torch.Tensor: Entropy scores of shape (...,)
        
    Examples:
        >>> logits = torch.randn(2, 1000)
        >>> entropy = entropy_from_logits(logits)
        >>> print(entropy.shape)  # torch.Size([2])
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    return entropy


def kl_divergence_from_uniform(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculate KL divergence between the model's distribution and uniform distribution.
    
    This is another way to measure confidence - higher KL divergence from uniform
    indicates the model is more confident (distribution is more peaked).
    
    Args:
        logits (torch.Tensor): Model logits of shape (..., vocab_size)
        dim (int): Dimension along which to compute KL divergence
        
    Returns:
        torch.Tensor: KL divergence scores of shape (...,)
        
    Examples:
        >>> logits = torch.randn(2, 1000)
        >>> kl_div = kl_divergence_from_uniform(logits)
        >>> print(kl_div.shape)  # torch.Size([2])
    """
    vocab_size = logits.size(dim)
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    
    # Uniform distribution has log probability of -log(vocab_size) for each token
    uniform_log_prob = -torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
    
    # KL(P || U) = sum(P * (log(P) - log(U)))
    kl_div = torch.sum(probs * (log_probs - uniform_log_prob), dim=dim)
    return kl_div


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute mean of tensor with masked values.
    
    Args:
        values (torch.Tensor): Values to average
        mask (torch.Tensor): Mask tensor (1 for valid values, 0 for invalid)
        dim (int, optional): Dimension along which to compute mean
        
    Returns:
        torch.Tensor: Masked mean
    """
    return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


def confidence_from_variance(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculate confidence based on the variance of logits.
    
    Higher variance in logits can indicate lower confidence as the model
    is less sure about the relative probabilities.
    
    Args:
        logits (torch.Tensor): Model logits of shape (..., vocab_size)
        dim (int): Dimension along which to compute variance
        
    Returns:
        torch.Tensor: Confidence scores based on variance (lower variance = higher confidence)
    """
    return -torch.var(logits, dim=dim)


def max_probability_confidence(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculate confidence as the maximum probability in the distribution.
    
    This is a simple baseline where confidence is just the highest probability
    the model assigns to any token.
    
    Args:
        logits (torch.Tensor): Model logits of shape (..., vocab_size)
        dim (int): Dimension along which to find max probability
        
    Returns:
        torch.Tensor: Maximum probability confidence scores
    """
    probs = F.softmax(logits, dim=dim)
    return torch.max(probs, dim=dim)[0]


def top_k_confidence(logits: torch.Tensor, k: int = 5, dim: int = -1) -> torch.Tensor:
    """
    Calculate confidence as the sum of top-k probabilities.
    
    Args:
        logits (torch.Tensor): Model logits of shape (..., vocab_size)
        k (int): Number of top probabilities to sum
        dim (int): Dimension along which to compute top-k
        
    Returns:
        torch.Tensor: Top-k confidence scores
    """
    probs = F.softmax(logits, dim=dim)
    top_k_probs, _ = torch.topk(probs, k=k, dim=dim)
    return torch.sum(top_k_probs, dim=-1) 