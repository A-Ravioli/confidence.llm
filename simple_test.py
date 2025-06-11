#!/usr/bin/env python3
"""
Simple test script to verify core functionality of the confidence LLM package.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from confidence_llm import ConfidenceEstimator, self_certainty_from_logits


def test_utility_functions():
    """Test core utility functions."""
    print("Testing utility functions...")
    
    # Test self-certainty calculation
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 100)  # batch=2, seq_len=3, vocab=100
    
    certainty = self_certainty_from_logits(logits)
    print(f"✓ Self-certainty shape: {certainty.shape}")
    print(f"✓ Self-certainty values: {certainty[0].tolist()}")
    
    # Verify the formula manually
    expected = torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)
    assert torch.allclose(certainty, expected), "Self-certainty calculation incorrect!"
    print("✓ Self-certainty formula verified")
    
    return True


def test_confidence_estimator():
    """Test ConfidenceEstimator functionality."""
    print("\nTesting ConfidenceEstimator...")
    
    # Load small model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    estimator = ConfidenceEstimator(model, tokenizer)
    print("✓ Model loaded successfully")
    
    # Test single text confidence
    text = "The capital of France is"
    confidence = estimator.estimate_confidence(text)
    print(f"✓ Single text confidence: {confidence:.4f}")
    assert isinstance(confidence, float), "Confidence should be a float"
    assert np.isfinite(confidence), "Confidence should be finite"
    
    # Test batch processing
    texts = ["The capital of France is", "2 + 2 equals", "Machine learning is"]
    confidences = estimator.estimate_confidence_batch(texts)
    print(f"✓ Batch confidences: {[f'{c:.4f}' for c in confidences]}")
    assert len(confidences) == len(texts), "Should get one confidence per text"
    
    # Test different methods
    methods = ["self_certainty", "entropy", "kl_uniform", "max_prob"]
    print("✓ Testing different methods:")
    for method in methods:
        conf = estimator.estimate_confidence(text, method=method)
        print(f"  {method}: {conf:.4f}")
        assert isinstance(conf, float) and np.isfinite(conf), f"Method {method} failed"
    
    # Test token-level analysis
    tokens, scores = estimator.get_token_level_confidence("The capital of France")
    print(f"✓ Token-level analysis: {len(tokens)} tokens, {len(scores)} scores")
    assert len(tokens) == len(scores), "Should get equal number of tokens and scores"
    
    return True


def test_mathematical_properties():
    """Test mathematical properties of confidence measures."""
    print("\nTesting mathematical properties...")
    
    # Create test distributions
    vocab_size = 100
    
    # Very peaked distribution (high confidence)
    peaked_logits = torch.zeros(1, 1, vocab_size)
    peaked_logits[0, 0, 0] = 10.0
    
    # Uniform distribution (low confidence)
    uniform_logits = torch.zeros(1, 1, vocab_size)
    
    # Test self-certainty
    peaked_certainty = self_certainty_from_logits(peaked_logits).item()
    uniform_certainty = self_certainty_from_logits(uniform_logits).item()
    
    print(f"✓ Peaked distribution certainty: {peaked_certainty:.4f}")
    print(f"✓ Uniform distribution certainty: {uniform_certainty:.4f}")
    
    # Peaked should have higher self-certainty
    assert peaked_certainty > uniform_certainty, "Peaked distribution should have higher certainty"
    print("✓ Mathematical properties verified")
    
    return True


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")
    
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    estimator = ConfidenceEstimator(model, tokenizer)
    
    # Test invalid method
    try:
        estimator.estimate_confidence("test", method="invalid_method")
        assert False, "Should have raised ValueError for invalid method"
    except ValueError:
        print("✓ Invalid method properly rejected")
    
    # Test very long text (should truncate)
    long_text = "The capital of France is Paris. " * 100
    confidence = estimator.estimate_confidence(long_text, max_length=50)
    print(f"✓ Long text handled: {confidence:.4f}")
    assert isinstance(confidence, float) and np.isfinite(confidence)
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLE CONFIDENCE LLM PACKAGE TEST")
    print("=" * 60)
    
    try:
        # Run tests
        test_utility_functions()
        test_confidence_estimator()
        test_mathematical_properties()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! The package is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 