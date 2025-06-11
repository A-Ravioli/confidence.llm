#!/usr/bin/env python3
"""
Simple test script to verify the confidence-llm package works correctly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from confidence_llm import ConfidenceEstimator, self_certainty_from_logits


def test_basic_functionality():
    """Test basic functionality of the package."""
    print("Testing Confidence LLM package...")
    
    # Test 1: Load model and create estimator
    print("1. Loading model and creating estimator...")
    try:
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")  # Small model for testing
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        estimator = ConfidenceEstimator(model, tokenizer)
        print("   ✓ Successfully created ConfidenceEstimator")
    except Exception as e:
        print(f"   ✗ Error creating estimator: {e}")
        return False
    
    # Test 2: Basic confidence estimation
    print("2. Testing basic confidence estimation...")
    try:
        test_text = "The capital of France is"
        confidence = estimator.estimate_confidence(test_text)
        print(f"   ✓ Confidence for '{test_text}': {confidence:.4f}")
        
        if not isinstance(confidence, float):
            print(f"   ✗ Expected float, got {type(confidence)}")
            return False
    except Exception as e:
        print(f"   ✗ Error in confidence estimation: {e}")
        return False
    
    # Test 3: Batch processing
    print("3. Testing batch processing...")
    try:
        test_texts = [
            "The capital of France is",
            "2 + 2 equals",
            "Machine learning is"
        ]
        confidences = estimator.estimate_confidence_batch(test_texts)
        print(f"   ✓ Batch processing successful: {len(confidences)} scores")
        
        if len(confidences) != len(test_texts):
            print(f"   ✗ Expected {len(test_texts)} scores, got {len(confidences)}")
            return False
    except Exception as e:
        print(f"   ✗ Error in batch processing: {e}")
        return False
    
    # Test 4: Different methods
    print("4. Testing different confidence methods...")
    try:
        methods = ["self_certainty", "entropy", "kl_uniform", "max_prob"]
        test_text = "The capital of France is"
        
        for method in methods:
            confidence = estimator.estimate_confidence(test_text, method=method)
            print(f"   ✓ {method}: {confidence:.4f}")
    except Exception as e:
        print(f"   ✗ Error testing methods: {e}")
        return False
    
    # Test 5: Utility functions
    print("5. Testing utility functions...")
    try:
        # Create dummy logits
        logits = torch.randn(2, 10, 1000)  # batch_size=2, seq_len=10, vocab_size=1000
        
        certainty_scores = self_certainty_from_logits(logits)
        print(f"   ✓ self_certainty_from_logits: shape {certainty_scores.shape}")
        
        if certainty_scores.shape != (2, 10):
            print(f"   ✗ Expected shape (2, 10), got {certainty_scores.shape}")
            return False
    except Exception as e:
        print(f"   ✗ Error testing utility functions: {e}")
        return False
    
    # Test 6: Token-level confidence
    print("6. Testing token-level confidence...")
    try:
        tokens, scores = estimator.get_token_level_confidence("The capital of France")
        print(f"   ✓ Token-level analysis: {len(tokens)} tokens, {len(scores)} scores")
        
        if len(tokens) != len(scores):
            print(f"   ✗ Token count mismatch: {len(tokens)} tokens, {len(scores)} scores")
            return False
    except Exception as e:
        print(f"   ✗ Error in token-level confidence: {e}")
        return False
    
    print("\n✓ All tests passed! The package is working correctly.")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        estimator = ConfidenceEstimator(model, tokenizer)
        
        # Test empty string
        try:
            confidence = estimator.estimate_confidence("")
            print(f"   ✓ Empty string handled: {confidence:.4f}")
        except Exception as e:
            print(f"   ! Empty string error (expected): {e}")
        
        # Test very long text
        try:
            long_text = "The capital of France is " * 100
            confidence = estimator.estimate_confidence(long_text, max_length=50)
            print(f"   ✓ Long text truncated properly: {confidence:.4f}")
        except Exception as e:
            print(f"   ✗ Error with long text: {e}")
            return False
        
        # Test invalid method
        try:
            confidence = estimator.estimate_confidence("Test", method="invalid_method")
            print(f"   ✗ Should have raised error for invalid method")
            return False
        except ValueError:
            print(f"   ✓ Invalid method properly rejected")
        except Exception as e:
            print(f"   ✗ Unexpected error with invalid method: {e}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error in edge case testing: {e}")
        return False
    
    print("   ✓ Edge case testing completed")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("CONFIDENCE LLM PACKAGE TEST")
    print("=" * 50)
    
    success = True
    success &= test_basic_functionality()
    success &= test_edge_cases()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! The package is ready to use.")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main() 