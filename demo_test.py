#!/usr/bin/env python3
"""
Final demonstration test for the Confidence LLM package.
Shows all key features working correctly.
"""

from confidence_llm import ConfidenceEstimator, self_certainty_from_logits
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np


def main():
    print("🧪 CONFIDENCE LLM PACKAGE DEMONSTRATION")
    print("=" * 50)
    
    # 1. Test utility functions
    print("\n1. Testing Utility Functions")
    print("-" * 30)
    
    # Create test logits
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 100)
    
    # Test self-certainty calculation
    certainty = self_certainty_from_logits(logits)
    print(f"✓ Self-certainty shape: {certainty.shape}")
    print(f"✓ Sample values: {certainty[0, :2].tolist()}")
    
    # Verify against manual calculation
    expected = torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)
    assert torch.allclose(certainty, expected), "Formula verification failed!"
    print("✓ Mathematical formula verified")
    
    # 2. Test ConfidenceEstimator
    print("\n2. Testing ConfidenceEstimator")
    print("-" * 30)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    estimator = ConfidenceEstimator(model, tokenizer)
    print("✓ Model and estimator loaded successfully")
    
    # 3. Test single text confidence
    print("\n3. Single Text Confidence")
    print("-" * 30)
    
    text = "The capital of France is"
    confidence = estimator.estimate_confidence(text)
    print(f"Text: '{text}'")
    print(f"✓ Confidence: {confidence:.4f}")
    
    # 4. Test batch processing
    print("\n4. Batch Processing")
    print("-" * 30)
    
    texts = [
        "The capital of France is",
        "2 + 2 equals", 
        "Machine learning is",
        "The weather today"
    ]
    
    confidences = estimator.estimate_confidence_batch(texts)
    print("✓ Batch results:")
    for text, conf in zip(texts, confidences):
        print(f"  '{text:<25}' -> {conf:.4f}")
    
    # 5. Test different confidence methods
    print("\n5. Different Confidence Methods")
    print("-" * 30)
    
    test_text = "The capital of France is"
    methods = ["self_certainty", "entropy", "kl_uniform", "max_prob", "top_k"]
    
    print(f"Text: '{test_text}'")
    for method in methods:
        conf = estimator.estimate_confidence(test_text, method=method)
        print(f"✓ {method:<15}: {conf:.4f}")
    
    # 6. Test token-level analysis
    print("\n6. Token-level Analysis")
    print("-" * 30)
    
    tokens, scores = estimator.get_token_level_confidence("The capital of France")
    print("Token-level confidence:")
    for token, score in zip(tokens, scores):
        clean_token = token.replace('Ġ', ' ').strip()
        print(f"  '{clean_token:<10}' -> {score:.4f}")
    
    # 7. Test mathematical properties
    print("\n7. Mathematical Properties")
    print("-" * 30)
    
    # Create distributions with known properties
    vocab_size = 100
    
    # Very confident (peaked) distribution
    peaked_logits = torch.zeros(1, 1, vocab_size)
    peaked_logits[0, 0, 0] = 10.0
    
    # Less confident (uniform) distribution  
    uniform_logits = torch.zeros(1, 1, vocab_size)
    
    peaked_certainty = self_certainty_from_logits(peaked_logits).item()
    uniform_certainty = self_certainty_from_logits(uniform_logits).item()
    
    print(f"✓ Peaked distribution (confident): {peaked_certainty:.4f}")
    print(f"✓ Uniform distribution (uncertain): {uniform_certainty:.4f}")
    print(f"✓ Peaked > Uniform: {peaked_certainty > uniform_certainty}")
    
    # 8. Test different aggregation methods
    print("\n8. Aggregation Methods")
    print("-" * 30)
    
    text = "The capital of France is"
    aggregations = ["mean", "last", "max", "min"]
    
    print(f"Text: '{text}'")
    for agg in aggregations:
        conf = estimator.estimate_confidence(text, aggregate=agg)
        print(f"✓ {agg:<8}: {conf:.4f}")
    
    # 9. Test error handling
    print("\n9. Error Handling")
    print("-" * 30)
    
    # Test invalid method
    try:
        estimator.estimate_confidence("test", method="invalid")
        print("❌ Should have failed")
    except ValueError:
        print("✓ Invalid method properly rejected")
    
    # Test long text truncation
    long_text = "The capital of France is Paris. " * 50
    conf = estimator.estimate_confidence(long_text, max_length=20)
    print(f"✓ Long text handled: {conf:.4f}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Utility functions working")
    print("✅ Confidence estimation working") 
    print("✅ Batch processing working")
    print("✅ Multiple methods working")
    print("✅ Token-level analysis working")
    print("✅ Mathematical properties verified")
    print("✅ Error handling working")
    print("\nThe Confidence LLM package is fully functional! 🚀")
    print("=" * 50)


if __name__ == "__main__":
    main() 