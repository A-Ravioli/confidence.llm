#!/usr/bin/env python3
"""
Comprehensive unit tests for the Confidence LLM package.

This module contains tests for all major functionality including:
- Utility functions
- ConfidenceEstimator class
- Different confidence methods
- Error handling
- Edge cases
"""

import unittest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the parent directory to the path so we can import confidence_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from confidence_llm import (
    ConfidenceEstimator,
    self_certainty_from_logits,
    entropy_from_logits,
    kl_divergence_from_uniform,
)
from confidence_llm.utils import (
    masked_mean,
    max_probability_confidence,
    top_k_confidence,
    confidence_from_variance,
)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for confidence computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample logits for testing
        torch.manual_seed(42)  # For reproducibility
        self.logits = torch.randn(2, 5, 100)  # batch_size=2, seq_len=5, vocab_size=100
        self.small_logits = torch.randn(1, 3, 10)  # Smaller for manual verification
    
    def test_self_certainty_from_logits(self):
        """Test self-certainty calculation."""
        certainty = self_certainty_from_logits(self.logits)
        
        # Check shape
        self.assertEqual(certainty.shape, (2, 5))
        
        # Check that values are finite
        self.assertTrue(torch.all(torch.isfinite(certainty)))
        
        # Test manual calculation for small example
        small_certainty = self_certainty_from_logits(self.small_logits)
        expected = torch.logsumexp(self.small_logits, dim=-1) - self.small_logits.mean(dim=-1)
        torch.testing.assert_close(small_certainty, expected)
    
    def test_entropy_from_logits(self):
        """Test entropy calculation."""
        entropy = entropy_from_logits(self.logits)
        
        # Check shape
        self.assertEqual(entropy.shape, (2, 5))
        
        # Check that entropy is positive (for most cases)
        # Note: entropy can be negative for very peaked distributions, but should be finite
        self.assertTrue(torch.all(torch.isfinite(entropy)))
        self.assertTrue(torch.all(entropy >= 0))  # Should be non-negative for valid probability distributions
    
    def test_kl_divergence_from_uniform(self):
        """Test KL divergence from uniform distribution."""
        kl_div = kl_divergence_from_uniform(self.logits)
        
        # Check shape
        self.assertEqual(kl_div.shape, (2, 5))
        
        # Check that KL divergence is non-negative
        self.assertTrue(torch.all(kl_div >= 0))
        
        # Check that values are finite
        self.assertTrue(torch.all(torch.isfinite(kl_div)))
    
    def test_max_probability_confidence(self):
        """Test max probability confidence."""
        max_prob = max_probability_confidence(self.logits)
        
        # Check shape
        self.assertEqual(max_prob.shape, (2, 5))
        
        # Check that probabilities are between 0 and 1
        self.assertTrue(torch.all(max_prob >= 0))
        self.assertTrue(torch.all(max_prob <= 1))
    
    def test_top_k_confidence(self):
        """Test top-k confidence."""
        top_k_conf = top_k_confidence(self.logits, k=5)
        
        # Check shape
        self.assertEqual(top_k_conf.shape, (2, 5))
        
        # Check that values are between 0 and 1 (sum of probabilities)
        self.assertTrue(torch.all(top_k_conf >= 0))
        self.assertTrue(torch.all(top_k_conf <= 1))
    
    def test_masked_mean(self):
        """Test masked mean calculation."""
        values = torch.randn(2, 5)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float)
        
        masked_avg = masked_mean(values, mask, dim=1)
        
        # Check shape
        self.assertEqual(masked_avg.shape, (2,))
        
        # Manual calculation for first example
        expected_0 = values[0, :3].mean()  # First 3 elements
        expected_1 = values[1, :2].mean()  # First 2 elements
        
        torch.testing.assert_close(masked_avg[0], expected_0, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(masked_avg[1], expected_1, atol=1e-6, rtol=1e-6)


class TestConfidenceEstimator(unittest.TestCase):
    """Test ConfidenceEstimator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test model (once for all tests)."""
        # Use a small model for faster testing
        cls.model_name = "distilgpt2"
        print(f"Loading {cls.model_name} for testing...")
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.estimator = ConfidenceEstimator(cls.model, cls.tokenizer)
        print("Model loaded successfully!")
    
    def test_initialization(self):
        """Test ConfidenceEstimator initialization."""
        # Test that estimator was created successfully
        self.assertIsNotNone(self.estimator)
        self.assertIsNotNone(self.estimator.model)
        self.assertIsNotNone(self.estimator.tokenizer)
        
        # Test device assignment
        self.assertIn(self.estimator.device, ['cpu', 'cuda'])
        
        # Test that padding token is set
        self.assertIsNotNone(self.estimator.tokenizer.pad_token)
    
    def test_single_text_confidence(self):
        """Test confidence estimation for a single text."""
        text = "The capital of France is"
        confidence = self.estimator.estimate_confidence(text)
        
        # Check that we get a float
        self.assertIsInstance(confidence, float)
        
        # Check that confidence is finite
        self.assertTrue(np.isfinite(confidence))
        
        print(f"Confidence for '{text}': {confidence:.4f}")
    
    def test_batch_confidence(self):
        """Test batch confidence estimation."""
        texts = [
            "The capital of France is",
            "2 + 2 equals",
            "Machine learning is"
        ]
        
        confidences = self.estimator.estimate_confidence_batch(texts)
        
        # Check that we get the right number of scores
        self.assertEqual(len(confidences), len(texts))
        
        # Check that all scores are finite floats
        for conf in confidences:
            self.assertIsInstance(conf, float)
            self.assertTrue(np.isfinite(conf))
        
        print(f"Batch confidences: {confidences}")
    
    def test_different_confidence_methods(self):
        """Test different confidence estimation methods."""
        text = "The capital of France is"
        methods = ["self_certainty", "entropy", "kl_uniform", "max_prob", "top_k"]
        
        results = {}
        for method in methods:
            confidence = self.estimator.estimate_confidence(text, method=method)
            results[method] = confidence
            
            # Check that we get a finite float
            self.assertIsInstance(confidence, float)
            self.assertTrue(np.isfinite(confidence))
        
        print("Method comparison:")
        for method, score in results.items():
            print(f"  {method}: {score:.4f}")
        
        # Verify that different methods give different results (in most cases)
        values = list(results.values())
        # At least some methods should give different results
        self.assertGreater(len(set([round(v, 3) for v in values])), 1)
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        text = "The capital of France is"
        aggregation_methods = ["mean", "last", "max", "min"]
        
        results = {}
        for agg in aggregation_methods:
            confidence = self.estimator.estimate_confidence(text, aggregate=agg)
            results[agg] = confidence
            
            # Check that we get a finite float
            self.assertIsInstance(confidence, float)
            self.assertTrue(np.isfinite(confidence))
        
        print("Aggregation comparison:")
        for agg, score in results.items():
            print(f"  {agg}: {score:.4f}")
    
    def test_token_level_confidence(self):
        """Test token-level confidence analysis."""
        text = "The capital of France"
        tokens, scores = self.estimator.get_token_level_confidence(text)
        
        # Check that we get equal number of tokens and scores
        self.assertEqual(len(tokens), len(scores))
        
        # Check that all scores are finite
        for score in scores:
            self.assertTrue(np.isfinite(score))
        
        print(f"Token-level analysis for '{text}':")
        for token, score in zip(tokens, scores):
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()
            print(f"  '{clean_token}': {score:.4f}")
    
    def test_confidence_with_generation(self):
        """Test confidence estimation with text generation."""
        prompt = "The future of AI"
        generated_texts, confidences = self.estimator.estimate_confidence_with_generation(
            prompt, max_length=20, num_samples=3
        )
        
        # Check that we get the right number of results
        self.assertEqual(len(generated_texts), 3)
        self.assertEqual(len(confidences), 3)
        
        # Check that all confidences are finite
        for conf in confidences:
            self.assertTrue(np.isfinite(conf))
        
        print(f"Generated text with confidence for '{prompt}':")
        for i, (text, conf) in enumerate(zip(generated_texts, confidences)):
            print(f"  {i+1}. [{conf:.4f}] {text.strip()}")
    
    def test_compare_methods(self):
        """Test method comparison functionality."""
        texts = ["The capital of France is", "2 + 2 equals"]
        results = self.estimator.compare_methods(texts)
        
        # Check that we get results for all default methods
        expected_methods = ["self_certainty", "entropy", "kl_uniform", "max_prob", "top_k"]
        for method in expected_methods:
            self.assertIn(method, results)
            self.assertEqual(len(results[method]), len(texts))
        
        print("Method comparison results:")
        for method, scores in results.items():
            print(f"  {method}: {scores}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test model."""
        cls.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        cls.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        cls.estimator = ConfidenceEstimator(cls.model, cls.tokenizer)
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace text."""
        # This might raise an error or handle gracefully - test the behavior
        try:
            confidence = self.estimator.estimate_confidence("")
            print(f"Empty string confidence: {confidence}")
            # If it doesn't raise an error, check that result is finite
            self.assertTrue(np.isfinite(confidence))
        except Exception as e:
            print(f"Empty string raises error (expected): {e}")
            # This is acceptable behavior
    
    def test_very_long_text(self):
        """Test handling of very long text (should be truncated)."""
        long_text = "The capital of France is Paris. " * 100
        confidence = self.estimator.estimate_confidence(long_text, max_length=50)
        
        # Should handle gracefully
        self.assertIsInstance(confidence, float)
        self.assertTrue(np.isfinite(confidence))
        print(f"Long text confidence: {confidence:.4f}")
    
    def test_invalid_method(self):
        """Test error handling for invalid methods."""
        with self.assertRaises(ValueError):
            self.estimator.estimate_confidence("Test", method="invalid_method")
    
    def test_invalid_aggregation(self):
        """Test error handling for invalid aggregation methods."""
        with self.assertRaises(ValueError):
            self.estimator.estimate_confidence("Test", aggregate="invalid_agg")
    
    def test_batch_with_mixed_lengths(self):
        """Test batch processing with texts of different lengths."""
        texts = [
            "Short",
            "This is a medium length text",
            "This is a much longer text that should test the padding and batching functionality properly"
        ]
        
        confidences = self.estimator.estimate_confidence_batch(texts)
        
        # Should handle gracefully
        self.assertEqual(len(confidences), len(texts))
        for conf in confidences:
            self.assertIsInstance(conf, float)
            self.assertTrue(np.isfinite(conf))
        
        print(f"Mixed length batch confidences: {confidences}")


class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of confidence measures."""
    
    def setUp(self):
        """Set up test data."""
        torch.manual_seed(42)
        # Create logits with known properties
        self.vocab_size = 100
        
        # Very peaked distribution (high confidence)
        self.peaked_logits = torch.zeros(1, 1, self.vocab_size)
        self.peaked_logits[0, 0, 0] = 10.0  # Very high value for first token
        
        # Uniform-like distribution (low confidence)
        self.uniform_logits = torch.zeros(1, 1, self.vocab_size)
        
        # Random distribution
        self.random_logits = torch.randn(1, 1, self.vocab_size)
    
    def test_self_certainty_properties(self):
        """Test properties of self-certainty measure."""
        # Peaked distribution should have higher self-certainty than uniform
        peaked_certainty = self_certainty_from_logits(self.peaked_logits)
        uniform_certainty = self_certainty_from_logits(self.uniform_logits)
        
        print(f"Peaked certainty: {peaked_certainty.item():.4f}")
        print(f"Uniform certainty: {uniform_certainty.item():.4f}")
        
        # Peaked should be higher than uniform
        self.assertGreater(peaked_certainty.item(), uniform_certainty.item())
    
    def test_entropy_properties(self):
        """Test properties of entropy measure."""
        # Uniform distribution should have higher entropy than peaked
        peaked_entropy = entropy_from_logits(self.peaked_logits)
        uniform_entropy = entropy_from_logits(self.uniform_logits)
        
        print(f"Peaked entropy: {peaked_entropy.item():.4f}")
        print(f"Uniform entropy: {uniform_entropy.item():.4f}")
        
        # Uniform should have higher entropy (more uncertainty)
        self.assertGreater(uniform_entropy.item(), peaked_entropy.item())
    
    def test_kl_divergence_properties(self):
        """Test properties of KL divergence from uniform."""
        # Peaked distribution should have higher KL divergence from uniform
        peaked_kl = kl_divergence_from_uniform(self.peaked_logits)
        uniform_kl = kl_divergence_from_uniform(self.uniform_logits)
        
        print(f"Peaked KL from uniform: {peaked_kl.item():.4f}")
        print(f"Uniform KL from uniform: {uniform_kl.item():.4f}")
        
        # Peaked should be farther from uniform
        self.assertGreater(peaked_kl.item(), uniform_kl.item())
        
        # KL divergence should be non-negative
        self.assertGreaterEqual(peaked_kl.item(), 0)
        self.assertGreaterEqual(uniform_kl.item(), 0)


def run_tests():
    """Run all tests and provide a summary."""
    print("=" * 70)
    print("RUNNING COMPREHENSIVE TESTS FOR CONFIDENCE LLM PACKAGE")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUtilityFunctions,
        TestConfidenceEstimator,
        TestEdgeCases,
        TestMathematicalProperties
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL TESTS PASSED! The package is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED. Please check the output above.")
    
    return success


if __name__ == "__main__":
    run_tests() 