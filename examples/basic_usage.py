#!/usr/bin/env python3
"""
Basic usage example for the Confidence LLM package.

This example demonstrates how to use the ConfidenceEstimator to measure
how confident a language model is in its predictions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from confidence_llm import ConfidenceEstimator


def main():
    """Main function demonstrating basic usage."""
    
    print("Loading model and tokenizer...")
    # Load a small model for demonstration (you can use any causal LM)
    model_name = "gpt2"  # or "microsoft/DialoGPT-small", "distilgpt2", etc.
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create confidence estimator
    confidence_estimator = ConfidenceEstimator(model, tokenizer)
    
    print(f"Using device: {confidence_estimator.device}")
    print()
    
    # Example 1: Single text confidence
    print("=" * 50)
    print("Example 1: Single Text Confidence")
    print("=" * 50)
    
    text = "The capital of France is"
    confidence_score = confidence_estimator.estimate_confidence(text)
    print(f"Text: '{text}'")
    print(f"Confidence score: {confidence_score:.4f}")
    print()
    
    # Example 2: Multiple texts
    print("=" * 50)
    print("Example 2: Multiple Texts")
    print("=" * 50)
    
    texts = [
        "The capital of France is",
        "The meaning of life is",
        "2 + 2 equals",
        "The weather today will be",
        "Machine learning is"
    ]
    
    confidence_scores = confidence_estimator.estimate_confidence_batch(
        texts, show_progress=True
    )
    
    print("Confidence scores:")
    for text, score in zip(texts, confidence_scores):
        print(f"  '{text:<25}' -> {score:.4f}")
    print()
    
    # Example 3: Different confidence methods
    print("=" * 50)
    print("Example 3: Comparing Different Methods")
    print("=" * 50)
    
    test_text = "The capital of France is"
    methods = ["self_certainty", "entropy", "kl_uniform", "max_prob", "top_k"]
    
    print(f"Text: '{test_text}'")
    print("Method comparison:")
    
    for method in methods:
        score = confidence_estimator.estimate_confidence(test_text, method=method)
        print(f"  {method:<15}: {score:.4f}")
    print()
    
    # Example 4: Token-level confidence
    print("=" * 50)
    print("Example 4: Token-level Analysis")
    print("=" * 50)
    
    tokens, token_scores = confidence_estimator.get_token_level_confidence(test_text)
    
    print(f"Token-level confidence for: '{test_text}'")
    print("Token                 | Confidence")
    print("-" * 35)
    for token, score in zip(tokens, token_scores):
        # Clean up token representation
        display_token = token.replace('Ġ', ' ').replace('▁', ' ')
        if display_token.startswith(' '):
            display_token = display_token[1:]
        print(f"{display_token:<20} | {score:>8.4f}")
    print()
    
    # Example 5: Confidence with text generation
    print("=" * 50)
    print("Example 5: Confidence with Generation")
    print("=" * 50)
    
    prompt = "The future of artificial intelligence"
    generated_texts, confidence_scores = confidence_estimator.estimate_confidence_with_generation(
        prompt, max_length=30, num_samples=3
    )
    
    print(f"Prompt: '{prompt}'")
    print("Generated completions with confidence:")
    for i, (text, score) in enumerate(zip(generated_texts, confidence_scores), 1):
        print(f"  {i}. {text.strip():<40} (confidence: {score:.4f})")


if __name__ == "__main__":
    main() 