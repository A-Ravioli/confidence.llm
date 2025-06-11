#!/usr/bin/env python3
"""
Confidence analysis example for the Confidence LLM package.

This example demonstrates advanced confidence analysis including correlation
analysis, confidence patterns, and insights into model behavior.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from confidence_llm import ConfidenceEstimator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def main():
    """Main function for confidence analysis."""
    
    print("Starting comprehensive confidence analysis...")
    
    # Load model
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    estimator = ConfidenceEstimator(model, tokenizer)
    
    print(f"Using model: {model_name}")
    print(f"Device: {estimator.device}")
    print()
    
    # Run various analyses
    analyze_confidence_by_prompt_length(estimator)
    analyze_confidence_by_domain(estimator)
    analyze_confidence_correlation(estimator)
    analyze_token_level_patterns(estimator)
    analyze_generation_confidence(estimator)


def analyze_confidence_by_prompt_length(estimator):
    """Analyze how confidence varies with prompt length."""
    print("=" * 60)
    print("ANALYSIS 1: Confidence vs Prompt Length")
    print("=" * 60)
    
    base_prompts = [
        "Paris",
        "The capital of France",
        "The capital of France is located",
        "The capital of France is located in the heart of",
        "The capital of France is located in the heart of Europe and"
    ]
    
    lengths = []
    confidences = []
    
    for prompt in base_prompts:
        confidence = estimator.estimate_confidence(prompt)
        token_count = len(estimator.tokenizer.encode(prompt))
        
        lengths.append(token_count)
        confidences.append(confidence)
        
        print(f"Length: {token_count:2d} tokens | Confidence: {confidence:.4f} | '{prompt}'")
    
    # Calculate correlation
    correlation = np.corrcoef(lengths, confidences)[0, 1]
    print(f"\nCorrelation between length and confidence: {correlation:.4f}")
    print()


def analyze_confidence_by_domain(estimator):
    """Analyze confidence across different knowledge domains."""
    print("=" * 60)
    print("ANALYSIS 2: Confidence by Knowledge Domain")
    print("=" * 60)
    
    domain_prompts = {
        "Geography": [
            "The capital of France is",
            "The largest country in the world is",
            "Mount Everest is located in",
            "The Nile River flows through"
        ],
        "Mathematics": [
            "2 + 2 equals",
            "The square root of 16 is",
            "Pi is approximately",
            "10 factorial is"
        ],
        "Science": [
            "The speed of light is",
            "Water boils at",
            "DNA stands for",
            "The periodic table has"
        ],
        "History": [
            "World War II ended in",
            "The American Civil War began in",
            "Napoleon was defeated at",
            "The Renaissance started in"
        ],
        "Literature": [
            "Shakespeare wrote",
            "The author of 1984 is",
            "Moby Dick was written by",
            "Pride and Prejudice is by"
        ]
    }
    
    domain_results = {}
    
    for domain, prompts in domain_prompts.items():
        confidences = estimator.estimate_confidence_batch(prompts)
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        domain_results[domain] = {
            'mean': mean_confidence,
            'std': std_confidence,
            'scores': confidences
        }
        
        print(f"{domain:<12}: Mean={mean_confidence:.4f}, Std={std_confidence:.4f}")
    
    # Find most and least confident domains
    best_domain = max(domain_results.keys(), key=lambda k: domain_results[k]['mean'])
    worst_domain = min(domain_results.keys(), key=lambda k: domain_results[k]['mean'])
    
    print(f"\nMost confident domain: {best_domain} ({domain_results[best_domain]['mean']:.4f})")
    print(f"Least confident domain: {worst_domain} ({domain_results[worst_domain]['mean']:.4f})")
    print()


def analyze_confidence_correlation(estimator):
    """Analyze correlation between different confidence methods."""
    print("=" * 60)
    print("ANALYSIS 3: Correlation Between Confidence Methods")
    print("=" * 60)
    
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "The weather today",
        "2 + 2 equals",
        "Shakespeare wrote",
        "The internet was invented",
        "Climate change is",
        "Python programming",
        "The human brain",
        "Quantum physics"
    ]
    
    methods = ["self_certainty", "entropy", "kl_uniform", "max_prob", "top_k"]
    method_scores = {}
    
    for method in methods:
        scores = estimator.estimate_confidence_batch(test_prompts, method=method)
        method_scores[method] = scores
    
    # Calculate correlation matrix
    print("Correlation matrix between confidence methods:")
    print("Method          ", end="")
    for method in methods:
        print(f"{method[:8]:>8}", end="")
    print()
    
    for method1 in methods:
        print(f"{method1:<15} ", end="")
        for method2 in methods:
            corr = np.corrcoef(method_scores[method1], method_scores[method2])[0, 1]
            print(f"{corr:>8.3f}", end="")
        print()
    print()


def analyze_token_level_patterns(estimator):
    """Analyze patterns in token-level confidence."""
    print("=" * 60)
    print("ANALYSIS 4: Token-level Confidence Patterns")
    print("=" * 60)
    
    test_sentences = [
        "The capital of France is Paris",
        "Two plus two equals four",
        "Shakespeare wrote many famous plays"
    ]
    
    for sentence in test_sentences:
        tokens, scores = estimator.get_token_level_confidence(sentence)
        
        print(f"\nSentence: '{sentence}'")
        print("Position | Token                | Confidence")
        print("-" * 50)
        
        for i, (token, score) in enumerate(zip(tokens, scores)):
            # Clean up token display
            display_token = token.replace('Ġ', ' ').replace('▁', ' ')
            if display_token.startswith(' '):
                display_token = display_token[1:]
            print(f"{i:>8} | {display_token:<20} | {score:>10.4f}")
        
        # Analyze patterns
        confidence_trend = np.diff(scores)
        avg_trend = np.mean(confidence_trend)
        
        if avg_trend > 0.01:
            trend_desc = "increasing"
        elif avg_trend < -0.01:
            trend_desc = "decreasing"
        else:
            trend_desc = "stable"
            
        print(f"   Average confidence: {np.mean(scores):.4f}")
        print(f"   Confidence trend: {trend_desc} (Δ={avg_trend:.4f})")
    print()


def analyze_generation_confidence(estimator):
    """Analyze confidence in text generation."""
    print("=" * 60)
    print("ANALYSIS 5: Confidence in Text Generation")
    print("=" * 60)
    
    prompts = [
        "The future of artificial intelligence",
        "Climate change will",
        "The most important invention"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated_texts, confidences = estimator.estimate_confidence_with_generation(
            prompt, max_length=25, num_samples=5
        )
        
        # Sort by confidence
        sorted_results = sorted(zip(generated_texts, confidences), 
                              key=lambda x: x[1], reverse=True)
        
        print("Generated completions (sorted by confidence):")
        for i, (text, confidence) in enumerate(sorted_results, 1):
            print(f"  {i}. [{confidence:.4f}] {text.strip()}")
        
        # Analyze variance in confidence
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
        conf_range = max(confidences) - min(confidences)
        
        print(f"  Confidence stats: Mean={conf_mean:.4f}, Std={conf_std:.4f}, Range={conf_range:.4f}")


if __name__ == "__main__":
    main() 