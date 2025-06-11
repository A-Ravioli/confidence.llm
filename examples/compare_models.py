#!/usr/bin/env python3
"""
Model comparison example for the Confidence LLM package.

This example demonstrates how to compare confidence scores across different models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from confidence_llm import ConfidenceEstimator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_model_and_estimator(model_name: str):
    """Load a model and create a confidence estimator."""
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    estimator = ConfidenceEstimator(model, tokenizer)
    return estimator


def main():
    """Main function for model comparison."""
    
    # Define models to compare (using smaller models for faster execution)
    models = [
        "distilgpt2",
        "gpt2",
        # Add more models as needed, e.g.:
        # "microsoft/DialoGPT-small",
        # "openai-gpt",
    ]
    
    # Test prompts covering different types of knowledge
    test_prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "The speed of light is",
        "Machine learning is",
        "The weather today",
        "Shakespeare wrote",
        "Python programming",
        "The human brain",
        "Climate change is",
        "The internet was"
    ]
    
    print("Comparing confidence across different models...")
    print(f"Models: {models}")
    print(f"Number of test prompts: {len(test_prompts)}")
    print()
    
    # Store results
    results = []
    
    # Test each model
    for model_name in models:
        try:
            estimator = load_model_and_estimator(model_name)
            
            # Get confidence scores for all prompts
            confidence_scores = estimator.estimate_confidence_batch(
                test_prompts, 
                show_progress=True
            )
            
            # Store results
            for prompt, score in zip(test_prompts, confidence_scores):
                results.append({
                    'model': model_name,
                    'prompt': prompt,
                    'confidence': score,
                    'prompt_category': categorize_prompt(prompt)
                })
            
            print(f"Completed {model_name}")
            print()
            
            # Clean up memory
            del estimator
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results to analyze!")
        return
    
    # Print summary statistics
    print("=" * 60)
    print("CONFIDENCE COMPARISON SUMMARY")
    print("=" * 60)
    
    # Overall statistics by model
    print("\nOverall Statistics by Model:")
    model_stats = df.groupby('model')['confidence'].agg(['mean', 'std', 'min', 'max'])
    print(model_stats.round(4))
    
    # Statistics by prompt category
    print("\nStatistics by Prompt Category:")
    category_stats = df.groupby(['model', 'prompt_category'])['confidence'].mean().unstack()
    print(category_stats.round(4))
    
    # Detailed comparison for each prompt
    print("\nDetailed Comparison by Prompt:")
    prompt_comparison = df.pivot(index='prompt', columns='model', values='confidence')
    print(prompt_comparison.round(4))
    
    # Create visualizations (if matplotlib is available)
    try:
        create_visualizations(df)
        print("\nVisualizations saved as 'confidence_comparison.png'")
    except ImportError:
        print("\nSkipping visualizations (matplotlib not available)")
    except Exception as e:
        print(f"\nError creating visualizations: {e}")


def categorize_prompt(prompt: str) -> str:
    """Categorize prompts for analysis."""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['capital', 'country', 'geography']):
        return 'Geography'
    elif any(word in prompt_lower for word in ['math', '+', 'equals', 'speed', 'light']):
        return 'Math/Science'
    elif any(word in prompt_lower for word in ['shakespeare', 'wrote', 'literature']):
        return 'Literature'
    elif any(word in prompt_lower for word in ['machine learning', 'programming', 'python', 'internet']):
        return 'Technology'
    elif any(word in prompt_lower for word in ['weather', 'climate']):
        return 'Weather/Climate'
    elif any(word in prompt_lower for word in ['brain', 'human']):
        return 'Biology'
    else:
        return 'General'


def create_visualizations(df: pd.DataFrame):
    """Create visualizations for confidence comparison."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Confidence Comparison', fontsize=16, fontweight='bold')
    
    # 1. Box plot of confidence scores by model
    sns.boxplot(data=df, x='model', y='confidence', ax=axes[0, 0])
    axes[0, 0].set_title('Confidence Distribution by Model')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Heatmap of confidence by model and category
    heatmap_data = df.groupby(['model', 'prompt_category'])['confidence'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 1])
    axes[0, 1].set_title('Average Confidence by Model and Category')
    
    # 3. Bar plot comparing models
    model_means = df.groupby('model')['confidence'].mean().sort_values(ascending=False)
    model_means.plot(kind='bar', ax=axes[1, 0], color='skyblue')
    axes[1, 0].set_title('Average Confidence by Model')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylabel('Average Confidence')
    
    # 4. Scatter plot of confidence scores
    for i, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]
        axes[1, 1].scatter(range(len(model_data)), model_data['confidence'], 
                          label=model, alpha=0.7)
    axes[1, 1].set_title('Confidence Scores by Prompt')
    axes[1, 1].set_xlabel('Prompt Index')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('confidence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main() 