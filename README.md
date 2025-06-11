# Confidence LLM

A simple Python package for measuring language model confidence using KL divergence, inspired by the Intuitor paper's self-certainty approach.

## Overview

This package implements model confidence measurement using the self-certainty method from the paper ["Learning to Reason without External Rewards"](https://arxiv.org/abs/2505.19590). The approach measures how confident a model is in its predictions by computing the KL divergence between the model's output distribution and a uniform distribution.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

For examples with visualization:
```bash
pip install -e ".[examples]"
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from confidence_llm import ConfidenceEstimator

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create confidence estimator
confidence_estimator = ConfidenceEstimator(model, tokenizer)

# Measure confidence for a single text
text = "The capital of France is"
confidence_score = confidence_estimator.estimate_confidence(text)
print(f"Confidence score: {confidence_score:.4f}")

# Measure confidence for multiple texts
texts = [
    "The capital of France is",
    "The meaning of life is",
    "2 + 2 equals"
]
confidence_scores = confidence_estimator.estimate_confidence_batch(texts)
for text, score in zip(texts, confidence_scores):
    print(f"'{text}' -> Confidence: {score:.4f}")
```

## Key Features

- **Self-Certainty Measurement**: Implements the self-certainty approach using KL divergence
- **Batch Processing**: Efficient batch processing for multiple texts
- **Flexible Input**: Supports both single texts and lists of texts
- **GPU Support**: Automatic GPU utilization when available
- **HuggingFace Integration**: Works seamlessly with HuggingFace transformers

## How It Works

The confidence measurement is based on the self-certainty formula:

```
self_certainty = logsumexp(logits) - mean(logits)
```

This measures how far the model's output distribution is from a uniform distribution:
- **Higher values** indicate higher confidence (the model is more certain)
- **Lower values** indicate lower confidence (the model is less certain)

## API Reference

### ConfidenceEstimator

Main class for estimating model confidence.

#### Methods

- `estimate_confidence(text: str, max_length: int = 50) -> float`
  - Estimate confidence for a single text
  
- `estimate_confidence_batch(texts: List[str], max_length: int = 50, batch_size: int = 8) -> List[float]`
  - Estimate confidence for multiple texts efficiently
  
- `estimate_confidence_with_generation(text: str, max_length: int = 50, num_samples: int = 5) -> Tuple[List[str], List[float]]`
  - Generate text and compute confidence for each generated sequence

## Examples

See the `examples/` directory for more comprehensive examples:

- `basic_usage.py`: Basic confidence estimation
- `batch_processing.py`: Efficient batch processing
- `confidence_analysis.py`: Analyzing confidence patterns
- `compare_models.py`: Comparing confidence across different models

## Citation

This package is inspired by the Intuitor paper:

```bibtex
@article{zhao2025learning,
  title={Learning to Reason without External Rewards},
  author={Zhao, Xuandong and Kang, Zhewei and Feng, Aosong and Levine, Sergey and Song, Dawn},
  journal={arXiv preprint arXiv:2505.19590},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details. 