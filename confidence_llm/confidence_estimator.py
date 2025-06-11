"""
Main ConfidenceEstimator class for measuring language model confidence.

This module implements the ConfidenceEstimator class which provides methods
for estimating model confidence using various metrics including self-certainty.
"""

import torch
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import numpy as np

from .utils import (
    self_certainty_from_logits,
    entropy_from_logits,
    kl_divergence_from_uniform,
    masked_mean,
    max_probability_confidence,
    top_k_confidence,
)


class ConfidenceEstimator:
    """
    Estimate language model confidence using various metrics.
    
    This class provides methods to measure how confident a language model is
    in its predictions using different approaches, with the primary method
    being self-certainty from the Intuitor paper.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        """
        Initialize the ConfidenceEstimator.
        
        Args:
            model: HuggingFace causal language model
            tokenizer: HuggingFace tokenizer corresponding to the model
            device: Device to run the model on ('cuda' or 'cpu'). If None, auto-detect.
            batch_size: Default batch size for batch processing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_inputs(self, texts: Union[str, List[str]], max_length: int = 50) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.
        
        Args:
            texts: Input text(s) to process
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing input_ids and attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize the inputs
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        
        # Move to device
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _get_model_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get logits from the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for the sequences
            
        Returns:
            Model logits of shape (batch_size, seq_len, vocab_size)
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    def estimate_confidence(
        self,
        text: str,
        max_length: int = 50,
        method: str = "self_certainty",
        aggregate: str = "mean"
    ) -> float:
        """
        Estimate confidence for a single text.
        
        Args:
            text: Input text to analyze
            max_length: Maximum sequence length
            method: Confidence estimation method ('self_certainty', 'entropy', 'kl_uniform', 'max_prob', 'top_k')
            aggregate: How to aggregate token-level confidences ('mean', 'last', 'max', 'min')
            
        Returns:
            Confidence score as a float
        """
        scores = self.estimate_confidence_batch(
            [text], max_length=max_length, method=method, aggregate=aggregate
        )
        return scores[0]

    def estimate_confidence_batch(
        self,
        texts: List[str],
        max_length: int = 50,
        method: str = "self_certainty",
        aggregate: str = "mean",
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[float]:
        """
        Estimate confidence for multiple texts efficiently.
        
        Args:
            texts: List of input texts to analyze
            max_length: Maximum sequence length
            method: Confidence estimation method
            aggregate: How to aggregate token-level confidences
            batch_size: Batch size for processing (uses default if None)
            show_progress: Whether to show progress bar
            
        Returns:
            List of confidence scores
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        all_scores = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc="Computing confidence"):
            batch_texts = texts[i:i + batch_size]
            inputs = self._prepare_inputs(batch_texts, max_length)
            
            # Get logits
            logits = self._get_model_logits(inputs["input_ids"], inputs["attention_mask"])
            
            # Compute confidence scores
            token_scores = self._compute_confidence_scores(logits, method=method)
            
            # Aggregate scores
            batch_scores = self._aggregate_scores(
                token_scores, inputs["attention_mask"], aggregate=aggregate
            )
            
            all_scores.extend(batch_scores.cpu().numpy().tolist())
            
        return all_scores

    def _compute_confidence_scores(self, logits: torch.Tensor, method: str = "self_certainty") -> torch.Tensor:
        """
        Compute confidence scores from logits using the specified method.
        
        Args:
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            method: Confidence estimation method
            
        Returns:
            Token-level confidence scores of shape (batch_size, seq_len)
        """
        if method == "self_certainty":
            return self_certainty_from_logits(logits, dim=-1)
        elif method == "entropy":
            return -entropy_from_logits(logits, dim=-1)  # Negative because lower entropy = higher confidence
        elif method == "kl_uniform":
            return kl_divergence_from_uniform(logits, dim=-1)
        elif method == "max_prob":
            return max_probability_confidence(logits, dim=-1)
        elif method == "top_k":
            return top_k_confidence(logits, k=5, dim=-1)
        else:
            raise ValueError(f"Unknown confidence method: {method}")

    def _aggregate_scores(
        self,
        scores: torch.Tensor,
        attention_mask: torch.Tensor,
        aggregate: str = "mean"
    ) -> torch.Tensor:
        """
        Aggregate token-level confidence scores.
        
        Args:
            scores: Token-level confidence scores of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            aggregate: Aggregation method
            
        Returns:
            Aggregated confidence scores of shape (batch_size,)
        """
        if aggregate == "mean":
            return masked_mean(scores, attention_mask.float(), dim=1)
        elif aggregate == "last":
            # Get the last valid token for each sequence
            last_indices = attention_mask.sum(dim=1) - 1
            return scores[torch.arange(scores.size(0)), last_indices]
        elif aggregate == "max":
            # Set masked positions to very low values before taking max
            masked_scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
            return masked_scores.max(dim=1)[0]
        elif aggregate == "min":
            # Set masked positions to very high values before taking min
            masked_scores = scores.masked_fill(~attention_mask.bool(), float('inf'))
            return masked_scores.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

    def estimate_confidence_with_generation(
        self,
        text: str,
        max_length: int = 50,
        num_samples: int = 5,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Generate text and compute confidence for each generated sequence.
        
        Args:
            text: Input prompt text
            max_length: Maximum length for generation
            num_samples: Number of sequences to generate
            generation_kwargs: Additional arguments for generation
            
        Returns:
            Tuple of (generated_texts, confidence_scores)
        """
        if generation_kwargs is None:
            generation_kwargs = {}
            
        # Default generation parameters
        default_kwargs = {
            "max_length": max_length,
            "num_return_sequences": num_samples,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        default_kwargs.update(generation_kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate sequences
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **default_kwargs
            )
        
        # Decode generated texts
        generated_texts = []
        for ids in generated_ids:
            # Remove the input prompt from the generated sequence
            generated_part = ids[inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        # Compute confidence for each generated sequence
        confidence_scores = []
        for ids in generated_ids:
            # Create attention mask (all 1s for generated sequences)
            attention_mask = torch.ones_like(ids).to(self.device)
            
            # Get logits for the full sequence
            with torch.no_grad():
                logits = self.model(input_ids=ids.unsqueeze(0)).logits
            
            # Focus on the generated part (excluding the prompt)
            generated_logits = logits[0, inputs.input_ids.shape[1]-1:-1]  # -1 to align with targets
            generated_mask = attention_mask[inputs.input_ids.shape[1]:]
            
            # Compute confidence
            token_scores = self_certainty_from_logits(generated_logits, dim=-1)
            confidence = masked_mean(
                token_scores.unsqueeze(0),
                generated_mask.float().unsqueeze(0),
                dim=1
            ).item()
            confidence_scores.append(confidence)
        
        return generated_texts, confidence_scores

    def compare_methods(
        self,
        texts: Union[str, List[str]],
        max_length: int = 50,
        methods: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Compare different confidence estimation methods on the same texts.
        
        Args:
            texts: Input text(s) to analyze
            max_length: Maximum sequence length
            methods: List of methods to compare (uses all if None)
            
        Returns:
            Dictionary mapping method names to lists of confidence scores
        """
        if methods is None:
            methods = ["self_certainty", "entropy", "kl_uniform", "max_prob", "top_k"]
            
        if isinstance(texts, str):
            texts = [texts]
            
        results = {}
        for method in methods:
            scores = self.estimate_confidence_batch(
                texts, max_length=max_length, method=method
            )
            results[method] = scores
            
        return results

    def get_token_level_confidence(
        self,
        text: str,
        max_length: int = 50,
        method: str = "self_certainty"
    ) -> Tuple[List[str], List[float]]:
        """
        Get token-level confidence scores for detailed analysis.
        
        Args:
            text: Input text to analyze
            max_length: Maximum sequence length
            method: Confidence estimation method
            
        Returns:
            Tuple of (tokens, confidence_scores)
        """
        inputs = self._prepare_inputs(text, max_length)
        logits = self._get_model_logits(inputs["input_ids"], inputs["attention_mask"])
        
        # Compute token-level scores
        token_scores = self._compute_confidence_scores(logits, method=method)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Filter out padding tokens
        attention_mask = inputs["attention_mask"][0]
        valid_indices = attention_mask.bool()
        
        valid_tokens = [tokens[i] for i in range(len(tokens)) if valid_indices[i]]
        valid_scores = token_scores[0][valid_indices].cpu().numpy().tolist()
        
        return valid_tokens, valid_scores 