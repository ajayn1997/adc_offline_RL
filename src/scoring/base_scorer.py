"""
Base class for data scoring functions.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseScorer(ABC):
    """
    Abstract base class for data scoring functions.
    
    Scoring functions assign utility scores to transitions in the offline dataset
    to guide curriculum construction.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize scorer.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
    
    @abstractmethod
    def score_batch(self, 
                   observations: np.ndarray,
                   actions: np.ndarray,
                   rewards: np.ndarray,
                   next_observations: np.ndarray,
                   terminals: np.ndarray,
                   q_function: Any) -> np.ndarray:
        """
        Score a batch of transitions.
        
        Args:
            observations: Batch of observations [batch_size, obs_dim]
            actions: Batch of actions [batch_size, action_dim]
            rewards: Batch of rewards [batch_size]
            next_observations: Batch of next observations [batch_size, obs_dim]
            terminals: Batch of terminal flags [batch_size]
            q_function: Q-function for scoring
        
        Returns:
            Array of scores for each transition [batch_size]
        """
        pass
    
    def score_dataset(self, 
                     dataset: Dict[str, np.ndarray], 
                     q_function: Any,
                     batch_size: int = 1000) -> np.ndarray:
        """
        Score all transitions in a dataset.
        
        Args:
            dataset: Dataset dictionary
            q_function: Q-function for scoring
            batch_size: Batch size for processing
        
        Returns:
            Array of scores for all transitions
        """
        num_transitions = len(dataset['observations'])
        scores = np.zeros(num_transitions)
        
        for i in range(0, num_transitions, batch_size):
            end_idx = min(i + batch_size, num_transitions)
            
            batch_scores = self.score_batch(
                observations=dataset['observations'][i:end_idx],
                actions=dataset['actions'][i:end_idx],
                rewards=dataset['rewards'][i:end_idx],
                next_observations=dataset['next_observations'][i:end_idx],
                terminals=dataset['terminals'][i:end_idx],
                q_function=q_function
            )
            
            scores[i:end_idx] = batch_scores
        
        return scores
    
    def get_score_statistics(self, scores: np.ndarray) -> Dict[str, float]:
        """
        Get statistics about the computed scores.
        
        Args:
            scores: Array of scores
        
        Returns:
            Dictionary with score statistics
        """
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75))
        }
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the scoring function."""
        pass

