"""
Base class for Active Data Curation (ADC) framework.

This module provides the abstract base class that defines the interface
for both Static and Dynamic ADC implementations.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ADCConfig:
    """Configuration for ADC framework."""
    num_stages: int = 5  # K in the paper
    total_steps: int = 1_000_000
    scoring_method: str = "advantage"  # "advantage" or "td_error"
    pretrain_steps: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class ADCBase(ABC):
    """
    Abstract base class for Active Data Curation framework.
    
    This class defines the interface that both Static and Dynamic ADC
    implementations must follow. It provides common functionality for
    data scoring and curriculum construction.
    """
    
    def __init__(self, 
                 base_algorithm: Any,
                 config: ADCConfig,
                 scorer: Any = None):
        """
        Initialize ADC framework.
        
        Args:
            base_algorithm: The base offline RL algorithm (CQL, IQL, BC)
            config: ADC configuration
            scorer: Data scoring function (will be created if None)
        """
        self.base_algorithm = base_algorithm
        self.config = config
        self.scorer = scorer
        self.dataset = None
        self.sorted_indices = None
        self.stage_datasets = {}
        
        # Set random seeds for reproducibility
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    def set_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        """
        Set the offline dataset for training.
        
        Args:
            dataset: Dictionary containing 'observations', 'actions', 
                    'rewards', 'next_observations', 'terminals'
        """
        self.dataset = dataset
        self.dataset_size = len(dataset['observations'])
        print(f"Dataset loaded with {self.dataset_size} transitions")
    
    def score_transitions(self, q_function: Any = None) -> np.ndarray:
        """
        Score all transitions in the dataset.
        
        Args:
            q_function: Q-function for scoring (if None, uses base algorithm's Q-function)
            
        Returns:
            Array of scores for each transition
        """
        if self.dataset is None:
            raise ValueError("Dataset must be set before scoring transitions")
        
        if q_function is None:
            q_function = self.base_algorithm.q_function
        
        scores = np.zeros(self.dataset_size)
        
        # Score transitions in batches for efficiency
        batch_size = 1000
        for i in range(0, self.dataset_size, batch_size):
            end_idx = min(i + batch_size, self.dataset_size)
            batch_scores = self._score_batch(
                i, end_idx, q_function
            )
            # Ensure batch_scores is 1D
            if len(batch_scores.shape) > 1:
                batch_scores = batch_scores.flatten()
            scores[i:end_idx] = batch_scores[:end_idx-i]  # Handle size mismatch
        
        return scores
    
    def _score_batch(self, start_idx: int, end_idx: int, q_function: Any) -> np.ndarray:
        """Score a batch of transitions."""
        batch_size = end_idx - start_idx
        
        # Extract batch data
        obs = self.dataset['observations'][start_idx:end_idx]
        actions = self.dataset['actions'][start_idx:end_idx]
        rewards = self.dataset['rewards'][start_idx:end_idx]
        next_obs = self.dataset['next_observations'][start_idx:end_idx]
        terminals = self.dataset['terminals'][start_idx:end_idx]
        
        if self.config.scoring_method == "td_error":
            return self._compute_td_error_scores(
                obs, actions, rewards, next_obs, terminals, q_function
            )
        elif self.config.scoring_method == "advantage":
            return self._compute_advantage_scores(
                obs, actions, q_function
            )
        else:
            raise ValueError(f"Unknown scoring method: {self.config.scoring_method}")
    
    def _compute_td_error_scores(self, obs, actions, rewards, next_obs, terminals, q_function):
        """Compute TD-error based scores."""
        with torch.no_grad():
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs).to(self.config.device)
            actions_tensor = torch.FloatTensor(actions).to(self.config.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.config.device)
            next_obs_tensor = torch.FloatTensor(next_obs).to(self.config.device)
            terminals_tensor = torch.FloatTensor(terminals).to(self.config.device)
            
            # Compute current Q-values
            current_q = q_function(obs_tensor, actions_tensor)
            
            # Compute target Q-values
            if hasattr(q_function, 'target_q_function'):
                target_q_func = q_function.target_q_function
            else:
                target_q_func = q_function
            
            # Get max Q-value for next state
            next_q_values = target_q_func(next_obs_tensor)
            if len(next_q_values.shape) > 1:
                next_q_max = torch.max(next_q_values, dim=1)[0]
            else:
                next_q_max = next_q_values
            
            # Compute TD target
            gamma = 0.99  # Standard discount factor
            td_target = rewards_tensor + gamma * (1 - terminals_tensor) * next_q_max
            
            # Compute TD error
            td_error = torch.abs(td_target - current_q.squeeze())
            
            return td_error.cpu().numpy()
    
    def _compute_advantage_scores(self, obs, actions, q_function):
        """Compute advantage-based scores."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.config.device)
            actions_tensor = torch.FloatTensor(actions).to(self.config.device)
            
            # Compute Q(s,a)
            q_values = q_function(obs_tensor, actions_tensor).squeeze()
            
            # Compute V(s) as expectation over behavior policy
            # For simplicity, we approximate with mean over action dimension
            if hasattr(q_function, 'compute_value'):
                state_values = q_function.compute_value(obs_tensor)
            else:
                # Approximate V(s) by sampling actions and averaging Q-values
                num_samples = 10
                sampled_q_values = []
                for _ in range(num_samples):
                    # Sample random actions (approximating behavior policy)
                    random_actions = torch.rand_like(actions_tensor)
                    sampled_q = q_function(obs_tensor, random_actions)
                    sampled_q_values.append(sampled_q)
                state_values = torch.mean(torch.stack(sampled_q_values), dim=0).squeeze()
            
            # Compute advantage
            advantages = q_values - state_values
            
            return advantages.cpu().numpy()
    
    def create_stage_datasets(self, scores: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Create datasets for each curriculum stage based on scores.
        
        Args:
            scores: Array of scores for each transition
            
        Returns:
            Dictionary mapping stage number to indices of transitions
        """
        # Sort indices by scores (descending order)
        sorted_indices = np.argsort(scores)[::-1]
        
        stage_datasets = {}
        for stage in range(1, self.config.num_stages + 1):
            # Include top (stage/num_stages) * 100% of data
            num_transitions = int((stage / self.config.num_stages) * self.dataset_size)
            stage_datasets[stage] = sorted_indices[:num_transitions]
        
        return stage_datasets
    
    def get_stage_dataset(self, stage: int, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get dataset for a specific stage.
        
        Args:
            stage: Stage number
            indices: Indices of transitions to include
            
        Returns:
            Dictionary containing subset of original dataset
        """
        stage_dataset = {}
        for key in self.dataset.keys():
            stage_dataset[key] = self.dataset[key][indices]
        
        return stage_dataset
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the model using ADC curriculum.
        
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained policy.
        
        Args:
            env: Environment for evaluation
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        returns = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_return = 0
            done = False
            
            while not done:
                action = self.base_algorithm.predict(obs)
                obs, reward, done, _ = env.step(action)
                episode_return += reward
            
            returns.append(episode_return)
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns)
        }

