"""
TD-error based scoring function.

This module implements the TD-error scoring function as described in the paper:
C_TD(s,a,r,s') = |r + γ max_a' Q(s',a') - Q(s,a)|
"""

import numpy as np
import torch
from typing import Any
from .base_scorer import BaseScorer


class TDErrorScorer(BaseScorer):
    """
    TD-error based scoring function.
    
    The TD-error measures the "surprise" or inconsistency of a transition
    with respect to a given Q-function. Transitions with high TD-error
    represent events that the agent's value function does not model well,
    and thus they are valuable learning targets.
    
    Score: C_TD(s,a,r,s') = |r + γ max_a' Q(s',a') - Q(s,a)|
    """
    
    def __init__(self, gamma: float = 0.99, device: str = 'cpu'):
        """
        Initialize TD-error scorer.
        
        Args:
            gamma: Discount factor
            device: Device for computations
        """
        super().__init__(device)
        self.gamma = gamma
    
    def score_batch(self, 
                   observations: np.ndarray,
                   actions: np.ndarray,
                   rewards: np.ndarray,
                   next_observations: np.ndarray,
                   terminals: np.ndarray,
                   q_function: Any) -> np.ndarray:
        """
        Score a batch of transitions using TD-error.
        
        Args:
            observations: Current observations [batch_size, obs_dim]
            actions: Actions taken [batch_size, action_dim]
            rewards: Rewards received [batch_size]
            next_observations: Next observations [batch_size, obs_dim]
            terminals: Terminal flags [batch_size]
            q_function: Q-function for scoring
        
        Returns:
            TD-error scores [batch_size]
        """
        with torch.no_grad():
            # Convert to tensors
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            next_obs_tensor = torch.FloatTensor(next_observations).to(self.device)
            terminals_tensor = torch.FloatTensor(terminals.astype(float)).to(self.device)
            
            # Compute current Q-values: Q(s,a)
            current_q_values = q_function(obs_tensor, actions_tensor)
            if len(current_q_values.shape) > 1:
                current_q_values = current_q_values.squeeze(-1)
            
            # Compute target Q-values: r + γ max_a' Q(s',a')
            target_q_values = self._compute_target_q_values(
                next_obs_tensor, rewards_tensor, terminals_tensor, q_function
            )
            
            # Compute TD-error: |target - current|
            td_errors = torch.abs(target_q_values - current_q_values)
            
            return td_errors.cpu().numpy()
    
    def _compute_target_q_values(self, 
                                next_obs: torch.Tensor,
                                rewards: torch.Tensor,
                                terminals: torch.Tensor,
                                q_function: Any) -> torch.Tensor:
        """
        Compute target Q-values for TD-error calculation.
        
        Args:
            next_obs: Next observations
            rewards: Rewards
            terminals: Terminal flags
            q_function: Q-function
        
        Returns:
            Target Q-values
        """
        batch_size = next_obs.shape[0]
        action_dim = self._infer_action_dim(q_function, next_obs)
        
        # Sample multiple actions to approximate max_a' Q(s',a')
        num_action_samples = 10
        max_q_values = []
        
        for _ in range(num_action_samples):
            # Sample random actions (uniform in [-1, 1])
            random_actions = torch.rand(batch_size, action_dim).to(self.device) * 2 - 1
            
            # Compute Q-values for sampled actions
            q_values = q_function(next_obs, random_actions)
            if len(q_values.shape) > 1:
                q_values = q_values.squeeze(-1)
            
            max_q_values.append(q_values)
        
        # Take maximum over sampled actions
        max_q_values = torch.stack(max_q_values, dim=0)
        next_q_max = torch.max(max_q_values, dim=0)[0]
        
        # Compute target: r + γ * (1 - terminal) * max_a' Q(s',a')
        target_q = rewards + self.gamma * (1 - terminals) * next_q_max
        
        return target_q
    
    def _infer_action_dim(self, q_function: Any, obs: torch.Tensor) -> int:
        """
        Infer action dimension from Q-function.
        
        Args:
            q_function: Q-function
            obs: Sample observation
        
        Returns:
            Action dimension
        """
        # Try common action dimensions
        for action_dim in [1, 2, 3, 4, 6, 8]:
            try:
                sample_action = torch.zeros(1, action_dim).to(self.device)
                sample_obs = obs[:1]
                q_function(sample_obs, sample_action)
                return action_dim
            except:
                continue
        
        # Default fallback
        return 3
    
    @property
    def name(self) -> str:
        """Return scorer name."""
        return "TD-Error"
    
    def __str__(self) -> str:
        return f"TDErrorScorer(gamma={self.gamma})"

