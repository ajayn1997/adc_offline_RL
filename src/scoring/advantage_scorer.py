"""
Advantage-based scoring function.

This module implements the advantage-based scoring function as described in the paper:
C_Adv(s,a) = Q(s,a) - V(s)

Where V(s) is approximated by E_a~π_BC[Q(s,a)] (expectation over behavior cloning policy).
"""

import numpy as np
import torch
from typing import Any
from .base_scorer import BaseScorer


class AdvantageScorer(BaseScorer):
    """
    Advantage-based scoring function.
    
    The advantage function A(s,a) = Q(s,a) - V(s) indicates how much better
    a given action is compared to the average action according to the policy.
    Transitions with high advantages likely correspond to near-optimal or
    expert-like behavior within the dataset.
    
    Score: C_Adv(s,a) = Q(s,a) - V(s)
    where V(s) ≈ E_a~π_BC[Q(s,a)]
    """
    
    def __init__(self, 
                 num_action_samples: int = 10,
                 device: str = 'cpu'):
        """
        Initialize advantage scorer.
        
        Args:
            num_action_samples: Number of action samples for V(s) approximation
            device: Device for computations
        """
        super().__init__(device)
        self.num_action_samples = num_action_samples
    
    def score_batch(self, 
                   observations: np.ndarray,
                   actions: np.ndarray,
                   rewards: np.ndarray,
                   next_observations: np.ndarray,
                   terminals: np.ndarray,
                   q_function: Any) -> np.ndarray:
        """
        Score a batch of transitions using advantage.
        
        Args:
            observations: Current observations [batch_size, obs_dim]
            actions: Actions taken [batch_size, action_dim]
            rewards: Rewards received [batch_size] (not used for advantage)
            next_observations: Next observations [batch_size, obs_dim] (not used)
            terminals: Terminal flags [batch_size] (not used)
            q_function: Q-function for scoring
        
        Returns:
            Advantage scores [batch_size]
        """
        with torch.no_grad():
            # Convert to tensors
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            
            # Compute Q(s,a) for actual actions
            q_values = q_function(obs_tensor, actions_tensor)
            if len(q_values.shape) > 1:
                q_values = q_values.squeeze(-1)
            
            # Compute V(s) by sampling actions and averaging Q-values
            state_values = self._compute_state_values(obs_tensor, q_function)
            
            # Compute advantage: A(s,a) = Q(s,a) - V(s)
            advantages = q_values - state_values
            
            return advantages.cpu().numpy()
    
    def _compute_state_values(self, 
                             observations: torch.Tensor,
                             q_function: Any) -> torch.Tensor:
        """
        Compute state values V(s) by sampling actions.
        
        This approximates V(s) = E_a~π_BC[Q(s,a)] by sampling actions
        from the behavior policy (approximated as uniform random).
        
        Args:
            observations: Batch of observations
            q_function: Q-function
        
        Returns:
            State values V(s)
        """
        batch_size = observations.shape[0]
        action_dim = self._infer_action_dim(q_function, observations)
        
        # Sample actions and compute Q-values
        sampled_q_values = []
        
        for _ in range(self.num_action_samples):
            # Sample actions from behavior policy approximation
            # For simplicity, we use uniform random actions in [-1, 1]
            # In practice, this could be improved by using a learned behavior policy
            sampled_actions = self._sample_behavior_actions(batch_size, action_dim)
            
            # Compute Q-values for sampled actions
            q_vals = q_function(observations, sampled_actions)
            if len(q_vals.shape) > 1:
                q_vals = q_vals.squeeze(-1)
            
            sampled_q_values.append(q_vals)
        
        # Average over sampled actions to get V(s)
        sampled_q_values = torch.stack(sampled_q_values, dim=0)
        state_values = torch.mean(sampled_q_values, dim=0)
        
        return state_values
    
    def _sample_behavior_actions(self, batch_size: int, action_dim: int) -> torch.Tensor:
        """
        Sample actions from an approximation of the behavior policy.
        
        Args:
            batch_size: Number of actions to sample
            action_dim: Action dimension
        
        Returns:
            Sampled actions
        """
        # Simple approximation: uniform random actions in [-1, 1]
        # This could be improved by learning a behavior cloning policy
        actions = torch.rand(batch_size, action_dim).to(self.device) * 2 - 1
        return actions
    
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
    
    def score_with_behavior_policy(self,
                                  observations: np.ndarray,
                                  actions: np.ndarray,
                                  q_function: Any,
                                  behavior_policy: Any) -> np.ndarray:
        """
        Score transitions using a learned behavior policy for V(s) computation.
        
        Args:
            observations: Observations
            actions: Actions
            q_function: Q-function
            behavior_policy: Learned behavior policy
        
        Returns:
            Advantage scores
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            
            # Compute Q(s,a)
            q_values = q_function(obs_tensor, actions_tensor)
            if len(q_values.shape) > 1:
                q_values = q_values.squeeze(-1)
            
            # Compute V(s) using behavior policy
            state_values = self._compute_state_values_with_policy(
                obs_tensor, q_function, behavior_policy
            )
            
            # Compute advantage
            advantages = q_values - state_values
            
            return advantages.cpu().numpy()
    
    def _compute_state_values_with_policy(self,
                                         observations: torch.Tensor,
                                         q_function: Any,
                                         behavior_policy: Any) -> torch.Tensor:
        """
        Compute state values using a learned behavior policy.
        
        Args:
            observations: Observations
            q_function: Q-function
            behavior_policy: Behavior policy
        
        Returns:
            State values
        """
        sampled_q_values = []
        
        for _ in range(self.num_action_samples):
            # Sample actions from behavior policy
            sampled_actions = behavior_policy.predict(observations.cpu().numpy())
            sampled_actions = torch.FloatTensor(sampled_actions).to(self.device)
            
            # Add noise for exploration
            noise = torch.randn_like(sampled_actions) * 0.1
            sampled_actions = torch.clamp(sampled_actions + noise, -1, 1)
            
            # Compute Q-values
            q_vals = q_function(observations, sampled_actions)
            if len(q_vals.shape) > 1:
                q_vals = q_vals.squeeze(-1)
            
            sampled_q_values.append(q_vals)
        
        # Average over samples
        sampled_q_values = torch.stack(sampled_q_values, dim=0)
        state_values = torch.mean(sampled_q_values, dim=0)
        
        return state_values
    
    @property
    def name(self) -> str:
        """Return scorer name."""
        return "Advantage"
    
    def __str__(self) -> str:
        return f"AdvantageScorer(num_action_samples={self.num_action_samples})"

