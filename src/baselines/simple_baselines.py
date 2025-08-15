"""
Simple baseline implementations for offline RL algorithms.

This module provides basic implementations of CQL, IQL, and BC algorithms
that can be used for demonstration and testing purposes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
from collections import deque
import random


class MLP(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SimpleBC:
    """Simple Behavioral Cloning implementation."""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-3, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Policy network
        self.policy = MLP(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.is_fitted = False
    
    def set_dataset(self, dataset: Dict[str, np.ndarray]):
        """Set training dataset."""
        self.dataset = dataset
        self.dataset_size = len(dataset['observations'])
    
    def pretrain(self, steps: int):
        """Pretrain for initial estimation."""
        self.train_steps(steps)
    
    def train_steps(self, steps: int):
        """Train for specified number of steps."""
        batch_size = 256
        
        for step in range(steps):
            # Sample batch
            indices = np.random.choice(self.dataset_size, batch_size, replace=True)
            
            obs = torch.FloatTensor(self.dataset['observations'][indices]).to(self.device)
            actions = torch.FloatTensor(self.dataset['actions'][indices]).to(self.device)
            
            # Forward pass
            predicted_actions = self.policy(obs)
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.is_fitted = True
    
    def train(self, dataset: Optional[Dict[str, np.ndarray]] = None, steps: int = 1000):
        """Train the model."""
        if dataset is not None:
            self.set_dataset(dataset)
        
        self.train_steps(steps)
        return {'algorithm': 'BC', 'steps': steps}
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict action for observation."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.policy(obs_tensor).cpu().numpy()[0]
        
        return action
    
    @property
    def q_function(self):
        """Return a mock Q-function for compatibility."""
        return MockQFunction(self.policy, self.device)


class SimpleCQL:
    """Simple Conservative Q-Learning implementation."""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, 
                 alpha: float = 1.0, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alpha = alpha  # Conservative regularization parameter
        self.device = device
        
        # Q-networks
        self.q1 = MLP(obs_dim + action_dim, 1).to(device)
        self.q2 = MLP(obs_dim + action_dim, 1).to(device)
        self.target_q1 = MLP(obs_dim + action_dim, 1).to(device)
        self.target_q2 = MLP(obs_dim + action_dim, 1).to(device)
        
        # Policy network
        self.policy = MLP(obs_dim, action_dim).to(device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Copy to target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        self.is_fitted = False
        self.gamma = 0.99
        self.tau = 0.005  # Target network update rate
    
    def set_dataset(self, dataset: Dict[str, np.ndarray]):
        """Set training dataset."""
        self.dataset = dataset
        self.dataset_size = len(dataset['observations'])
    
    def pretrain(self, steps: int):
        """Pretrain for initial Q-function estimation."""
        self.train_steps(steps)
    
    def train_steps(self, steps: int):
        """Train for specified number of steps."""
        batch_size = 256
        
        for step in range(steps):
            # Sample batch
            indices = np.random.choice(self.dataset_size, batch_size, replace=True)
            
            obs = torch.FloatTensor(self.dataset['observations'][indices]).to(self.device)
            actions = torch.FloatTensor(self.dataset['actions'][indices]).to(self.device)
            rewards = torch.FloatTensor(self.dataset['rewards'][indices]).to(self.device)
            next_obs = torch.FloatTensor(self.dataset['next_observations'][indices]).to(self.device)
            terminals = torch.FloatTensor(self.dataset['terminals'][indices]).to(self.device)
            
            # Train Q-functions
            self._train_q_functions(obs, actions, rewards, next_obs, terminals)
            
            # Train policy (less frequently)
            if step % 2 == 0:
                self._train_policy(obs)
            
            # Update target networks
            if step % 2 == 0:
                self._update_target_networks()
        
        self.is_fitted = True
    
    def _train_q_functions(self, obs, actions, rewards, next_obs, terminals):
        """Train Q-functions with CQL regularization."""
        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.policy(next_obs)
            target_q1 = self.target_q1(torch.cat([next_obs, next_actions], dim=1))
            target_q2 = self.target_q2(torch.cat([next_obs, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - terminals.unsqueeze(1)) * target_q
        
        # Current Q-values
        current_q1 = self.q1(torch.cat([obs, actions], dim=1))
        current_q2 = self.q2(torch.cat([obs, actions], dim=1))
        
        # Q-function loss
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        # CQL regularization
        random_actions = torch.rand_like(actions) * 2 - 1  # Random actions in [-1, 1]
        cql_q1 = self.q1(torch.cat([obs, random_actions], dim=1))
        cql_q2 = self.q2(torch.cat([obs, random_actions], dim=1))
        
        cql_loss = self.alpha * (torch.mean(cql_q1) + torch.mean(cql_q2) - 
                                torch.mean(current_q1) - torch.mean(current_q2))
        
        total_loss = q1_loss + q2_loss + cql_loss
        
        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()
    
    def _train_policy(self, obs):
        """Train policy to maximize Q-values."""
        actions = self.policy(obs)
        q1_values = self.q1(torch.cat([obs, actions], dim=1))
        q2_values = self.q2(torch.cat([obs, actions], dim=1))
        q_values = torch.min(q1_values, q2_values)
        
        policy_loss = -torch.mean(q_values)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def _update_target_networks(self):
        """Update target networks with soft updates."""
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, dataset: Optional[Dict[str, np.ndarray]] = None, steps: int = 1000):
        """Train the model."""
        if dataset is not None:
            self.set_dataset(dataset)
        
        self.train_steps(steps)
        return {'algorithm': 'CQL', 'steps': steps}
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict action for observation."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.policy(obs_tensor).cpu().numpy()[0]
        
        return action
    
    @property
    def q_function(self):
        """Return Q-function for scoring."""
        return CQLQFunction(self.q1, self.q2, self.device)


class SimpleIQL:
    """Simple Implicit Q-Learning implementation."""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, 
                 expectile: float = 0.7, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.expectile = expectile
        self.device = device
        
        # Networks
        self.q1 = MLP(obs_dim + action_dim, 1).to(device)
        self.q2 = MLP(obs_dim + action_dim, 1).to(device)
        self.value = MLP(obs_dim, 1).to(device)
        self.policy = MLP(obs_dim, action_dim).to(device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.is_fitted = False
        self.gamma = 0.99
    
    def set_dataset(self, dataset: Dict[str, np.ndarray]):
        """Set training dataset."""
        self.dataset = dataset
        self.dataset_size = len(dataset['observations'])
    
    def pretrain(self, steps: int):
        """Pretrain for initial estimation."""
        self.train_steps(steps)
    
    def expectile_loss(self, diff, expectile):
        """Compute expectile loss."""
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)
    
    def train_steps(self, steps: int):
        """Train for specified number of steps."""
        batch_size = 256
        
        for step in range(steps):
            # Sample batch
            indices = np.random.choice(self.dataset_size, batch_size, replace=True)
            
            obs = torch.FloatTensor(self.dataset['observations'][indices]).to(self.device)
            actions = torch.FloatTensor(self.dataset['actions'][indices]).to(self.device)
            rewards = torch.FloatTensor(self.dataset['rewards'][indices]).to(self.device)
            next_obs = torch.FloatTensor(self.dataset['next_observations'][indices]).to(self.device)
            terminals = torch.FloatTensor(self.dataset['terminals'][indices]).to(self.device)
            
            # Train Q-functions
            self._train_q_functions(obs, actions, rewards, next_obs, terminals)
            
            # Train value function
            self._train_value_function(obs, actions)
            
            # Train policy
            if step % 2 == 0:
                self._train_policy(obs, actions)
        
        self.is_fitted = True
    
    def _train_q_functions(self, obs, actions, rewards, next_obs, terminals):
        """Train Q-functions."""
        with torch.no_grad():
            next_values = self.value(next_obs)
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - terminals.unsqueeze(1)) * next_values
        
        current_q1 = self.q1(torch.cat([obs, actions], dim=1))
        current_q2 = self.q2(torch.cat([obs, actions], dim=1))
        
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        total_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()
    
    def _train_value_function(self, obs, actions):
        """Train value function using expectile regression."""
        with torch.no_grad():
            q1_values = self.q1(torch.cat([obs, actions], dim=1))
            q2_values = self.q2(torch.cat([obs, actions], dim=1))
            q_values = torch.min(q1_values, q2_values)
        
        values = self.value(obs)
        diff = q_values - values
        value_loss = torch.mean(self.expectile_loss(diff, self.expectile))
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def _train_policy(self, obs, actions):
        """Train policy using advantage weighting."""
        with torch.no_grad():
            q1_values = self.q1(torch.cat([obs, actions], dim=1))
            q2_values = self.q2(torch.cat([obs, actions], dim=1))
            q_values = torch.min(q1_values, q2_values)
            values = self.value(obs)
            advantages = q_values - values
            weights = torch.exp(advantages / 0.1)  # Temperature = 0.1
            weights = torch.clamp(weights, max=100.0)
        
        predicted_actions = self.policy(obs)
        policy_loss = torch.mean(weights * nn.MSELoss(reduction='none')(predicted_actions, actions))
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def train(self, dataset: Optional[Dict[str, np.ndarray]] = None, steps: int = 1000):
        """Train the model."""
        if dataset is not None:
            self.set_dataset(dataset)
        
        self.train_steps(steps)
        return {'algorithm': 'IQL', 'steps': steps}
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict action for observation."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.policy(obs_tensor).cpu().numpy()[0]
        
        return action
    
    @property
    def q_function(self):
        """Return Q-function for scoring."""
        return IQLQFunction(self.q1, self.q2, self.value, self.device)


class MockQFunction:
    """Mock Q-function for BC algorithm."""
    
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
    
    def __call__(self, observations, actions):
        # For BC, we don't have a real Q-function, so return random values
        batch_size = observations.shape[0]
        return torch.randn(batch_size, 1).to(self.device)
    
    def compute_value(self, observations):
        batch_size = observations.shape[0]
        return torch.randn(batch_size, 1).to(self.device)


class CQLQFunction:
    """Q-function wrapper for CQL."""
    
    def __init__(self, q1, q2, device):
        self.q1 = q1
        self.q2 = q2
        self.device = device
    
    def __call__(self, observations, actions):
        obs_actions = torch.cat([observations, actions], dim=1)
        q1_values = self.q1(obs_actions)
        q2_values = self.q2(obs_actions)
        return torch.min(q1_values, q2_values)
    
    def compute_value(self, observations):
        # Approximate V(s) by sampling actions
        batch_size = observations.shape[0]
        action_dim = self._infer_action_dim(observations)
        
        num_samples = 10
        q_values_list = []
        
        for _ in range(num_samples):
            random_actions = torch.rand(batch_size, action_dim).to(observations.device) * 2 - 1
            q_values = self(observations, random_actions)
            q_values_list.append(q_values)
        
        return torch.mean(torch.stack(q_values_list), dim=0)
    
    def _infer_action_dim(self, observations):
        """Infer action dimension from the Q-network architecture."""
        # The Q-network expects obs_dim + action_dim inputs
        obs_dim = observations.shape[1]
        
        # Try to infer from the first layer of q1
        first_layer = self.q1.network[0]
        total_input_dim = first_layer.in_features
        action_dim = total_input_dim - obs_dim
        
        return max(1, action_dim)  # Ensure at least 1


class IQLQFunction:
    """Q-function wrapper for IQL."""
    
    def __init__(self, q1, q2, value_func, device):
        self.q1 = q1
        self.q2 = q2
        self.value_func = value_func
        self.device = device
    
    def __call__(self, observations, actions):
        obs_actions = torch.cat([observations, actions], dim=1)
        q1_values = self.q1(obs_actions)
        q2_values = self.q2(obs_actions)
        return torch.min(q1_values, q2_values)
    
    def compute_value(self, observations):
        return self.value_func(observations)


def create_simple_algorithm(algorithm_name: str, obs_dim: int, action_dim: int, **kwargs):
    """
    Create a simple baseline algorithm.
    
    Args:
        algorithm_name: Name of algorithm ('bc', 'cql', 'iql')
        obs_dim: Observation dimension
        action_dim: Action dimension
        **kwargs: Additional arguments
    
    Returns:
        Algorithm instance
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if algorithm_name.lower() == 'bc':
        return SimpleBC(obs_dim, action_dim, device=device, **kwargs)
    elif algorithm_name.lower() == 'cql':
        return SimpleCQL(obs_dim, action_dim, device=device, **kwargs)
    elif algorithm_name.lower() == 'iql':
        return SimpleIQL(obs_dim, action_dim, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

