"""
Environment utilities for evaluation and testing.

This module provides synthetic environments that mimic D4RL environments
for policy evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class SyntheticEnv:
    """
    Synthetic environment that mimics D4RL environments.
    
    This is a simple environment for demonstration purposes that provides
    the same interface as gym environments.
    """
    
    def __init__(self, env_name: str, obs_dim: int = 17, action_dim: int = 6, max_steps: int = 1000):
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        
        self.current_step = 0
        self.current_obs = None
        
        # Environment-specific parameters
        if 'antmaze' in env_name:
            self.is_sparse_reward = True
            self.goal_position = np.array([5.0, 5.0])
            self.goal_threshold = 0.5
        else:
            self.is_sparse_reward = False
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.current_step = 0
        self.current_obs = np.random.randn(self.obs_dim) * 0.5
        return self.current_obs.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment."""
        action = np.clip(action, -1, 1)  # Clip actions to valid range
        
        # Update observation based on action
        action_effect = np.zeros(self.obs_dim)
        action_effect[:min(len(action), self.obs_dim)] = action[:min(len(action), self.obs_dim)]
        next_obs = self.current_obs + 0.1 * action_effect + np.random.randn(self.obs_dim) * 0.02
        
        # Compute reward based on environment type
        reward = self._compute_reward(self.current_obs, action, next_obs)
        
        # Check termination conditions
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Check for early termination (e.g., falling in locomotion tasks)
        if not self.is_sparse_reward and self.current_step > 10:
            # Simple termination condition: if observation goes too far from origin
            if np.linalg.norm(next_obs) > 10:
                done = True
                reward -= 10  # Penalty for early termination
        
        self.current_obs = next_obs
        
        info = {
            'TimeLimit.truncated': self.current_step >= self.max_steps,
            'episode_step': self.current_step
        }
        
        return next_obs.copy(), reward, done, info
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
        """Compute reward based on environment type."""
        if self.is_sparse_reward:
            # Sparse reward for antmaze-like environments
            if self.obs_dim >= 2:
                current_pos = next_obs[:2]
                distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
                if distance_to_goal < self.goal_threshold:
                    return 1.0
            return 0.0
        else:
            # Dense reward for locomotion tasks
            if 'hopper' in self.env_name or 'walker2d' in self.env_name:
                # Reward for forward progress
                forward_progress = next_obs[0] - obs[0] if self.obs_dim > 0 else 0
                control_cost = -0.001 * np.sum(action ** 2)
                healthy_reward = 1.0
                return forward_progress + control_cost + healthy_reward
            
            elif 'halfcheetah' in self.env_name:
                # Reward for speed
                forward_progress = next_obs[0] - obs[0] if self.obs_dim > 0 else 0
                control_cost = -0.1 * np.sum(action ** 2)
                return forward_progress + control_cost
            
            else:
                # Generic reward
                control_cost = -0.01 * np.sum(action ** 2)
                stability_reward = -0.01 * np.sum(next_obs ** 2)
                return control_cost + stability_reward


def create_synthetic_env(env_name: str) -> SyntheticEnv:
    """
    Create a synthetic environment based on environment name.
    
    Args:
        env_name: Name of the environment
    
    Returns:
        SyntheticEnv instance
    """
    # Define environment dimensions based on D4RL environments
    env_configs = {
        'hopper-medium-replay-v2': {'obs_dim': 11, 'action_dim': 3},
        'walker2d-medium-replay-v2': {'obs_dim': 17, 'action_dim': 6},
        'halfcheetah-medium-replay-v2': {'obs_dim': 17, 'action_dim': 6},
        'hopper-medium-expert-v2': {'obs_dim': 11, 'action_dim': 3},
        'walker2d-medium-expert-v2': {'obs_dim': 17, 'action_dim': 6},
        'antmaze-medium-play-v2': {'obs_dim': 29, 'action_dim': 8},
        'antmaze-medium-diverse-v2': {'obs_dim': 29, 'action_dim': 8},
        'hopper-99r-1e': {'obs_dim': 11, 'action_dim': 3},
    }
    
    # Extract base environment name
    base_name = env_name.split('-')[0]
    
    config = env_configs.get(env_name, {'obs_dim': 17, 'action_dim': 6})
    
    return SyntheticEnv(
        env_name=base_name,
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim']
    )


def evaluate_policy(policy, env: SyntheticEnv, num_episodes: int = 10, max_steps: int = 1000) -> Dict[str, float]:
    """
    Evaluate a policy in the environment.
    
    Args:
        policy: Policy to evaluate (should have predict method)
        env: Environment to evaluate in
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_returns = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_return = 0
        episode_length = 0
        
        for step in range(max_steps):
            try:
                action = policy.predict(obs)
            except Exception as e:
                # If policy prediction fails, use random action
                action = np.random.randn(env.action_dim)
            
            obs, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
            
            if done:
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        # Check for success (for sparse reward environments)
        if env.is_sparse_reward and episode_return > 0:
            success_count += 1
    
    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / num_episodes if env.is_sparse_reward else None
    }
    
    return results


def get_reference_scores(env_name: str) -> Dict[str, float]:
    """
    Get reference scores for normalization (mimicking D4RL reference scores).
    
    Args:
        env_name: Environment name
    
    Returns:
        Dictionary with random and expert reference scores
    """
    # These are approximate reference scores based on D4RL paper
    reference_scores = {
        'hopper-medium-replay-v2': {'random': -20.272305, 'expert': 3234.3},
        'walker2d-medium-replay-v2': {'random': 1.629008, 'expert': 4592.3},
        'halfcheetah-medium-replay-v2': {'random': -280.178953, 'expert': 12135.0},
        'hopper-medium-expert-v2': {'random': -20.272305, 'expert': 3234.3},
        'walker2d-medium-expert-v2': {'random': 1.629008, 'expert': 4592.3},
        'antmaze-medium-play-v2': {'random': 0.0, 'expert': 1.0},
        'antmaze-medium-diverse-v2': {'random': 0.0, 'expert': 1.0},
        'hopper-99r-1e': {'random': -20.272305, 'expert': 3234.3},
    }
    
    return reference_scores.get(env_name, {'random': 0.0, 'expert': 100.0})


class MockD4RLEnvironment:
    """Mock D4RL environment for compatibility."""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.env = create_synthetic_env(env_name)
        self.reference_scores = get_reference_scores(env_name)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def get_normalized_score(self, score: float) -> float:
        """Compute normalized score like D4RL."""
        random_score = self.reference_scores['random']
        expert_score = self.reference_scores['expert']
        
        if expert_score == random_score:
            return 0.0
        
        normalized = 100.0 * (score - random_score) / (expert_score - random_score)
        return normalized

