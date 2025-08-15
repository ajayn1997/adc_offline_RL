"""
Data utilities for creating and handling D4RL-like datasets.

This module provides functions to create synthetic datasets that mimic
the structure and characteristics of D4RL datasets.
"""

import numpy as np
import pickle
import os
from typing import Dict, Tuple, Optional, List
import json


def create_synthetic_dataset(
    env_name: str,
    dataset_type: str,
    num_trajectories: int = 1000,
    max_episode_steps: int = 1000,
    obs_dim: int = 17,
    action_dim: int = 6,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create a synthetic dataset that mimics D4RL structure.
    
    Args:
        env_name: Environment name (e.g., 'hopper', 'walker2d', 'halfcheetah')
        dataset_type: Dataset type ('medium-replay', 'medium-expert', 'random', 'expert')
        num_trajectories: Number of trajectories to generate
        max_episode_steps: Maximum steps per episode
        obs_dim: Observation dimension
        action_dim: Action dimension
        seed: Random seed
    
    Returns:
        Dictionary containing dataset arrays
    """
    np.random.seed(seed)
    
    # Generate trajectories based on dataset type
    if dataset_type == 'expert':
        # High-quality trajectories with high rewards
        reward_scale = 1.0
        noise_scale = 0.1
        success_rate = 0.9
    elif dataset_type == 'medium':
        # Medium-quality trajectories
        reward_scale = 0.6
        noise_scale = 0.3
        success_rate = 0.5
    elif dataset_type == 'random':
        # Low-quality random trajectories
        reward_scale = 0.1
        noise_scale = 0.8
        success_rate = 0.1
    elif dataset_type == 'medium-replay':
        # Mix of medium and random data (noisy)
        reward_scale = 0.4
        noise_scale = 0.6
        success_rate = 0.3
    elif dataset_type == 'medium-expert':
        # Mix of medium and expert data
        reward_scale = 0.8
        noise_scale = 0.2
        success_rate = 0.7
    elif dataset_type in ['medium-play', 'medium-diverse']:
        # AntMaze specific dataset types
        reward_scale = 0.3
        noise_scale = 0.4
        success_rate = 0.2
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    timeouts = []
    
    for traj_idx in range(num_trajectories):
        # Generate trajectory length
        if success_rate > np.random.random():
            # Successful trajectory (longer)
            traj_length = np.random.randint(max_episode_steps // 2, max_episode_steps)
        else:
            # Failed trajectory (shorter)
            traj_length = np.random.randint(10, max_episode_steps // 2)
        
        # Generate trajectory
        traj_obs = []
        traj_actions = []
        traj_rewards = []
        traj_next_obs = []
        traj_terminals = []
        traj_timeouts = []
        
        # Initial observation
        obs = np.random.randn(obs_dim) * 0.5
        
        for step in range(traj_length):
            # Generate action (with noise based on dataset quality)
            if dataset_type == 'expert':
                # Expert actions are more consistent
                action = np.tanh(obs[:action_dim] + np.random.randn(action_dim) * noise_scale)
            else:
                # Other datasets have more random actions
                action = np.random.randn(action_dim) * (1 + noise_scale)
                action = np.clip(action, -1, 1)
            
            # Generate next observation
            action_effect = np.zeros(obs_dim)
            action_effect[:min(action_dim, obs_dim)] = action[:min(action_dim, obs_dim)]
            next_obs = obs + 0.1 * action_effect + np.random.randn(obs_dim) * 0.05
            
            # Generate reward based on "progress" and dataset quality
            if env_name in ['hopper', 'walker2d']:
                # Locomotion tasks: reward for forward progress
                progress = next_obs[0] - obs[0] if obs_dim > 0 else 0
                base_reward = progress * 10
            elif env_name == 'halfcheetah':
                # HalfCheetah: reward for speed
                speed = np.linalg.norm(action)
                base_reward = speed * 5
            elif 'antmaze' in env_name:
                # Sparse reward environment
                goal_distance = np.linalg.norm(next_obs[:2] - np.array([5, 5]))
                base_reward = 1.0 if goal_distance < 0.5 else 0.0
            else:
                # Generic reward
                base_reward = -np.linalg.norm(action) * 0.1
            
            reward = base_reward * reward_scale + np.random.randn() * 0.1
            
            # Terminal condition
            is_terminal = (step == traj_length - 1)
            is_timeout = (step == max_episode_steps - 1)
            
            traj_obs.append(obs.copy())
            traj_actions.append(action.copy())
            traj_rewards.append(reward)
            traj_next_obs.append(next_obs.copy())
            traj_terminals.append(is_terminal and not is_timeout)
            traj_timeouts.append(is_timeout)
            
            obs = next_obs
        
        # Add trajectory to dataset
        observations.extend(traj_obs)
        actions.extend(traj_actions)
        rewards.extend(traj_rewards)
        next_observations.extend(traj_next_obs)
        terminals.extend(traj_terminals)
        timeouts.extend(traj_timeouts)
    
    # Convert to numpy arrays
    dataset = {
        'observations': np.array(observations, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'next_observations': np.array(next_observations, dtype=np.float32),
        'terminals': np.array(terminals, dtype=bool),
        'timeouts': np.array(timeouts, dtype=bool)
    }
    
    print(f"Created synthetic {env_name}-{dataset_type} dataset:")
    print(f"  - {len(observations)} transitions")
    print(f"  - {num_trajectories} trajectories")
    print(f"  - Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"  - Mean reward: {np.mean(rewards):.2f}")
    
    return dataset


def create_mixed_dataset(
    base_dataset: Dict[str, np.ndarray],
    noise_dataset: Dict[str, np.ndarray],
    mix_ratio: float
) -> Dict[str, np.ndarray]:
    """
    Create a mixed dataset by combining two datasets.
    
    Args:
        base_dataset: Base dataset (e.g., expert data)
        noise_dataset: Noise dataset (e.g., random data)
        mix_ratio: Ratio of noise data (0.0 = all base, 1.0 = all noise)
    
    Returns:
        Mixed dataset
    """
    base_size = len(base_dataset['observations'])
    noise_size = len(noise_dataset['observations'])
    
    # Calculate number of samples from each dataset
    total_size = base_size + noise_size
    num_noise = int(total_size * mix_ratio)
    num_base = total_size - num_noise
    
    # Sample indices
    base_indices = np.random.choice(base_size, min(num_base, base_size), replace=False)
    noise_indices = np.random.choice(noise_size, min(num_noise, noise_size), replace=False)
    
    # Combine datasets
    mixed_dataset = {}
    for key in base_dataset.keys():
        base_data = base_dataset[key][base_indices]
        noise_data = noise_dataset[key][noise_indices]
        mixed_dataset[key] = np.concatenate([base_data, noise_data], axis=0)
    
    # Shuffle the combined dataset
    total_samples = len(mixed_dataset['observations'])
    shuffle_indices = np.random.permutation(total_samples)
    
    for key in mixed_dataset.keys():
        mixed_dataset[key] = mixed_dataset[key][shuffle_indices]
    
    print(f"Created mixed dataset with {mix_ratio*100:.1f}% noise data:")
    print(f"  - {total_samples} total transitions")
    print(f"  - {len(base_indices)} base transitions")
    print(f"  - {len(noise_indices)} noise transitions")
    
    return mixed_dataset


def save_dataset(dataset: Dict[str, np.ndarray], filepath: str) -> None:
    """Save dataset to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath: str) -> Dict[str, np.ndarray]:
    """Load dataset from file."""
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset loaded from {filepath}")
    print(f"  - {len(dataset['observations'])} transitions")
    
    return dataset


def normalize_dataset(dataset: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Normalize dataset observations and actions.
    
    Args:
        dataset: Input dataset
    
    Returns:
        Tuple of (normalized_dataset, normalization_stats)
    """
    normalized_dataset = dataset.copy()
    stats = {}
    
    # Normalize observations
    obs_mean = np.mean(dataset['observations'], axis=0)
    obs_std = np.std(dataset['observations'], axis=0) + 1e-8
    normalized_dataset['observations'] = (dataset['observations'] - obs_mean) / obs_std
    normalized_dataset['next_observations'] = (dataset['next_observations'] - obs_mean) / obs_std
    
    stats['observations'] = {'mean': obs_mean, 'std': obs_std}
    
    # Normalize actions (to [-1, 1] range)
    action_min = np.min(dataset['actions'], axis=0)
    action_max = np.max(dataset['actions'], axis=0)
    action_range = action_max - action_min + 1e-8
    normalized_dataset['actions'] = 2 * (dataset['actions'] - action_min) / action_range - 1
    
    stats['actions'] = {'min': action_min, 'max': action_max, 'range': action_range}
    
    print("Dataset normalized:")
    print(f"  - Observations: mean={np.mean(obs_mean):.3f}, std={np.mean(obs_std):.3f}")
    print(f"  - Actions: range=[{np.mean(action_min):.3f}, {np.mean(action_max):.3f}]")
    
    return normalized_dataset, stats


def get_dataset_info(dataset: Dict[str, np.ndarray]) -> Dict[str, any]:
    """Get information about a dataset."""
    info = {
        'num_transitions': len(dataset['observations']),
        'obs_dim': dataset['observations'].shape[1],
        'action_dim': dataset['actions'].shape[1],
        'reward_stats': {
            'mean': float(np.mean(dataset['rewards'])),
            'std': float(np.std(dataset['rewards'])),
            'min': float(np.min(dataset['rewards'])),
            'max': float(np.max(dataset['rewards']))
        },
        'terminal_rate': float(np.mean(dataset['terminals'])),
        'timeout_rate': float(np.mean(dataset['timeouts'])) if 'timeouts' in dataset else 0.0
    }
    
    return info


def create_d4rl_datasets(data_dir: str = 'data') -> Dict[str, str]:
    """
    Create all synthetic D4RL datasets used in the paper.
    
    Args:
        data_dir: Directory to save datasets
    
    Returns:
        Dictionary mapping dataset names to file paths
    """
    os.makedirs(data_dir, exist_ok=True)
    
    datasets_to_create = [
        # MuJoCo locomotion tasks
        ('hopper', 'medium-replay', 17, 3),
        ('walker2d', 'medium-replay', 17, 6),
        ('halfcheetah', 'medium-replay', 17, 6),
        ('hopper', 'medium-expert', 17, 3),
        ('walker2d', 'medium-expert', 17, 6),
        
        # AntMaze tasks (sparse reward)
        ('antmaze', 'medium-play', 29, 8),
        ('antmaze', 'medium-diverse', 29, 8),
    ]
    
    dataset_paths = {}
    
    for env_name, dataset_type, obs_dim, action_dim in datasets_to_create:
        print(f"\nCreating {env_name}-{dataset_type} dataset...")
        
        if dataset_type in ['medium-replay', 'medium-expert']:
            # Create mixed datasets
            if dataset_type == 'medium-replay':
                # Mix medium and random data
                medium_data = create_synthetic_dataset(
                    env_name, 'medium', 500, obs_dim=obs_dim, action_dim=action_dim
                )
                random_data = create_synthetic_dataset(
                    env_name, 'random', 500, obs_dim=obs_dim, action_dim=action_dim
                )
                dataset = create_mixed_dataset(medium_data, random_data, 0.4)
            else:  # medium-expert
                # Mix medium and expert data
                medium_data = create_synthetic_dataset(
                    env_name, 'medium', 500, obs_dim=obs_dim, action_dim=action_dim
                )
                expert_data = create_synthetic_dataset(
                    env_name, 'expert', 500, obs_dim=obs_dim, action_dim=action_dim
                )
                dataset = create_mixed_dataset(medium_data, expert_data, 0.3)
        else:
            # Create single-type dataset
            dataset = create_synthetic_dataset(
                env_name, dataset_type, 1000, obs_dim=obs_dim, action_dim=action_dim
            )
        
        # Save dataset
        dataset_name = f"{env_name}-{dataset_type}-v2"
        filepath = os.path.join(data_dir, f"{dataset_name}.pkl")
        save_dataset(dataset, filepath)
        dataset_paths[dataset_name] = filepath
    
    # Create robustness analysis datasets (99% random, 1% expert)
    print("\nCreating robustness analysis datasets...")
    for env_name, _, obs_dim, action_dim in [('hopper', '', 17, 3)]:
        expert_data = create_synthetic_dataset(
            env_name, 'expert', 50, obs_dim=obs_dim, action_dim=action_dim
        )
        random_data = create_synthetic_dataset(
            env_name, 'random', 4950, obs_dim=obs_dim, action_dim=action_dim
        )
        noisy_dataset = create_mixed_dataset(expert_data, random_data, 0.99)
        
        dataset_name = f"{env_name}-99r-1e"
        filepath = os.path.join(data_dir, f"{dataset_name}.pkl")
        save_dataset(noisy_dataset, filepath)
        dataset_paths[dataset_name] = filepath
    
    # Save dataset registry
    registry_path = os.path.join(data_dir, 'dataset_registry.json')
    with open(registry_path, 'w') as f:
        json.dump(dataset_paths, f, indent=2)
    
    print(f"\nAll datasets created and saved to {data_dir}/")
    print(f"Dataset registry saved to {registry_path}")
    
    return dataset_paths

