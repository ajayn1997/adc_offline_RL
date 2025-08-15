"""
Wrapper for d3rlpy algorithms to provide consistent interface for ADC framework.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
import d3rlpy
from d3rlpy.algos import CQL, IQL, BC
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import MinMaxActionScaler, StandardObservationScaler


class D3RLPyWrapper:
    """
    Wrapper class for d3rlpy algorithms to provide consistent interface.
    """
    
    def __init__(self, 
                 algorithm_name: str,
                 **kwargs):
        """
        Initialize d3rlpy algorithm wrapper.
        
        Args:
            algorithm_name: Name of algorithm ('cql', 'iql', 'bc')
            **kwargs: Additional arguments for algorithm initialization
        """
        self.algorithm_name = algorithm_name.lower()
        self.algorithm = self._create_algorithm(**kwargs)
        self.dataset = None
        self.is_fitted = False
        
    def _create_algorithm(self, **kwargs):
        """Create the d3rlpy algorithm instance."""
        default_kwargs = {
            'use_gpu': torch.cuda.is_available(),
            'batch_size': 256,
            'learning_rate': 3e-4,
        }
        default_kwargs.update(kwargs)
        
        if self.algorithm_name == 'cql':
            return CQL(
                alpha=1.0,  # Conservative regularization parameter
                **default_kwargs
            )
        elif self.algorithm_name == 'iql':
            return IQL(
                expectile=0.7,  # Expectile parameter for IQL
                **default_kwargs
            )
        elif self.algorithm_name == 'bc':
            return BC(**default_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def set_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        """
        Set dataset for training.
        
        Args:
            dataset: Dictionary with keys 'observations', 'actions', 'rewards', 
                    'next_observations', 'terminals'
        """
        # Convert to d3rlpy MDPDataset format
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards'].reshape(-1, 1)
        terminals = dataset['terminals'].astype(bool)
        
        # Create MDPDataset
        self.dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            action_scaler=MinMaxActionScaler(),
            observation_scaler=StandardObservationScaler()
        )
        
        print(f"Dataset set for {self.algorithm_name} with {len(observations)} transitions")
    
    def pretrain(self, steps: int) -> None:
        """
        Pretrain the algorithm for initial Q-function estimation.
        
        Args:
            steps: Number of training steps
        """
        if self.dataset is None:
            raise ValueError("Dataset must be set before pretraining")
        
        print(f"Pretraining {self.algorithm_name} for {steps} steps...")
        
        # Fit the algorithm for specified steps
        self.algorithm.fit(
            self.dataset,
            n_steps=steps,
            verbose=False,
            show_progress=False
        )
        
        self.is_fitted = True
        print("Pretraining completed")
    
    def train(self, dataset: Optional[Dict[str, np.ndarray]] = None, steps: int = 1000) -> Dict[str, Any]:
        """
        Train the algorithm.
        
        Args:
            dataset: Training dataset (if None, uses self.dataset)
            steps: Number of training steps
            
        Returns:
            Training metrics
        """
        if dataset is not None:
            self.set_dataset(dataset)
        
        if self.dataset is None:
            raise ValueError("No dataset available for training")
        
        # Train the algorithm
        results = self.algorithm.fit(
            self.dataset,
            n_steps=steps,
            verbose=False,
            show_progress=True
        )
        
        self.is_fitted = True
        
        return {
            'training_steps': steps,
            'algorithm': self.algorithm_name
        }
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action for given observation.
        
        Args:
            observation: State observation
            
        Returns:
            Predicted action
        """
        if not self.is_fitted:
            raise ValueError("Algorithm must be trained before prediction")
        
        return self.algorithm.predict([observation])[0]
    
    def predict_value(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Predict Q-value for given state-action pair.
        
        Args:
            observation: State observation
            action: Action
            
        Returns:
            Q-value
        """
        if not self.is_fitted:
            raise ValueError("Algorithm must be trained before prediction")
        
        # For d3rlpy, we need to use the internal Q-function
        if hasattr(self.algorithm, 'predict_value'):
            return self.algorithm.predict_value([observation], [action])[0]
        else:
            # Fallback: use the algorithm's internal Q-function
            obs_tensor = torch.FloatTensor([observation])
            action_tensor = torch.FloatTensor([action])
            
            if hasattr(self.algorithm, '_impl') and self.algorithm._impl is not None:
                q_value = self.algorithm._impl.compute_target(obs_tensor, action_tensor)
                return q_value.item()
            else:
                raise NotImplementedError(f"Q-value prediction not available for {self.algorithm_name}")
    
    @property
    def q_function(self):
        """Get the Q-function for scoring."""
        if not self.is_fitted:
            raise ValueError("Algorithm must be trained before accessing Q-function")
        
        return QFunctionWrapper(self.algorithm)


class QFunctionWrapper:
    """Wrapper to provide consistent Q-function interface."""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __call__(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for given state-action pairs.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions
            
        Returns:
            Q-values
        """
        # Convert to numpy for d3rlpy
        obs_np = observations.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        
        # Use d3rlpy's predict_value if available
        if hasattr(self.algorithm, 'predict_value'):
            q_values = []
            for obs, action in zip(obs_np, actions_np):
                q_val = self.algorithm.predict_value([obs], [action])[0]
                q_values.append(q_val)
            return torch.FloatTensor(q_values).to(self.device)
        
        # Fallback to internal implementation
        if hasattr(self.algorithm, '_impl') and self.algorithm._impl is not None:
            with torch.no_grad():
                q_values = self.algorithm._impl.compute_target(observations, actions)
                return q_values
        
        raise NotImplementedError("Q-function computation not available")
    
    def compute_value(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute state values V(s).
        
        Args:
            observations: Batch of observations
            
        Returns:
            State values
        """
        # For offline RL, we approximate V(s) by sampling actions
        batch_size = observations.shape[0]
        action_dim = self.algorithm.action_size
        
        # Sample multiple actions and average Q-values
        num_samples = 10
        q_values_list = []
        
        for _ in range(num_samples):
            # Sample random actions (approximating behavior policy)
            random_actions = torch.rand(batch_size, action_dim).to(observations.device)
            q_values = self(observations, random_actions)
            q_values_list.append(q_values)
        
        # Average over sampled actions
        avg_q_values = torch.mean(torch.stack(q_values_list), dim=0)
        return avg_q_values

