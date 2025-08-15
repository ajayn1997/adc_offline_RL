"""
Comprehensive ADC wrapper that integrates all components.

This module provides a high-level interface for using ADC with any offline RL algorithm.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from .static_adc import StaticADC
from .dynamic_adc import DynamicADC
from .base import ADCConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scoring import create_scorer, BaseScorer
from baselines import create_simple_algorithm
from utils.env_utils import create_synthetic_env, evaluate_policy
from utils.metrics import compute_d4rl_score


@dataclass
class ADCExperimentConfig:
    """Configuration for ADC experiments."""
    # ADC parameters
    curriculum_type: str = "static"  # "static" or "dynamic"
    scoring_method: str = "advantage"  # "advantage" or "td_error"
    num_stages: int = 5
    total_steps: int = 100000  # Reduced for demo
    pretrain_steps: int = 5000
    
    # Algorithm parameters
    algorithm: str = "cql"  # "cql", "iql", "bc"
    
    # Environment parameters
    env_name: str = "hopper-medium-replay-v2"
    
    # Evaluation parameters
    eval_episodes: int = 10
    
    # System parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class ADCWrapper:
    """
    High-level wrapper for Active Data Curation.
    
    This class provides a simple interface for running ADC experiments
    with different configurations and algorithms.
    """
    
    def __init__(self, config: ADCExperimentConfig):
        """
        Initialize ADC wrapper.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.dataset = None
        self.algorithm = None
        self.adc_trainer = None
        self.scorer = None
        self.env = None
        
        # Set random seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    def setup(self, dataset: Dict[str, np.ndarray]) -> None:
        """
        Setup the experiment with dataset and components.
        
        Args:
            dataset: Offline dataset
        """
        self.dataset = dataset
        
        # Infer dimensions from dataset
        obs_dim = dataset['observations'].shape[1]
        action_dim = dataset['actions'].shape[1]
        
        print(f"Setting up ADC experiment:")
        print(f"  - Environment: {self.config.env_name}")
        print(f"  - Algorithm: {self.config.algorithm}")
        print(f"  - Curriculum: {self.config.curriculum_type}")
        print(f"  - Scoring: {self.config.scoring_method}")
        print(f"  - Dataset size: {len(dataset['observations'])} transitions")
        print(f"  - Obs dim: {obs_dim}, Action dim: {action_dim}")
        
        # Create base algorithm
        self.algorithm = create_simple_algorithm(
            self.config.algorithm,
            obs_dim=obs_dim,
            action_dim=action_dim
        )
        
        # Create scorer
        self.scorer = create_scorer(
            self.config.scoring_method,
            device=self.config.device
        )
        
        # Create ADC configuration
        adc_config = ADCConfig(
            num_stages=self.config.num_stages,
            total_steps=self.config.total_steps,
            scoring_method=self.config.scoring_method,
            pretrain_steps=self.config.pretrain_steps,
            device=self.config.device,
            seed=self.config.seed
        )
        
        # Create ADC trainer
        if self.config.curriculum_type.lower() == "static":
            self.adc_trainer = StaticADC(self.algorithm, adc_config, self.scorer)
        elif self.config.curriculum_type.lower() == "dynamic":
            self.adc_trainer = DynamicADC(self.algorithm, adc_config, self.scorer)
        elif self.config.curriculum_type.lower() == "none":
            # Baseline: train on full dataset without curriculum
            self.adc_trainer = None  # Will handle specially
        else:
            raise ValueError(f"Unknown curriculum type: {self.config.curriculum_type}")
        
        # Set dataset
        if self.adc_trainer is not None:
            self.adc_trainer.set_dataset(dataset)
        else:
            # For baseline, set dataset directly on algorithm
            self.algorithm.set_dataset(dataset)
        
        # Create environment for evaluation
        self.env = create_synthetic_env(self.config.env_name)
        
        print("Setup completed successfully!")
    
    def train(self) -> Dict[str, Any]:
        """
        Train using ADC curriculum or baseline.
        
        Returns:
            Training results
        """
        if self.adc_trainer is not None:
            # Use ADC curriculum
            print(f"\nStarting {self.config.curriculum_type} ADC training...")
            results = self.adc_trainer.train()
            print("Training completed!")
            return results
        else:
            # Baseline training (no curriculum)
            print(f"\nStarting baseline training (no curriculum)...")
            results = self.algorithm.train(steps=self.config.total_steps)
            print("Baseline training completed!")
            return {
                'method': 'Baseline',
                'config': self.config,
                'training_results': results,
                'total_steps': self.config.total_steps
            }
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained policy.
        
        Returns:
            Evaluation results
        """
        if self.algorithm is None or self.env is None:
            raise ValueError("Must train before evaluation")
        
        print(f"\nEvaluating policy for {self.config.eval_episodes} episodes...")
        eval_results = evaluate_policy(
            self.algorithm, 
            self.env, 
            num_episodes=self.config.eval_episodes
        )
        
        # Compute normalized score
        raw_score = eval_results['mean_return']
        normalized_score = compute_d4rl_score(self.config.env_name, raw_score)
        
        eval_results['normalized_score'] = normalized_score
        eval_results['raw_score'] = raw_score
        
        print(f"Evaluation results:")
        print(f"  - Raw score: {raw_score:.2f}")
        print(f"  - Normalized score: {normalized_score:.2f}")
        print(f"  - Episode length: {eval_results['mean_length']:.1f}")
        
        return eval_results
    
    def run_experiment(self, dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run complete ADC experiment.
        
        Args:
            dataset: Offline dataset
        
        Returns:
            Complete experiment results
        """
        # Setup
        self.setup(dataset)
        
        # Train
        training_results = self.train()
        
        # Evaluate
        eval_results = self.evaluate()
        
        # Combine results
        experiment_results = {
            'config': self.config,
            'training_results': training_results,
            'evaluation_results': eval_results,
            'final_score': eval_results['normalized_score']
        }
        
        return experiment_results
    
    def get_curriculum_analysis(self) -> Dict[str, Any]:
        """
        Get analysis of the curriculum structure.
        
        Returns:
            Curriculum analysis
        """
        if self.adc_trainer is None:
            return {"error": "No training performed yet"}
        
        if hasattr(self.adc_trainer, 'get_curriculum_info'):
            return self.adc_trainer.get_curriculum_info()
        elif hasattr(self.adc_trainer, 'get_curriculum_evolution'):
            return self.adc_trainer.get_curriculum_evolution()
        else:
            return {"error": "Curriculum analysis not available"}
    
    def compare_with_baseline(self, dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare ADC performance with baseline (no curriculum).
        
        Args:
            dataset: Offline dataset
        
        Returns:
            Comparison results
        """
        print("\n" + "="*50)
        print("RUNNING BASELINE COMPARISON")
        print("="*50)
        
        # Run ADC experiment
        adc_results = self.run_experiment(dataset)
        adc_score = adc_results['final_score']
        
        print("\n" + "-"*30)
        print("Running baseline (no curriculum)...")
        print("-"*30)
        
        # Run baseline experiment (train on full dataset)
        obs_dim = dataset['observations'].shape[1]
        action_dim = dataset['actions'].shape[1]
        
        baseline_algorithm = create_simple_algorithm(
            self.config.algorithm,
            obs_dim=obs_dim,
            action_dim=action_dim
        )
        
        baseline_algorithm.set_dataset(dataset)
        baseline_algorithm.train(steps=self.config.total_steps)
        
        # Evaluate baseline
        baseline_eval = evaluate_policy(
            baseline_algorithm,
            self.env,
            num_episodes=self.config.eval_episodes
        )
        
        baseline_score = compute_d4rl_score(
            self.config.env_name, 
            baseline_eval['mean_return']
        )
        
        print(f"Baseline results:")
        print(f"  - Raw score: {baseline_eval['mean_return']:.2f}")
        print(f"  - Normalized score: {baseline_score:.2f}")
        
        # Compute improvement
        improvement = adc_score - baseline_score
        improvement_pct = (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0
        
        comparison_results = {
            'adc_results': adc_results,
            'baseline_score': baseline_score,
            'adc_score': adc_score,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'config': self.config
        }
        
        print(f"\n" + "="*50)
        print("COMPARISON RESULTS")
        print("="*50)
        print(f"ADC Score: {adc_score:.2f}")
        print(f"Baseline Score: {baseline_score:.2f}")
        print(f"Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
        print("="*50)
        
        return comparison_results


def run_adc_experiment(
    dataset: Dict[str, np.ndarray],
    env_name: str,
    algorithm: str = "cql",
    curriculum_type: str = "static",
    scoring_method: str = "advantage",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run an ADC experiment.
    
    Args:
        dataset: Offline dataset
        env_name: Environment name
        algorithm: Algorithm name
        curriculum_type: Curriculum type
        scoring_method: Scoring method
        **kwargs: Additional configuration parameters
    
    Returns:
        Experiment results
    """
    config = ADCExperimentConfig(
        env_name=env_name,
        algorithm=algorithm,
        curriculum_type=curriculum_type,
        scoring_method=scoring_method,
        **kwargs
    )
    
    wrapper = ADCWrapper(config)
    return wrapper.run_experiment(dataset)

