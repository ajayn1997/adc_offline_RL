"""
Static Active Data Curation (Static ADC) implementation.

This module implements Algorithm 1 from the paper, where data is scored once
at the beginning and used to create a fixed curriculum.
"""

import numpy as np
import torch
from typing import Dict, Any, List
from tqdm import tqdm

from .base import ADCBase, ADCConfig


class StaticADC(ADCBase):
    """
    Static Active Data Curation implementation.
    
    In Static ADC, the dataset is scored and sorted only once at the beginning
    of training using an initial Q-function. This fixed ranking is then used
    to create a sequence of progressively larger data buffers.
    """
    
    def __init__(self, base_algorithm: Any, config: ADCConfig, scorer: Any = None):
        """
        Initialize Static ADC.
        
        Args:
            base_algorithm: Base offline RL algorithm (CQL, IQL, BC)
            config: ADC configuration
            scorer: Data scoring function (optional)
        """
        super().__init__(base_algorithm, config, scorer)
        self.initial_scores = None
        self.training_history = []
    
    def train(self) -> Dict[str, Any]:
        """
        Train using Static ADC curriculum (Algorithm 1).
        
        Returns:
            Dictionary containing training results and metrics
        """
        if self.dataset is None:
            raise ValueError("Dataset must be set before training")
        
        print("Starting Static ADC training...")
        print(f"Configuration: {self.config.num_stages} stages, "
              f"{self.config.total_steps} total steps, "
              f"scoring method: {self.config.scoring_method}")
        
        # Step 1: Pre-train Q-network on all data for initial value estimate
        print(f"Pre-training for {self.config.pretrain_steps} steps...")
        self.base_algorithm.set_dataset(self.dataset)
        self.base_algorithm.pretrain(self.config.pretrain_steps)
        
        # Step 2: Score all transitions using initial Q-function
        print("Scoring all transitions...")
        self.initial_scores = self.score_transitions()
        print(f"Score statistics - Mean: {np.mean(self.initial_scores):.4f}, "
              f"Std: {np.std(self.initial_scores):.4f}, "
              f"Min: {np.min(self.initial_scores):.4f}, "
              f"Max: {np.max(self.initial_scores):.4f}")
        
        # Step 3: Sort dataset in descending order based on scores
        self.sorted_indices = np.argsort(self.initial_scores)[::-1]
        
        # Step 4: Create stage datasets
        stage_datasets = self.create_stage_datasets(self.initial_scores)
        
        # Step 5: Train through curriculum stages
        steps_per_stage = self.config.total_steps // self.config.num_stages
        
        for stage in range(1, self.config.num_stages + 1):
            print(f"\n--- Stage {stage}/{self.config.num_stages} ---")
            
            # Get active dataset for this stage
            stage_indices = stage_datasets[stage]
            stage_dataset = self.get_stage_dataset(stage, stage_indices)
            
            data_percentage = (stage / self.config.num_stages) * 100
            print(f"Training on top {data_percentage:.1f}% of data "
                  f"({len(stage_indices)} transitions)")
            
            # Train for this stage
            stage_results = self.base_algorithm.train(
                dataset=stage_dataset,
                steps=steps_per_stage
            )
            
            # Record training history
            stage_info = {
                'stage': stage,
                'data_percentage': data_percentage,
                'num_transitions': len(stage_indices),
                'steps': steps_per_stage,
                'results': stage_results
            }
            self.training_history.append(stage_info)
            
            print(f"Stage {stage} completed")
        
        print("\nStatic ADC training completed!")
        
        return {
            'method': 'Static ADC',
            'config': self.config,
            'initial_scores': self.initial_scores,
            'sorted_indices': self.sorted_indices,
            'training_history': self.training_history,
            'total_steps': self.config.total_steps
        }
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """
        Get information about the curriculum structure.
        
        Returns:
            Dictionary with curriculum information
        """
        if self.initial_scores is None:
            return {"error": "Training not started yet"}
        
        stage_info = []
        for stage in range(1, self.config.num_stages + 1):
            num_transitions = int((stage / self.config.num_stages) * self.dataset_size)
            data_percentage = (stage / self.config.num_stages) * 100
            
            # Get score statistics for this stage
            stage_indices = self.sorted_indices[:num_transitions]
            stage_scores = self.initial_scores[stage_indices]
            
            stage_info.append({
                'stage': stage,
                'num_transitions': num_transitions,
                'data_percentage': data_percentage,
                'score_mean': np.mean(stage_scores),
                'score_std': np.std(stage_scores),
                'score_min': np.min(stage_scores),
                'score_max': np.max(stage_scores)
            })
        
        return {
            'method': 'Static ADC',
            'total_transitions': self.dataset_size,
            'num_stages': self.config.num_stages,
            'scoring_method': self.config.scoring_method,
            'stage_info': stage_info
        }
    
    def analyze_data_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of scores and data selection.
        
        Returns:
            Dictionary with analysis results
        """
        if self.initial_scores is None:
            return {"error": "Training not started yet"}
        
        # Score distribution analysis
        score_percentiles = np.percentile(self.initial_scores, [10, 25, 50, 75, 90])
        
        # Analyze what percentage of data is used at each stage
        stage_analysis = []
        for stage in range(1, self.config.num_stages + 1):
            num_transitions = int((stage / self.config.num_stages) * self.dataset_size)
            threshold_score = self.initial_scores[self.sorted_indices[num_transitions-1]]
            
            stage_analysis.append({
                'stage': stage,
                'data_percentage': (stage / self.config.num_stages) * 100,
                'score_threshold': threshold_score,
                'num_transitions': num_transitions
            })
        
        return {
            'score_statistics': {
                'mean': np.mean(self.initial_scores),
                'std': np.std(self.initial_scores),
                'min': np.min(self.initial_scores),
                'max': np.max(self.initial_scores),
                'percentiles': {
                    '10th': score_percentiles[0],
                    '25th': score_percentiles[1],
                    '50th': score_percentiles[2],
                    '75th': score_percentiles[3],
                    '90th': score_percentiles[4]
                }
            },
            'stage_analysis': stage_analysis
        }

