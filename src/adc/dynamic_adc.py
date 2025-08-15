"""
Dynamic Active Data Curation (Dynamic ADC) implementation.

This module implements Algorithm 2 from the paper, where data is adaptively
re-scored at each stage to align with the agent's learning progress.
"""

import numpy as np
import torch
from typing import Dict, Any, List
from tqdm import tqdm

from .base import ADCBase, ADCConfig


class DynamicADC(ADCBase):
    """
    Dynamic Active Data Curation implementation.
    
    In Dynamic ADC, the entire dataset is adaptively re-scored at each stage
    of training, allowing the curriculum to evolve alongside the agent's
    own capabilities.
    """
    
    def __init__(self, base_algorithm: Any, config: ADCConfig, scorer: Any = None):
        """
        Initialize Dynamic ADC.
        
        Args:
            base_algorithm: Base offline RL algorithm (CQL, IQL, BC)
            config: ADC configuration
            scorer: Data scoring function (optional)
        """
        super().__init__(base_algorithm, config, scorer)
        self.stage_scores = {}  # Store scores for each stage
        self.stage_indices = {}  # Store sorted indices for each stage
        self.training_history = []
    
    def train(self) -> Dict[str, Any]:
        """
        Train using Dynamic ADC curriculum (Algorithm 2).
        
        Returns:
            Dictionary containing training results and metrics
        """
        if self.dataset is None:
            raise ValueError("Dataset must be set before training")
        
        print("Starting Dynamic ADC training...")
        print(f"Configuration: {self.config.num_stages} stages, "
              f"{self.config.total_steps} total steps, "
              f"scoring method: {self.config.scoring_method}")
        
        # Step 1: Pre-train Q-network on all data for a few epochs
        print(f"Initial pre-training for {self.config.pretrain_steps} steps...")
        self.base_algorithm.set_dataset(self.dataset)
        self.base_algorithm.pretrain(self.config.pretrain_steps)
        
        # Step 2: Train through curriculum stages with adaptive re-scoring
        steps_per_stage = self.config.total_steps // self.config.num_stages
        
        for stage in range(1, self.config.num_stages + 1):
            print(f"\n--- Stage {stage}/{self.config.num_stages} ---")
            
            # Re-score and re-sort data at the start of each stage
            print("Re-scoring all transitions with current Q-function...")
            current_scores = self.score_transitions()
            self.stage_scores[stage] = current_scores
            
            print(f"Stage {stage} score statistics - "
                  f"Mean: {np.mean(current_scores):.4f}, "
                  f"Std: {np.std(current_scores):.4f}, "
                  f"Min: {np.min(current_scores):.4f}, "
                  f"Max: {np.max(current_scores):.4f}")
            
            # Sort dataset based on current scores
            sorted_indices = np.argsort(current_scores)[::-1]
            self.stage_indices[stage] = sorted_indices
            
            # Define active dataset as top (stage/num_stages) * 100% of data
            num_transitions = int((stage / self.config.num_stages) * self.dataset_size)
            stage_indices = sorted_indices[:num_transitions]
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
                'score_statistics': {
                    'mean': np.mean(current_scores),
                    'std': np.std(current_scores),
                    'min': np.min(current_scores),
                    'max': np.max(current_scores)
                },
                'results': stage_results
            }
            self.training_history.append(stage_info)
            
            print(f"Stage {stage} completed")
        
        print("\nDynamic ADC training completed!")
        
        return {
            'method': 'Dynamic ADC',
            'config': self.config,
            'stage_scores': self.stage_scores,
            'stage_indices': self.stage_indices,
            'training_history': self.training_history,
            'total_steps': self.config.total_steps
        }
    
    def get_curriculum_evolution(self) -> Dict[str, Any]:
        """
        Analyze how the curriculum evolved across stages.
        
        Returns:
            Dictionary with curriculum evolution analysis
        """
        if not self.stage_scores:
            return {"error": "Training not started yet"}
        
        evolution_analysis = []
        
        for stage in range(1, self.config.num_stages + 1):
            if stage not in self.stage_scores:
                continue
                
            scores = self.stage_scores[stage]
            indices = self.stage_indices[stage]
            
            num_transitions = int((stage / self.config.num_stages) * self.dataset_size)
            stage_indices = indices[:num_transitions]
            stage_scores = scores[stage_indices]
            
            evolution_analysis.append({
                'stage': stage,
                'data_percentage': (stage / self.config.num_stages) * 100,
                'num_transitions': num_transitions,
                'score_statistics': {
                    'mean': np.mean(stage_scores),
                    'std': np.std(stage_scores),
                    'min': np.min(stage_scores),
                    'max': np.max(stage_scores)
                },
                'global_score_statistics': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            })
        
        return {
            'method': 'Dynamic ADC',
            'total_transitions': self.dataset_size,
            'num_stages': self.config.num_stages,
            'scoring_method': self.config.scoring_method,
            'evolution_analysis': evolution_analysis
        }
    
    def compare_stage_rankings(self) -> Dict[str, Any]:
        """
        Compare how data rankings changed between stages.
        
        Returns:
            Dictionary with ranking comparison analysis
        """
        if len(self.stage_indices) < 2:
            return {"error": "Need at least 2 stages for comparison"}
        
        ranking_changes = []
        
        for stage in range(2, self.config.num_stages + 1):
            prev_indices = self.stage_indices[stage - 1]
            curr_indices = self.stage_indices[stage]
            
            # Calculate ranking correlation
            # Create ranking arrays
            prev_ranking = np.zeros(self.dataset_size)
            curr_ranking = np.zeros(self.dataset_size)
            
            for rank, idx in enumerate(prev_indices):
                prev_ranking[idx] = rank
            for rank, idx in enumerate(curr_indices):
                curr_ranking[idx] = rank
            
            # Compute Spearman correlation
            correlation = np.corrcoef(prev_ranking, curr_ranking)[0, 1]
            
            # Analyze top-k overlap
            top_k_overlaps = {}
            for k in [100, 500, 1000]:
                if k <= self.dataset_size:
                    prev_top_k = set(prev_indices[:k])
                    curr_top_k = set(curr_indices[:k])
                    overlap = len(prev_top_k.intersection(curr_top_k))
                    top_k_overlaps[f'top_{k}'] = overlap / k
            
            ranking_changes.append({
                'stage_transition': f'{stage-1} -> {stage}',
                'ranking_correlation': correlation,
                'top_k_overlaps': top_k_overlaps
            })
        
        return {
            'method': 'Dynamic ADC',
            'ranking_changes': ranking_changes
        }
    
    def analyze_score_evolution(self) -> Dict[str, Any]:
        """
        Analyze how individual transition scores evolved across stages.
        
        Returns:
            Dictionary with score evolution analysis
        """
        if len(self.stage_scores) < 2:
            return {"error": "Need at least 2 stages for analysis"}
        
        # Track score changes for each transition
        score_evolution = np.zeros((self.dataset_size, len(self.stage_scores)))
        
        for stage_idx, (stage, scores) in enumerate(self.stage_scores.items()):
            score_evolution[:, stage_idx] = scores
        
        # Analyze score stability
        score_std = np.std(score_evolution, axis=1)
        score_mean = np.mean(score_evolution, axis=1)
        
        # Find transitions with most/least stable scores
        most_stable_idx = np.argmin(score_std)
        least_stable_idx = np.argmax(score_std)
        
        return {
            'method': 'Dynamic ADC',
            'score_evolution_statistics': {
                'mean_score_std': np.mean(score_std),
                'median_score_std': np.median(score_std),
                'max_score_std': np.max(score_std),
                'min_score_std': np.min(score_std)
            },
            'most_stable_transition': {
                'index': int(most_stable_idx),
                'score_std': float(score_std[most_stable_idx]),
                'score_mean': float(score_mean[most_stable_idx])
            },
            'least_stable_transition': {
                'index': int(least_stable_idx),
                'score_std': float(score_std[least_stable_idx]),
                'score_mean': float(score_mean[least_stable_idx])
            }
        }

