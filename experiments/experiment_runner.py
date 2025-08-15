"""
Comprehensive experiment runner for ADC paper reproduction.

This module provides functionality to run all experiments from the paper
and generate the corresponding tables and figures.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adc import ADCWrapper, ADCExperimentConfig
from utils.data_utils import load_dataset, create_synthetic_dataset, save_dataset
from utils.metrics import save_results_csv, create_results_table, compute_statistics


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    env_name: str
    algorithm: str
    curriculum_type: str
    scoring_method: str
    num_stages: int = 5
    total_steps: int = 10000  # Reduced for demo
    pretrain_steps: int = 1000
    eval_episodes: int = 5
    num_seeds: int = 3  # Reduced for demo
    device: str = "cpu"


class ExperimentRunner:
    """
    Runner for systematic experiments to reproduce paper results.
    """
    
    def __init__(self, 
                 results_dir: str = "results",
                 data_dir: str = "data",
                 num_workers: int = None):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to save results
            data_dir: Directory containing datasets
            num_workers: Number of parallel workers (None for auto)
        """
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.num_workers = num_workers or min(4, mp.cpu_count())
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/tables", exist_ok=True)
        os.makedirs(f"{results_dir}/figures", exist_ok=True)
        os.makedirs(f"{results_dir}/raw", exist_ok=True)
        
        print(f"Experiment runner initialized:")
        print(f"  - Results directory: {results_dir}")
        print(f"  - Data directory: {data_dir}")
        print(f"  - Number of workers: {self.num_workers}")
    
    def run_single_experiment(self, config: ExperimentConfig, seed: int) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration and seed.
        
        Args:
            config: Experiment configuration
            seed: Random seed
        
        Returns:
            Experiment results
        """
        try:
            # Load dataset
            dataset_path = os.path.join(self.data_dir, f"{config.env_name}.pkl")
            if not os.path.exists(dataset_path):
                print(f"Dataset {dataset_path} not found, creating synthetic dataset...")
                # Create synthetic dataset
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
                
                env_config = env_configs.get(config.env_name, {'obs_dim': 17, 'action_dim': 6})
                env_base = config.env_name.split('-')[0]
                dataset_type = '-'.join(config.env_name.split('-')[1:-1])
                
                dataset = create_synthetic_dataset(
                    env_base, dataset_type, 
                    num_trajectories=200,  # Smaller for demo
                    obs_dim=env_config['obs_dim'],
                    action_dim=env_config['action_dim'],
                    seed=seed
                )
                save_dataset(dataset, dataset_path)
            else:
                dataset = load_dataset(dataset_path)
            
            # Create ADC configuration
            adc_config = ADCExperimentConfig(
                curriculum_type=config.curriculum_type,
                scoring_method=config.scoring_method,
                algorithm=config.algorithm,
                env_name=config.env_name,
                num_stages=config.num_stages,
                total_steps=config.total_steps,
                pretrain_steps=config.pretrain_steps,
                eval_episodes=config.eval_episodes,
                device=config.device,
                seed=seed
            )
            
            # Run experiment
            wrapper = ADCWrapper(adc_config)
            results = wrapper.run_experiment(dataset)
            
            # Add metadata
            results['config'] = asdict(config)
            results['seed'] = seed
            results['success'] = True
            
            return results
            
        except Exception as e:
            print(f"Experiment failed: {config.name}, seed {seed}, error: {str(e)}")
            return {
                'config': asdict(config),
                'seed': seed,
                'success': False,
                'error': str(e),
                'final_score': 0.0
            }
    
    def run_experiment_batch(self, 
                           configs: List[ExperimentConfig],
                           parallel: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run a batch of experiments.
        
        Args:
            configs: List of experiment configurations
            parallel: Whether to run experiments in parallel
        
        Returns:
            Dictionary mapping experiment names to results
        """
        all_results = {}
        
        # Prepare all experiment tasks
        tasks = []
        for config in configs:
            for seed in range(config.num_seeds):
                tasks.append((config, seed))
        
        print(f"Running {len(tasks)} experiment tasks...")
        
        if parallel and self.num_workers > 1:
            # Run in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_task = {
                    executor.submit(self.run_single_experiment, config, seed): (config, seed)
                    for config, seed in tasks
                }
                
                completed = 0
                for future in as_completed(future_to_task):
                    config, seed = future_to_task[future]
                    result = future.result()
                    
                    if config.name not in all_results:
                        all_results[config.name] = []
                    all_results[config.name].append(result)
                    
                    completed += 1
                    print(f"Completed {completed}/{len(tasks)}: {config.name} (seed {seed})")
        else:
            # Run sequentially
            for i, (config, seed) in enumerate(tasks):
                result = self.run_single_experiment(config, seed)
                
                if config.name not in all_results:
                    all_results[config.name] = []
                all_results[config.name].append(result)
                
                print(f"Completed {i+1}/{len(tasks)}: {config.name} (seed {seed})")
        
        return all_results
    
    def save_raw_results(self, results: Dict[str, List[Dict[str, Any]]], filename: str):
        """Save raw results to JSON file."""
        filepath = os.path.join(self.results_dir, "raw", filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Raw results saved to {filepath}")
    
    def create_results_table(self, 
                           results: Dict[str, List[Dict[str, Any]]],
                           metric: str = 'final_score') -> pd.DataFrame:
        """
        Create a results table from experiment results.
        
        Args:
            results: Experiment results
            metric: Metric to extract
        
        Returns:
            Results DataFrame
        """
        table_data = []
        
        for exp_name, exp_results in results.items():
            # Extract scores for successful runs
            scores = [r[metric] for r in exp_results if r.get('success', False)]
            
            if scores:
                stats = compute_statistics(scores)
                table_data.append({
                    'Experiment': exp_name,
                    'Mean': stats['mean'],
                    'Std': stats['std'],
                    'Min': stats['min'],
                    'Max': stats['max'],
                    'Num_Seeds': len(scores),
                    'Formatted': f"{stats['mean']:.1f} ± {stats['std']:.1f}"
                })
            else:
                table_data.append({
                    'Experiment': exp_name,
                    'Mean': 0.0,
                    'Std': 0.0,
                    'Min': 0.0,
                    'Max': 0.0,
                    'Num_Seeds': 0,
                    'Formatted': "Failed"
                })
        
        return pd.DataFrame(table_data)
    
    def save_table(self, df: pd.DataFrame, filename: str, title: str = "Results"):
        """Save results table to CSV."""
        filepath = os.path.join(self.results_dir, "tables", filename)
        df.to_csv(filepath, index=False)
        print(f"Table saved to {filepath}")
        
        # Also save a formatted version
        formatted_filepath = filepath.replace('.csv', '_formatted.csv')
        formatted_df = df[['Experiment', 'Formatted']].copy()
        formatted_df.columns = ['Method', 'Score']
        formatted_df.to_csv(formatted_filepath, index=False)
        print(f"Formatted table saved to {formatted_filepath}")
    
    def run_baseline_comparison(self, env_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run baseline comparison experiments (Table 1 equivalent).
        
        Args:
            env_names: List of environment names
        
        Returns:
            Experiment results
        """
        print("Running baseline comparison experiments...")
        
        configs = []
        
        # Baseline algorithms
        for env_name in env_names:
            for algorithm in ['bc', 'cql', 'iql']:
                # Baseline (no curriculum)
                configs.append(ExperimentConfig(
                    name=f"{algorithm.upper()}-Baseline-{env_name}",
                    env_name=env_name,
                    algorithm=algorithm,
                    curriculum_type="none",  # Will be handled specially
                    scoring_method="advantage",
                    num_seeds=3
                ))
                
                # Static ADC with advantage scoring
                configs.append(ExperimentConfig(
                    name=f"Static-ADC-Adv-{algorithm.upper()}-{env_name}",
                    env_name=env_name,
                    algorithm=algorithm,
                    curriculum_type="static",
                    scoring_method="advantage",
                    num_seeds=3
                ))
        
        return self.run_experiment_batch(configs)
    
    def run_ablation_study(self, env_name: str = "hopper-medium-replay-v2") -> Dict[str, List[Dict[str, Any]]]:
        """
        Run ablation study (Table 2 equivalent).
        
        Args:
            env_name: Environment to test on
        
        Returns:
            Experiment results
        """
        print("Running ablation study...")
        
        configs = [
            # Full dataset baseline
            ExperimentConfig(
                name="CQL-Full-Dataset",
                env_name=env_name,
                algorithm="cql",
                curriculum_type="none",
                scoring_method="advantage"
            ),
            
            # Core-set (top 20% only)
            ExperimentConfig(
                name="CQL-Core-Set-20%",
                env_name=env_name,
                algorithm="cql",
                curriculum_type="static",
                scoring_method="advantage",
                num_stages=1  # Only use top 20%
            ),
            
            # Static ADC curriculum
            ExperimentConfig(
                name="Static-ADC-Curriculum",
                env_name=env_name,
                algorithm="cql",
                curriculum_type="static",
                scoring_method="advantage",
                num_stages=5
            )
        ]
        
        return self.run_experiment_batch(configs)
    
    def run_static_vs_dynamic(self, env_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run static vs dynamic comparison (Table 3 equivalent).
        
        Args:
            env_names: List of environment names
        
        Returns:
            Experiment results
        """
        print("Running static vs dynamic comparison...")
        
        configs = []
        
        for env_name in env_names:
            # Vanilla CQL baseline
            configs.append(ExperimentConfig(
                name=f"Vanilla-CQL-{env_name}",
                env_name=env_name,
                algorithm="cql",
                curriculum_type="none",
                scoring_method="advantage"
            ))
            
            # Static ADC
            configs.append(ExperimentConfig(
                name=f"Static-ADC-{env_name}",
                env_name=env_name,
                algorithm="cql",
                curriculum_type="static",
                scoring_method="advantage"
            ))
            
            # Dynamic ADC
            configs.append(ExperimentConfig(
                name=f"Dynamic-ADC-{env_name}",
                env_name=env_name,
                algorithm="cql",
                curriculum_type="dynamic",
                scoring_method="advantage"
            ))
        
        return self.run_experiment_batch(configs)
    
    def run_robustness_analysis(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run robustness analysis on noisy data (Table 4 equivalent).
        
        Returns:
            Experiment results
        """
        print("Running robustness analysis...")
        
        configs = [
            # Vanilla CQL on noisy data
            ExperimentConfig(
                name="Vanilla-CQL-Noisy",
                env_name="hopper-99r-1e",
                algorithm="cql",
                curriculum_type="none",
                scoring_method="advantage"
            ),
            
            # Static ADC on noisy data
            ExperimentConfig(
                name="Static-ADC-Noisy",
                env_name="hopper-99r-1e",
                algorithm="cql",
                curriculum_type="static",
                scoring_method="advantage"
            )
        ]
        
        return self.run_experiment_batch(configs)


def run_single_experiment_wrapper(args):
    """Wrapper function for multiprocessing."""
    config, seed, results_dir, data_dir = args
    runner = ExperimentRunner(results_dir, data_dir, num_workers=1)
    return runner.run_single_experiment(config, seed)

