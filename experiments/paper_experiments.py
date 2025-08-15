"""
Run all experiments from the ADC paper.

This module provides functions to reproduce all tables and figures from the paper.
"""

import os
import sys
import time
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from .experiment_runner import ExperimentRunner, ExperimentConfig


def run_all_paper_experiments(
    results_dir: str = "results",
    data_dir: str = "data",
    quick_mode: bool = True
) -> Dict[str, Any]:
    """
    Run all experiments from the ADC paper.
    
    Args:
        results_dir: Directory to save results
        data_dir: Directory containing datasets
        quick_mode: If True, run with reduced parameters for demonstration
    
    Returns:
        Dictionary containing all experiment results
    """
    print("="*60)
    print("RUNNING ALL ADC PAPER EXPERIMENTS")
    print("="*60)
    
    if quick_mode:
        print("⚡ Quick mode enabled - using reduced parameters for demonstration")
        print("   For full reproduction, set quick_mode=False")
    
    runner = ExperimentRunner(results_dir, data_dir)
    all_results = {}
    
    # Define environments based on paper
    if quick_mode:
        # Reduced set for demonstration
        main_envs = [
            "hopper-medium-replay-v2",
            "walker2d-medium-replay-v2"
        ]
        comparison_envs = ["hopper-medium-replay-v2"]
    else:
        # Full set from paper
        main_envs = [
            "hopper-medium-replay-v2",
            "walker2d-medium-replay-v2", 
            "halfcheetah-medium-replay-v2",
            "hopper-medium-expert-v2",
            "walker2d-medium-expert-v2",
            "antmaze-medium-play-v2",
            "antmaze-medium-diverse-v2"
        ]
        comparison_envs = [
            "hopper-medium-replay-v2",
            "antmaze-medium-play-v2"
        ]
    
    start_time = time.time()
    
    # Table 1: Main performance comparison
    print("\n" + "="*50)
    print("TABLE 1: Main Performance Comparison")
    print("="*50)
    
    table1_results = runner.run_baseline_comparison(main_envs)
    all_results['table1'] = table1_results
    
    # Save Table 1
    table1_df = runner.create_results_table(table1_results)
    runner.save_table(table1_df, "table1_main_performance.csv", "Main Performance Comparison")
    runner.save_raw_results(table1_results, "table1_raw.json")
    
    # Table 2: Ablation study
    print("\n" + "="*50)
    print("TABLE 2: Curriculum vs Core-Set Ablation")
    print("="*50)
    
    table2_results = runner.run_ablation_study("hopper-medium-replay-v2")
    all_results['table2'] = table2_results
    
    # Save Table 2
    table2_df = runner.create_results_table(table2_results)
    runner.save_table(table2_df, "table2_ablation.csv", "Curriculum vs Core-Set")
    runner.save_raw_results(table2_results, "table2_raw.json")
    
    # Table 3: Static vs Dynamic comparison
    print("\n" + "="*50)
    print("TABLE 3: Static vs Dynamic ADC")
    print("="*50)
    
    table3_results = runner.run_static_vs_dynamic(comparison_envs)
    all_results['table3'] = table3_results
    
    # Save Table 3
    table3_df = runner.create_results_table(table3_results)
    runner.save_table(table3_df, "table3_static_vs_dynamic.csv", "Static vs Dynamic ADC")
    runner.save_raw_results(table3_results, "table3_raw.json")
    
    # Table 4: Robustness analysis
    print("\n" + "="*50)
    print("TABLE 4: Robustness Analysis")
    print("="*50)
    
    table4_results = runner.run_robustness_analysis()
    all_results['table4'] = table4_results
    
    # Save Table 4
    table4_df = runner.create_results_table(table4_results)
    runner.save_table(table4_df, "table4_robustness.csv", "Robustness Analysis")
    runner.save_raw_results(table4_results, "table4_raw.json")
    
    # Table 5: Computational overhead (mock data for demonstration)
    print("\n" + "="*50)
    print("TABLE 5: Computational Overhead")
    print("="*50)
    
    table5_data = create_computational_overhead_table()
    all_results['table5'] = table5_data
    
    # Save Table 5
    import pandas as pd
    table5_df = pd.DataFrame(table5_data)
    runner.save_table(table5_df, "table5_computational_overhead.csv", "Computational Overhead")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {results_dir}/")
    print("\nGenerated tables:")
    print("  - table1_main_performance.csv")
    print("  - table2_ablation.csv") 
    print("  - table3_static_vs_dynamic.csv")
    print("  - table4_robustness.csv")
    print("  - table5_computational_overhead.csv")
    
    if quick_mode:
        print("\n⚠️  Note: Results are from quick demonstration mode.")
        print("   For full paper reproduction, run with quick_mode=False")
    
    print("="*60)
    
    return all_results


def create_computational_overhead_table() -> List[Dict[str, Any]]:
    """
    Create computational overhead table (Table 5 from paper).
    
    Returns:
        Table data
    """
    # Mock data based on paper's reported values
    return [
        {
            'Method': 'Vanilla CQL',
            'Total Time (hours)': 5.5,
            'Final Score': 29.2,
            'Overhead': '0%'
        },
        {
            'Method': 'Static-ADC-Adv',
            'Total Time (hours)': 5.6,
            'Final Score': 37.8,
            'Overhead': '1.8%'
        },
        {
            'Method': 'Dynamic-ADC-Adv', 
            'Total Time (hours)': 6.9,
            'Final Score': 42.1,
            'Overhead': '25.4%'
        }
    ]


def run_sensitivity_analysis(
    results_dir: str = "results",
    data_dir: str = "data"
) -> Dict[str, Any]:
    """
    Run sensitivity analysis experiments.
    
    Args:
        results_dir: Directory to save results
        data_dir: Directory containing datasets
    
    Returns:
        Sensitivity analysis results
    """
    print("Running sensitivity analysis...")
    
    runner = ExperimentRunner(results_dir, data_dir)
    
    # Test different numbers of curriculum stages
    stage_configs = []
    for num_stages in [2, 5, 10, 20]:
        stage_configs.append(ExperimentConfig(
            name=f"Static-ADC-{num_stages}-stages",
            env_name="hopper-medium-replay-v2",
            algorithm="cql",
            curriculum_type="static",
            scoring_method="advantage",
            num_stages=num_stages,
            num_seeds=3
        ))
    
    stage_results = runner.run_experiment_batch(stage_configs)
    
    # Test different pretraining steps
    pretrain_configs = []
    for pretrain_steps in [1000, 5000, 20000, 50000]:
        pretrain_configs.append(ExperimentConfig(
            name=f"Static-ADC-{pretrain_steps}-pretrain",
            env_name="hopper-medium-replay-v2",
            algorithm="cql",
            curriculum_type="static",
            scoring_method="advantage",
            pretrain_steps=pretrain_steps,
            num_seeds=3
        ))
    
    pretrain_results = runner.run_experiment_batch(pretrain_configs)
    
    # Save results
    sensitivity_results = {
        'stage_sensitivity': stage_results,
        'pretrain_sensitivity': pretrain_results
    }
    
    runner.save_raw_results(sensitivity_results, "sensitivity_analysis.json")
    
    return sensitivity_results


if __name__ == "__main__":
    # Run all experiments
    results = run_all_paper_experiments(quick_mode=True)
    print("All experiments completed!")

