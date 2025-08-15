"""
Metrics and evaluation utilities for offline RL.
"""

import numpy as np
from typing import Dict, List, Optional


def compute_normalized_score(score: float, random_score: float, expert_score: float) -> float:
    """
    Compute normalized score as used in D4RL.
    
    Args:
        score: Raw score to normalize
        random_score: Score of random policy
        expert_score: Score of expert policy
    
    Returns:
        Normalized score (0 = random, 100 = expert)
    """
    if expert_score == random_score:
        return 0.0
    
    normalized = 100.0 * (score - random_score) / (expert_score - random_score)
    return normalized


def compute_d4rl_score(env_name: str, raw_score: float) -> float:
    """
    Compute D4RL normalized score for a given environment.
    
    Args:
        env_name: Environment name
        raw_score: Raw evaluation score
    
    Returns:
        D4RL normalized score
    """
    # Reference scores from D4RL paper (approximate)
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
    
    if env_name not in reference_scores:
        # Default normalization
        return raw_score
    
    random_score = reference_scores[env_name]['random']
    expert_score = reference_scores[env_name]['expert']
    
    return compute_normalized_score(raw_score, random_score, expert_score)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics for a list of values.
    
    Args:
        values: List of numerical values
    
    Returns:
        Dictionary with mean, std, min, max, median
    """
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    
    values_array = np.array(values)
    
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array))
    }


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute confidence interval for a list of values.
    
    Args:
        values: List of numerical values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Dictionary with lower and upper bounds
    """
    if not values:
        return {'lower': 0.0, 'upper': 0.0}
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array)
    n = len(values_array)
    
    # Use t-distribution for small samples, normal for large samples
    if n < 30:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * std / np.sqrt(n)
    else:
        # Normal approximation
        z_value = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin_error = z_value * std / np.sqrt(n)
    
    return {
        'lower': float(mean - margin_error),
        'upper': float(mean + margin_error),
        'margin_error': float(margin_error)
    }


def format_score_with_std(mean: float, std: float, decimals: int = 1) -> str:
    """
    Format score with standard deviation for reporting.
    
    Args:
        mean: Mean score
        std: Standard deviation
        decimals: Number of decimal places
    
    Returns:
        Formatted string like "85.3 ± 2.1"
    """
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def create_results_table(results: Dict[str, Dict[str, List[float]]], 
                        environments: List[str],
                        methods: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Create a formatted results table for reporting.
    
    Args:
        results: Nested dictionary with results[env][method] = [scores]
        environments: List of environment names
        methods: List of method names
    
    Returns:
        Formatted table as nested dictionary
    """
    table = {}
    
    for env in environments:
        table[env] = {}
        for method in methods:
            if env in results and method in results[env]:
                scores = results[env][method]
                if scores:
                    mean = np.mean(scores)
                    std = np.std(scores)
                    table[env][method] = format_score_with_std(mean, std)
                else:
                    table[env][method] = "N/A"
            else:
                table[env][method] = "N/A"
    
    return table


def save_results_csv(results_table: Dict[str, Dict[str, str]], 
                    filepath: str,
                    title: str = "Results") -> None:
    """
    Save results table to CSV file.
    
    Args:
        results_table: Results table from create_results_table
        filepath: Output CSV file path
        title: Title for the table
    """
    import csv
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Get all methods (columns)
    all_methods = set()
    for env_results in results_table.values():
        all_methods.update(env_results.keys())
    methods = sorted(list(all_methods))
    
    # Write CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Environment'] + methods)
        
        # Data rows
        for env, env_results in results_table.items():
            row = [env]
            for method in methods:
                row.append(env_results.get(method, 'N/A'))
            writer.writerow(row)
    
    print(f"Results saved to {filepath}")


def compute_improvement_percentage(baseline_score: float, improved_score: float) -> float:
    """
    Compute percentage improvement over baseline.
    
    Args:
        baseline_score: Baseline score
        improved_score: Improved score
    
    Returns:
        Percentage improvement
    """
    if baseline_score == 0:
        return 0.0 if improved_score == 0 else float('inf')
    
    return ((improved_score - baseline_score) / abs(baseline_score)) * 100

