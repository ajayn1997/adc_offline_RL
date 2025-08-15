"""
Factory for creating scoring functions.
"""

from typing import Dict, Any
from .td_error_scorer import TDErrorScorer
from .advantage_scorer import AdvantageScorer
from .base_scorer import BaseScorer


def create_scorer(scorer_name: str, **kwargs) -> BaseScorer:
    """
    Create a scoring function instance.
    
    Args:
        scorer_name: Name of the scorer ('td_error', 'advantage')
        **kwargs: Additional arguments for scorer initialization
    
    Returns:
        Scorer instance
    """
    scorer_name = scorer_name.lower()
    
    if scorer_name in ['td_error', 'td-error', 'tderror']:
        return TDErrorScorer(**kwargs)
    elif scorer_name in ['advantage', 'adv']:
        return AdvantageScorer(**kwargs)
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}. "
                        f"Available scorers: ['td_error', 'advantage']")


def get_available_scorers() -> Dict[str, str]:
    """
    Get available scoring functions with descriptions.
    
    Returns:
        Dictionary mapping scorer names to descriptions
    """
    return {
        'td_error': 'TD-error based scoring: |r + γ max_a\' Q(s\',a\') - Q(s,a)|',
        'advantage': 'Advantage-based scoring: Q(s,a) - V(s)'
    }


def get_scorer_config(scorer_name: str) -> Dict[str, Any]:
    """
    Get default configuration for a scorer.
    
    Args:
        scorer_name: Name of the scorer
    
    Returns:
        Default configuration dictionary
    """
    configs = {
        'td_error': {
            'gamma': 0.99,
            'device': 'cpu'
        },
        'advantage': {
            'num_action_samples': 10,
            'device': 'cpu'
        }
    }
    
    return configs.get(scorer_name.lower(), {})

