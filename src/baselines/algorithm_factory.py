"""
Factory for creating baseline algorithms with consistent interface.
"""

from typing import Dict, Any
from .d3rlpy_wrapper import D3RLPyWrapper


def create_algorithm(algorithm_name: str, **kwargs) -> D3RLPyWrapper:
    """
    Create a baseline algorithm instance.
    
    Args:
        algorithm_name: Name of the algorithm ('cql', 'iql', 'bc')
        **kwargs: Additional arguments for algorithm initialization
        
    Returns:
        Algorithm wrapper instance
    """
    supported_algorithms = ['cql', 'iql', 'bc']
    
    if algorithm_name.lower() not in supported_algorithms:
        raise ValueError(f"Algorithm {algorithm_name} not supported. "
                        f"Supported algorithms: {supported_algorithms}")
    
    return D3RLPyWrapper(algorithm_name, **kwargs)


def get_default_config(algorithm_name: str) -> Dict[str, Any]:
    """
    Get default configuration for an algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        
    Returns:
        Default configuration dictionary
    """
    configs = {
        'cql': {
            'alpha': 1.0,
            'batch_size': 256,
            'learning_rate': 3e-4,
            'target_update_interval': 8000,
        },
        'iql': {
            'expectile': 0.7,
            'batch_size': 256,
            'learning_rate': 3e-4,
            'target_update_interval': 8000,
        },
        'bc': {
            'batch_size': 256,
            'learning_rate': 1e-3,
        }
    }
    
    return configs.get(algorithm_name.lower(), {})

