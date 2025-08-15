"""
Baseline algorithms using open-source implementations.

This module provides wrappers around established open-source implementations
of offline RL algorithms like CQL, IQL, and BC.
"""

from .simple_baselines import create_simple_algorithm, SimpleBC, SimpleCQL, SimpleIQL

__all__ = ['create_simple_algorithm', 'SimpleBC', 'SimpleCQL', 'SimpleIQL']

