"""
Group Robust Policy Optimization (GRPO) Implementation
"""

from .grpo_agent import GRPOAgent
from .grpo_trainer import GRPOTrainer

__version__ = "0.1.0"
__all__ = ["GRPOAgent", "GRPOTrainer"]