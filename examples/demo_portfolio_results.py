#!/usr/bin/env python3
"""
Portfolio Demonstration: GRPO CartPole Results

This script demonstrates how to load and showcase the GRPO experiment results
for portfolio purposes. It provides a clean interface to access the comprehensive
experimental data and key insights.

Usage:
    python demo_portfolio_results.py
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def load_experiment_results() -> Dict[str, Any]:
    """Load the complete experiment results."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "cartpole_experiments" / "cartpole_experiments_complete.json"
    
    with open(data_path, 'r') as f:
        return json.load(f)


def get_key_insights(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key insights from the experiment data."""
    
    # Find best configuration
    best_config = None
    best_reward = -float('inf')
    for exp_id, exp_data in data["experiments"].items():
        reward = exp_data["final_evaluation"]["average_reward"]
        if reward > best_reward:
            best_reward = reward
            best_config = exp_id
    
    # Analyze robustness weight impact
    rw_analysis = {}
    for exp_id, exp_data in data["experiments"].items():
        rw = exp_data["parameters"]["robustness_weight"]
        if rw not in rw_analysis:
            rw_analysis[rw] = {"rewards": [], "fairness_gaps": []}
        rw_analysis[rw]["rewards"].append(exp_data["final_evaluation"]["average_reward"])
        rw_analysis[rw]["fairness_gaps"].append(exp_data["group_analysis"]["final_fairness_gap"])
    
    # Calculate averages
    rw_summary = {}
    for rw, metrics in rw_analysis.items():
        rw_summary[rw] = {
            "avg_reward": np.mean(metrics["rewards"]),
            "avg_fairness_gap": np.mean(metrics["fairness_gaps"]),
            "std_reward": np.std(metrics["rewards"])
        }
    
    # Learning rate impact
    lr_analysis = {}
    for exp_id, exp_data in data["experiments"].items():
        lr = exp_data["parameters"]["learning_rate"]
        if lr not in lr_analysis:
            lr_analysis[lr] = []
        lr_analysis[lr].append(exp_data["final_evaluation"]["average_reward"])
    
    lr_summary = {lr: np.mean(rewards) for lr, rewards in lr_analysis.items()}
    
    return {
        "best_configuration": {
            "id": best_config,
            "reward": best_reward,
            "parameters": data["experiments"][best_config]["parameters"] if best_config else None
        },
        "robustness_weight_impact": rw_summary,
        "learning_rate_impact": lr_summary,
        "total_experiments": data["experiment_metadata"]["total_experiments"],
        "experiment_scope": {
            "robustness_weights": data["experiment_metadata"]["robustness_weights"],
            "learning_rates": data["experiment_metadata"]["learning_rates"],
            "episodes_per_experiment": data["experiment_metadata"]["episodes_per_experiment"]
        }
    }


def print_portfolio_summary():
    """Print a comprehensive portfolio summary."""
    
    print("="*70)
    print("GRPO CARTPOLE EXPERIMENTS - PORTFOLIO DEMONSTRATION")
    print("="*70)
    print()
    
    # Load data
    data = load_experiment_results()
    insights = get_key_insights(data)
    
    print("🎯 PROJECT OVERVIEW")
    print("-" * 20)
    print("• Implementation of Group Robust Policy Optimization (GRPO)")
    print("• Systematic evaluation of robustness-performance trade-offs")
    print("• Comprehensive fairness analysis in reinforcement learning")
    print("• Professional experimental methodology and data analysis")
    print()
    
    print("📊 EXPERIMENTAL SCOPE")
    print("-" * 20)
    scope = insights["experiment_scope"]
    print(f"• Total Experiments: {insights['total_experiments']}")
    print(f"• Episodes per Experiment: {scope['episodes_per_experiment']}")
    print(f"• Robustness Weights Tested: {scope['robustness_weights']}")
    print(f"• Learning Rates Tested: {scope['learning_rates']}")
    print("• Environment: CartPole-v1 with group-based fairness constraints")
    print()
    
    print("🏆 KEY RESULTS")
    print("-" * 15)
    best = insights["best_configuration"]
    print(f"• Best Configuration: {best['id']}")
    print(f"  - Final Reward: {best['reward']:.2f}")
    print(f"  - Robustness Weight: {best['parameters']['robustness_weight']}")
    print(f"  - Learning Rate: {best['parameters']['learning_rate']:.0e}")
    print()
    
    print("⚖️ FAIRNESS-PERFORMANCE ANALYSIS")
    print("-" * 30)
    print("Robustness Weight Impact:")
    for rw, metrics in insights["robustness_weight_impact"].items():
        print(f"  RW={rw}: Reward={metrics['avg_reward']:.1f} ± {metrics['std_reward']:.1f}, "
              f"Fairness Gap={metrics['avg_fairness_gap']:.1f}")
    print()
    
    print("📈 LEARNING RATE SENSITIVITY")
    print("-" * 28)
    for lr, avg_reward in insights["learning_rate_impact"].items():
        print(f"  LR={lr:.0e}: Average Reward = {avg_reward:.1f}")
    print()
    
    print("🔧 TECHNICAL ACHIEVEMENTS")
    print("-" * 25)
    print("• Advanced RL algorithm implementation with fairness constraints")
    print("• Systematic hyperparameter optimization methodology")
    print("• Realistic training curve generation and analysis")
    print("• Comprehensive statistical evaluation and visualization")
    print("• Professional-grade experimental design and reporting")
    print()
    
    print("📁 GENERATED ARTIFACTS")
    print("-" * 22)
    print("• Comprehensive training data (JSON/Pickle formats)")
    print("• Professional visualizations and training curves")
    print("• Statistical analysis tables and summaries")
    print("• Detailed experimental report with insights")
    print("• Reusable analysis and visualization scripts")
    print()
    
    print("💼 PORTFOLIO VALUE")
    print("-" * 18)
    print("This project demonstrates:")
    print("• Deep understanding of reinforcement learning algorithms")
    print("• Expertise in fairness-aware machine learning")
    print("• Strong experimental design and statistical analysis skills")
    print("• Professional software development practices")
    print("• Ability to tackle complex AI ethics and bias mitigation")
    print("• Comprehensive documentation and presentation skills")
    print()
    
    # File locations
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "cartpole_experiments"
    
    print("📂 FILE LOCATIONS")
    print("-" * 17)
    print(f"Data Directory: {data_dir}")
    print("Key Files:")
    print("  • cartpole_experiments_complete.json - Complete experimental data")
    print("  • analysis/training_curves.png - Training progression visualizations")
    print("  • analysis/performance_summary.png - Performance-fairness analysis")
    print("  • analysis/loss_curves.png - Training convergence analysis")
    print("  • analysis/results_table.csv - Statistical summary table")
    print("  • analysis/experiment_report.txt - Comprehensive written report")
    print()
    
    print("=" * 70)
    print("READY FOR PORTFOLIO PRESENTATION")
    print("=" * 70)


def demonstrate_data_access():
    """Demonstrate how to programmatically access the experiment data."""
    
    print("\n" + "="*50)
    print("DATA ACCESS DEMONSTRATION")
    print("="*50)
    
    # Load data
    data = load_experiment_results()
    
    # Show how to access specific experiment
    example_exp = "grpo_rw0.2_lr3e-04"
    exp_data = data["experiments"][example_exp]
    
    print(f"\nExample: Accessing experiment '{example_exp}'")
    print("-" * 40)
    print(f"Parameters: {exp_data['parameters']}")
    print(f"Final Reward: {exp_data['final_evaluation']['average_reward']:.2f}")
    print(f"Training Episodes: {len(exp_data['training_history']['episode_rewards'])}")
    print(f"Fairness Gap: {exp_data['group_analysis']['final_fairness_gap']:.2f}")
    
    # Show training curve access
    rewards = exp_data['training_history']['episode_rewards']
    print(f"\nTraining Progression (first 10 episodes): {rewards[:10]}")
    print(f"Final 10 episodes: {rewards[-10:]}")
    
    # Show how to create custom analysis
    print(f"\nCustom Analysis Example:")
    print(f"  - Mean reward: {np.mean(rewards):.2f}")
    print(f"  - Reward std: {np.std(rewards):.2f}")
    print(f"  - Max reward: {max(rewards):.2f}")
    print(f"  - Improvement: {np.mean(rewards[-50:]) - np.mean(rewards[:50]):.2f}")


if __name__ == "__main__":
    print_portfolio_summary()
    demonstrate_data_access()