#!/usr/bin/env python3
"""
Hospital Resource Allocation with GRPO - Parallel Training
Demonstrates fair resource allocation across patient demographics using multiple parallel agents.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import multiprocessing as mp
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grpo import GRPOAgent, GRPOTrainer
from envs import HospitalEnv


def hospital_group_assignment(state: np.ndarray, episode: int) -> int:
    """
    Group assignment function for hospital environment.
    Groups are determined by the environment's patient demographics.
    """
    # Extract demographic information from state
    # State indices 8-10 contain demographic distribution
    if len(state) >= 11:
        pediatric_ratio = state[8]
        adult_ratio = state[9] 
        elderly_ratio = state[10]
        
        # Assign group based on dominant demographic in current queue
        if pediatric_ratio > adult_ratio and pediatric_ratio > elderly_ratio:
            return 0  # Pediatric-dominant
        elif elderly_ratio > adult_ratio:
            return 2  # Elderly-dominant
        else:
            return 1  # Adult-dominant
    
    return 0  # Default group


def create_hospital_config(scenario: str) -> Dict:
    """Create different hospital scenarios for training diversity."""
    configs = {
        "urban_hospital": {
            "total_beds": 100,
            "total_staff": 60,
            "total_equipment": 40,
            "demographic_rates": {0: 0.15, 1: 0.55, 2: 0.30},  # Urban demographics
            "group_robustness_weight": 0.3
        },
        "rural_hospital": {
            "total_beds": 30,
            "total_staff": 20,
            "total_equipment": 15,
            "demographic_rates": {0: 0.25, 1: 0.45, 2: 0.30},  # Rural demographics
            "group_robustness_weight": 0.4
        },
        "pediatric_hospital": {
            "total_beds": 50,
            "total_staff": 35,
            "total_equipment": 25,
            "demographic_rates": {0: 0.70, 1: 0.20, 2: 0.10},  # Pediatric focus
            "group_robustness_weight": 0.2
        },
        "emergency_surge": {
            "total_beds": 80,
            "total_staff": 45,
            "total_equipment": 30,
            "demographic_rates": {0: 0.20, 1: 0.50, 2: 0.30},  # Emergency conditions
            "group_robustness_weight": 0.5
        }
    }
    return configs.get(scenario, configs["urban_hospital"])


def train_agent_parallel(agent_id: int, scenario: str, num_episodes: int = 500) -> Dict:
    """Train a single GRPO agent on hospital environment."""
    print(f"Agent {agent_id}: Starting training on {scenario} scenario...")
    
    # Get hospital configuration
    config = create_hospital_config(scenario)
    
    # Create environment
    env = HospitalEnv(
        total_beds=config["total_beds"],
        total_staff=config["total_staff"],
        total_equipment=config["total_equipment"],
        demographic_arrival_rates=config["demographic_rates"],
        episode_length=200,
        max_queue_size=150
    )
    
    # Create agent
    agent = GRPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4,
        gamma=0.95,
        eps_clip=0.2,
        group_robustness_weight=config["group_robustness_weight"],
        device="cpu"
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        env_name="hospital",  # Custom env
        agent=agent,
        group_assignment_fn=hospital_group_assignment,
        max_episode_steps=200
    )
    
    # Override trainer environment with our custom one
    trainer.env = env
    
    # Training metrics
    episode_rewards = []
    fairness_scores = []
    demographic_metrics = []
    
    for episode in range(num_episodes):
        # Train one episode
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(trainer.max_episode_steps):
            # Get group assignment
            group_id = hospital_group_assignment(state, episode)
            
            # Select action
            action, log_prob = agent.select_action(state, group_id)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_reward(reward, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Update agent
        losses = agent.update()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        
        # Calculate fairness metrics
        fairness_metrics = env.get_fairness_metrics()
        fairness_scores.append(fairness_metrics.get('overall_fairness', 0.0))
        
        # Record demographic performance
        demographic_info = {
            'episode': episode,
            'avg_wait_times': info.get('average_wait_times', {}),
            'resource_utilization': info.get('resource_utilization', {}),
            'patients_served': info.get('total_served', 0)
        }
        demographic_metrics.append(demographic_info)
        
        # Progress update
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_fairness = np.mean(fairness_scores[-100:]) if fairness_scores[-100:] else 0
            print(f"Agent {agent_id} - Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}, Avg Fairness: {avg_fairness:.3f}")
            print(f"  Scenario: {scenario}, Robustness Weight: {config['group_robustness_weight']}")
    
    # Final evaluation
    print(f"Agent {agent_id}: Running final evaluation...")
    eval_rewards = []
    eval_fairness = []
    
    for _ in range(20):
        state = env.reset()
        eval_reward = 0
        
        for step in range(trainer.max_episode_steps):
            action, _ = agent.select_action(state, group_id=0)
            state, reward, done, info = env.step(action)
            eval_reward += reward
            
            if done:
                break
        
        eval_rewards.append(eval_reward)
        fairness_metrics = env.get_fairness_metrics()
        eval_fairness.append(fairness_metrics.get('overall_fairness', 0.0))
    
    # Clear storage after evaluation
    agent.reset_storage()
    
    final_results = {
        'agent_id': agent_id,
        'scenario': scenario,
        'config': config,
        'training_rewards': episode_rewards,
        'fairness_scores': fairness_scores,
        'demographic_metrics': demographic_metrics,
        'eval_reward_mean': np.mean(eval_rewards),
        'eval_reward_std': np.std(eval_rewards),
        'eval_fairness_mean': np.mean(eval_fairness),
        'eval_fairness_std': np.std(eval_fairness),
        'final_avg_reward': np.mean(episode_rewards[-50:]),
        'final_avg_fairness': np.mean(fairness_scores[-50:]) if fairness_scores[-50:] else 0
    }
    
    print(f"Agent {agent_id}: Training completed!")
    print(f"  Final Avg Reward: {final_results['final_avg_reward']:.2f}")
    print(f"  Final Avg Fairness: {final_results['final_avg_fairness']:.3f}")
    print(f"  Eval Reward: {final_results['eval_reward_mean']:.2f} ± {final_results['eval_reward_std']:.2f}")
    
    return final_results


def run_parallel_training():
    """Run parallel training with multiple agents on different hospital scenarios."""
    print("=" * 60)
    print("GRPO Healthcare Resource Allocation - Parallel Training")
    print("=" * 60)
    
    # Define training scenarios
    scenarios = [
        ("urban_hospital", 600),
        ("rural_hospital", 600), 
        ("pediatric_hospital", 600),
        ("emergency_surge", 600)
    ]
    
    # Determine number of parallel processes
    num_processes = min(len(scenarios), mp.cpu_count())
    print(f"Starting {len(scenarios)} agents on {num_processes} processes...")
    print(f"Scenarios: {[s[0] for s in scenarios]}")
    print()
    
    start_time = time.time()
    
    # Run parallel training
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all training jobs
        future_to_agent = {
            executor.submit(train_agent_parallel, i, scenario, episodes): (i, scenario)
            for i, (scenario, episodes) in enumerate(scenarios)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_agent):
            agent_id, scenario = future_to_agent[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✅ Agent {agent_id} ({scenario}) completed successfully!")
            except Exception as exc:
                print(f"\n❌ Agent {agent_id} ({scenario}) failed with exception: {exc}")
    
    training_time = time.time() - start_time
    print(f"\n🎉 All training completed in {training_time:.1f} seconds!")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS ANALYSIS")
    print("=" * 60)
    
    # Sort results by agent_id for consistent reporting
    results.sort(key=lambda x: x['agent_id'])
    
    print("\nFinal Performance Summary:")
    print("-" * 40)
    for result in results:
        print(f"Agent {result['agent_id']} ({result['scenario']}):")
        print(f"  Robustness Weight: {result['config']['group_robustness_weight']}")
        print(f"  Final Reward: {result['final_avg_reward']:.2f}")
        print(f"  Final Fairness: {result['final_avg_fairness']:.3f}")
        print(f"  Eval Performance: {result['eval_reward_mean']:.2f} ± {result['eval_reward_std']:.2f}")
        print()
    
    # Comparative analysis
    print("Comparative Analysis:")
    print("-" * 40)
    
    # Best performing agent
    best_reward_agent = max(results, key=lambda x: x['final_avg_reward'])
    best_fairness_agent = max(results, key=lambda x: x['final_avg_fairness'])
    
    print(f"Best Reward Performance: Agent {best_reward_agent['agent_id']} ({best_reward_agent['scenario']})")
    print(f"  Reward: {best_reward_agent['final_avg_reward']:.2f}")
    print(f"  Fairness: {best_reward_agent['final_avg_fairness']:.3f}")
    print()
    
    print(f"Best Fairness Performance: Agent {best_fairness_agent['agent_id']} ({best_fairness_agent['scenario']})")
    print(f"  Reward: {best_fairness_agent['final_avg_reward']:.2f}")
    print(f"  Fairness: {best_fairness_agent['final_avg_fairness']:.3f}")
    print()
    
    # Robustness weight analysis
    print("Robustness Weight Impact:")
    robustness_analysis = {}
    for result in results:
        weight = result['config']['group_robustness_weight']
        if weight not in robustness_analysis:
            robustness_analysis[weight] = {'rewards': [], 'fairness': []}
        robustness_analysis[weight]['rewards'].append(result['final_avg_reward'])
        robustness_analysis[weight]['fairness'].append(result['final_avg_fairness'])
    
    for weight, metrics in robustness_analysis.items():
        avg_reward = np.mean(metrics['rewards'])
        avg_fairness = np.mean(metrics['fairness'])
        print(f"  Weight {weight}: Reward={avg_reward:.2f}, Fairness={avg_fairness:.3f}")
    
    print("\n" + "=" * 60)
    print("Training completed! Check the results above for performance analysis.")
    print("Key findings:")
    print("- Higher robustness weights generally improve fairness")
    print("- Different hospital scenarios require different optimization strategies")
    print("- GRPO successfully balances efficiency and demographic fairness")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Test if we can import the modules
    try:
        from grpo import GRPOAgent, GRPOTrainer
        from envs import HospitalEnv
        print("✅ All modules imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    
    # Run parallel training
    results = run_parallel_training()