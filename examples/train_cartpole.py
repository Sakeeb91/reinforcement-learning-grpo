#!/usr/bin/env python3
"""
Example training script for GRPO on CartPole environment.
"""

import numpy as np
from grpo import GRPOAgent, GRPOTrainer


def cartpole_group_assignment(state: np.ndarray, episode: int) -> int:
    """
    Group assignment function for CartPole based on cart position.
    Group 0: Cart position < -0.5 (left side)
    Group 1: Cart position between -0.5 and 0.5 (center)
    Group 2: Cart position > 0.5 (right side)
    """
    cart_position = state[0]
    if cart_position < -0.5:
        return 0
    elif cart_position > 0.5:
        return 2
    else:
        return 1


def main():
    # Environment parameters
    env_name = "CartPole-v1"
    state_dim = 4  # CartPole state dimension
    action_dim = 2  # CartPole action dimension
    
    # Training parameters
    num_episodes = 1000
    learning_rate = 3e-4
    gamma = 0.99
    eps_clip = 0.2
    group_robustness_weight = 0.2  # Weight for group robustness
    
    print("Initializing GRPO Agent...")
    
    # Create agent
    agent = GRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        eps_clip=eps_clip,
        group_robustness_weight=group_robustness_weight,
        device="cpu"
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        env_name=env_name,
        agent=agent,
        group_assignment_fn=cartpole_group_assignment,
        max_episode_steps=500
    )
    
    print(f"Training GRPO on {env_name} for {num_episodes} episodes...")
    print(f"Group robustness weight: {group_robustness_weight}")
    print("-" * 50)
    
    # Train the agent
    training_history = trainer.train(
        num_episodes=num_episodes,
        eval_freq=100
    )
    
    print("Training completed!")
    print(f"Final evaluation reward: {training_history['eval_rewards'][-1]:.2f}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_reward = trainer.evaluate(num_episodes=20)
    print(f"Final average reward over 20 episodes: {final_reward:.2f}")


if __name__ == "__main__":
    main()