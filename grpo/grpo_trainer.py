import gym
import numpy as np
from typing import Dict, List, Optional, Callable
from .grpo_agent import GRPOAgent


class GRPOTrainer:
    def __init__(
        self,
        env_name: str,
        agent: GRPOAgent,
        group_assignment_fn: Optional[Callable] = None,
        max_episode_steps: int = 1000
    ):
        self.env = gym.make(env_name)
        self.agent = agent
        self.group_assignment_fn = group_assignment_fn or self._default_group_assignment
        self.max_episode_steps = max_episode_steps
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
    
    def _default_group_assignment(self, state: np.ndarray, episode: int) -> int:
        """Default group assignment based on state features."""
        # Simple example: assign groups based on state magnitude
        state_norm = np.linalg.norm(state)
        if state_norm < 1.0:
            return 0  # Low magnitude states
        elif state_norm < 2.0:
            return 1  # Medium magnitude states
        else:
            return 2  # High magnitude states
    
    def train_episode(self, episode: int) -> Dict[str, float]:
        """Train for one episode."""
        # Handle both old and new gym API
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result
            
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.max_episode_steps):
            # Assign group for current state
            group_id = self.group_assignment_fn(state, episode)
            
            # Select action
            action, log_prob = self.agent.select_action(state, group_id)
            
            # Take step in environment
            step_result = self.env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            else:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            # Store experience
            self.agent.store_reward(reward, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Update agent
        losses = self.agent.update()
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            **losses
        }
    
    def train(self, num_episodes: int, eval_freq: int = 100) -> Dict[str, List[float]]:
        """Train the agent for multiple episodes."""
        training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "eval_rewards": []
        }
        
        for episode in range(num_episodes):
            # Training step
            metrics = self.train_episode(episode)
            
            # Record metrics
            training_history["episode_rewards"].append(metrics["episode_reward"])
            training_history["episode_lengths"].append(metrics["episode_length"])
            
            if "policy_loss" in metrics:
                training_history["policy_losses"].append(metrics["policy_loss"])
                training_history["value_losses"].append(metrics["value_loss"])
            
            # Evaluation
            if (episode + 1) % eval_freq == 0:
                eval_reward = self.evaluate(num_episodes=5)
                training_history["eval_rewards"].append(eval_reward)
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Train Reward: {metrics['episode_reward']:.2f}")
                print(f"  Eval Reward: {eval_reward:.2f}")
                if "policy_loss" in metrics:
                    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                    print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print()
        
        return training_history
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> float:
        """Evaluate the current policy."""
        total_reward = 0
        
        for _ in range(num_episodes):
            # Handle both old and new gym API
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
                
            episode_reward = 0
            
            for step in range(self.max_episode_steps):
                if render:
                    self.env.render()
                
                # Use deterministic action selection for evaluation
                action, _ = self.agent.select_action(state, group_id=0)
                
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    state, reward, done, _ = step_result
                else:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                    
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        # Clear storage after evaluation (no training)
        self.agent.reset_storage()
        
        return total_reward / num_episodes