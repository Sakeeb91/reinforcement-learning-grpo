import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(state), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class GRPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        group_robustness_weight: float = 0.1,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.group_robustness_weight = group_robustness_weight
        self.device = torch.device(device)
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # Storage for trajectories
        self.reset_storage()
    
    def reset_storage(self):
        """Reset trajectory storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.groups = []  # Group identifiers for robustness
    
    def select_action(self, state: np.ndarray, group_id: int = 0) -> Tuple[int, float]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
            value = self.value(state_tensor)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Store trajectory data
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        self.groups.append(group_id)
        
        return action.item(), log_prob.item()
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        G = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return torch.FloatTensor(returns).to(self.device)
    
    def compute_group_robust_loss(self, policy_loss: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Compute group-robust policy loss."""
        unique_groups = torch.unique(groups)
        group_losses = []
        
        for group in unique_groups:
            group_mask = (groups == group)
            if group_mask.sum() > 0:
                group_loss = policy_loss[group_mask].mean()
                group_losses.append(group_loss)
        
        if len(group_losses) > 1:
            # Robust loss: maximize performance on worst-performing group
            group_losses = torch.stack(group_losses)
            robust_loss = group_losses.max()
            return robust_loss
        else:
            return policy_loss.mean()
    
    def update(self) -> Dict[str, float]:
        """Update policy and value networks using GRPO."""
        if len(self.states) == 0:
            return {}
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        groups = torch.LongTensor(self.groups).to(self.device)
        
        # Compute returns and advantages
        returns = self.compute_returns()
        values = torch.FloatTensor(self.values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy update
        action_probs = self.policy(states)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2)
        
        # Group robust loss
        robust_policy_loss = self.compute_group_robust_loss(policy_loss, groups)
        standard_policy_loss = policy_loss.mean()
        
        # Combined loss
        total_policy_loss = (
            (1 - self.group_robustness_weight) * standard_policy_loss +
            self.group_robustness_weight * robust_policy_loss
        )
        
        # Value loss
        current_values = self.value(states).squeeze()
        value_loss = F.mse_loss(current_values, returns)
        
        # Update networks
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Reset storage
        self.reset_storage()
        
        return {
            "policy_loss": total_policy_loss.item(),
            "value_loss": value_loss.item(),
            "standard_policy_loss": standard_policy_loss.item(),
            "robust_policy_loss": robust_policy_loss.item()
        }