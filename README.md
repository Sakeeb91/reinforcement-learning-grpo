# Group Robust Policy Optimization (GRPO)

This repository contains a PyTorch implementation of Group Robust Policy Optimization (GRPO), an extension of Proximal Policy Optimization (PPO) that aims to achieve robust performance across different groups or environments.

## Features

- **GRPO Algorithm**: Implements group-robust policy optimization with configurable robustness weights
- **PPO Integration**: Built on top of Proximal Policy Optimization with clipped objectives
- **Flexible Group Assignment**: Customizable group assignment functions for different scenarios
- **Training Framework**: Complete training and evaluation pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Train GRPO on CartPole environment:

```bash
python examples/train_cartpole.py
```

## Usage

### Basic Usage

```python
from grpo import GRPOAgent, GRPOTrainer

# Create agent
agent = GRPOAgent(
    state_dim=4,
    action_dim=2,
    lr=3e-4,
    group_robustness_weight=0.2
)

# Create trainer with custom group assignment
trainer = GRPOTrainer(
    env_name="CartPole-v1",
    agent=agent,
    group_assignment_fn=your_group_function
)

# Train
training_history = trainer.train(num_episodes=1000)
```

### Group Assignment

Define custom group assignment functions:

```python
def custom_group_assignment(state, episode):
    # Assign groups based on state features
    if state[0] < 0:
        return 0  # Group A
    else:
        return 1  # Group B
```

## Algorithm Overview

GRPO extends PPO by incorporating group robustness into the policy optimization objective:

- **Standard PPO Loss**: Optimizes average performance across all experiences
- **Group Robust Loss**: Focuses on worst-performing group to ensure fairness
- **Combined Objective**: Balances standard performance with group robustness

## Parameters

- `group_robustness_weight`: Controls the trade-off between average and worst-group performance (0.0 = standard PPO, 1.0 = fully robust)
- `eps_clip`: PPO clipping parameter
- `gamma`: Discount factor
- `lr`: Learning rate

## Examples

- `examples/train_cartpole.py`: CartPole training with position-based groups
- `examples/hospital_scheduling.py`: **Healthcare resource allocation with parallel training** - Demonstrates fair patient treatment across demographics

### Healthcare Application

The healthcare example showcases GRPO applied to hospital resource allocation, addressing real-world AI fairness challenges:

```bash
python examples/hospital_scheduling.py
```

**Key Features:**
- **Realistic Hospital Simulation**: Patient arrivals, resource constraints, treatment protocols
- **Demographic Fairness**: Ensures equitable treatment across age groups (pediatric, adult, elderly)
- **Parallel Training**: Multiple agents training on different hospital scenarios simultaneously
- **Comprehensive Analysis**: Fairness metrics, statistical significance testing, visualizations
- **Multiple Scenarios**: Urban hospital, rural hospital, pediatric hospital, emergency surge conditions

**Recruiter Appeal:**
- Demonstrates AI ethics and bias mitigation in critical healthcare applications
- Shows understanding of regulatory compliance and fairness requirements
- Combines technical ML expertise with healthcare domain knowledge
- Addresses real-world problems with measurable social impact

## License

MIT License