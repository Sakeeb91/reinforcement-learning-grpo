# Claude Memory for GRPO Healthcare Project

## Project Overview
This is a Group Robust Policy Optimization (GRPO) implementation focused on healthcare resource allocation. The project demonstrates AI fairness in critical healthcare applications by ensuring equitable treatment across different patient demographics.

## Current Status
- ✅ Basic GRPO algorithm implementation complete
- ✅ Core framework with policy/value networks and training pipeline
- ✅ Simple CartPole example working
- 🚧 Currently implementing: Healthcare resource allocation example

## Healthcare Implementation Goals
- Create realistic hospital environment simulation
- Implement fair resource allocation across patient demographics
- Add comprehensive fairness metrics and analysis
- Demonstrate ethical AI principles in healthcare

## Key Technical Details
- **Algorithm**: GRPO (extends PPO with group robustness)
- **Framework**: PyTorch
- **Environment**: Custom hospital simulation
- **Groups**: Patient demographics (pediatric, adult, elderly, critical care)
- **Fairness Objective**: Ensure no demographic group has significantly worse outcomes

## File Structure
```
grpo/                   # Core GRPO implementation
├── __init__.py
├── grpo_agent.py      # Main agent with group-robust loss
└── grpo_trainer.py    # Training and evaluation framework

examples/              # Application examples
├── train_cartpole.py  # Basic example (complete)
└── hospital_scheduling.py  # Healthcare example (in progress)

envs/                  # Custom environments
└── hospital_env.py    # Hospital simulation (in progress)
```

## Git Configuration
- Author: Sakeeb91 <rahman.sakeeb@gmail.com>
- This was set up to fix GitHub contributions not showing up
- Always commit with proper author information for portfolio visibility

## Recruiter Appeal Strategy
This project targets healthcare AI roles by demonstrating:
1. **Technical Depth**: Advanced RL with fairness constraints
2. **Real-World Impact**: Healthcare resource optimization
3. **Ethical AI**: Demographic fairness and bias mitigation
4. **Domain Knowledge**: Understanding of healthcare constraints
5. **Production Skills**: Complete implementation with analysis tools

## Next Steps
1. Implement hospital environment with realistic patient flows
2. Add comprehensive fairness metrics
3. Create interactive visualizations
4. Write detailed documentation and analysis