# Experimental Methodology: GRPO Healthcare Resource Allocation

## Abstract

This document provides a comprehensive, reproducible methodology for conducting Group Robust Policy Optimization (GRPO) experiments in healthcare resource allocation. The methodology ensures statistical rigor, experimental validity, and reproducible results across diverse healthcare scenarios.

## 1. Experimental Design Framework

### 1.1 Research Questions

1. **Primary**: Does GRPO significantly improve demographic fairness compared to standard PPO?
2. **Secondary**: What is the optimal robustness weight (λ) for different hospital scenarios?
3. **Tertiary**: How does GRPO performance vary across hospital types and resource constraints?
4. **Quaternary**: What are the trade-offs between fairness and operational efficiency?

### 1.2 Hypotheses

- **H₁**: GRPO with λ > 0 will show statistically significant improvement in demographic fairness metrics compared to standard PPO (λ = 0)
- **H₂**: Optimal robustness weight will vary by hospital scenario based on resource constraints and patient demographics
- **H₃**: Fairness improvements will not come at the expense of operational efficiency (>95% retention)
- **H₄**: Statistical significance will be maintained across multiple random seeds and experimental conditions

### 1.3 Experimental Variables

#### Independent Variables
- **Hospital Scenario**: {Urban, Rural, Pediatric, Emergency Surge}
- **Robustness Weight (λ)**: {0.1, 0.2, 0.3, 0.4, 0.5}
- **Random Seed**: {42, 123, 456, 789, 321} (for reproducibility)

#### Dependent Variables
- **Primary**: Fairness Score (demographic parity)
- **Secondary**: Reward Score (operational efficiency)
- **Tertiary**: Wait Time Coefficient of Variation
- **Quaternary**: Resource Utilization Rates

#### Control Variables
- **Training Episodes**: 400 (fixed)
- **Evaluation Episodes**: 100 (fixed)
- **Network Architecture**: 3-layer [256, 128, 64] (fixed)
- **Learning Rate**: 3e-4 (fixed)
- **Batch Size**: 256 (fixed)

## 2. Hospital Scenario Specifications

### 2.1 Urban Hospital Configuration

```python
URBAN_HOSPITAL_CONFIG = {
    'beds': 120,
    'staff': 80,
    'equipment_units': 60,
    'patient_arrival_rate': 2.5,  # patients per time step
    'demographics': {
        'pediatric': 0.25,
        'adult': 0.50,
        'elderly': 0.25
    },
    'severity_distribution': {
        'routine': 0.60,
        'urgent': 0.25,
        'critical': 0.15
    },
    'treatment_duration': {
        'routine': {'mean': 4.0, 'std': 1.5},
        'urgent': {'mean': 8.0, 'std': 3.0},
        'critical': {'mean': 16.0, 'std': 6.0}
    }
}
```

### 2.2 Rural Hospital Configuration

```python
RURAL_HOSPITAL_CONFIG = {
    'beds': 40,
    'staff': 25,
    'equipment_units': 20,
    'patient_arrival_rate': 1.2,  # patients per time step
    'demographics': {
        'pediatric': 0.20,
        'adult': 0.40,
        'elderly': 0.40  # Higher elderly population
    },
    'severity_distribution': {
        'routine': 0.65,
        'urgent': 0.25,
        'critical': 0.10
    },
    'treatment_duration': {
        'routine': {'mean': 5.0, 'std': 2.0},  # Longer due to resource constraints
        'urgent': {'mean': 10.0, 'std': 4.0},
        'critical': {'mean': 20.0, 'std': 8.0}
    }
}
```

### 2.3 Pediatric Hospital Configuration

```python
PEDIATRIC_HOSPITAL_CONFIG = {
    'beds': 60,
    'staff': 45,
    'equipment_units': 35,
    'patient_arrival_rate': 1.8,  # patients per time step
    'demographics': {
        'pediatric': 0.75,  # Specialized pediatric focus
        'adult': 0.15,      # Family members
        'elderly': 0.10     # Grandparents
    },
    'severity_distribution': {
        'routine': 0.70,
        'urgent': 0.20,
        'critical': 0.10
    },
    'treatment_duration': {
        'routine': {'mean': 3.5, 'std': 1.0},  # Shorter pediatric treatments
        'urgent': {'mean': 6.0, 'std': 2.5},
        'critical': {'mean': 12.0, 'std': 5.0}
    }
}
```

### 2.4 Emergency Surge Configuration

```python
EMERGENCY_SURGE_CONFIG = {
    'beds': 100,  # Surge capacity
    'staff': 60,
    'equipment_units': 45,
    'patient_arrival_rate': 3.5,  # High surge arrivals
    'demographics': {
        'pediatric': 0.20,
        'adult': 0.45,
        'elderly': 0.35
    },
    'severity_distribution': {
        'routine': 0.40,
        'urgent': 0.30,
        'critical': 0.30  # Higher critical care ratio
    },
    'treatment_duration': {
        'routine': {'mean': 6.0, 'std': 2.0},  # Longer due to surge conditions
        'urgent': {'mean': 12.0, 'std': 4.0},
        'critical': {'mean': 24.0, 'std': 10.0}
    }
}
```

## 3. GRPO Algorithm Implementation

### 3.1 Core Algorithm

```python
def grpo_loss(policy_loss, value_loss, group_losses, robustness_weight):
    """
    Compute GRPO loss combining standard PPO with group robustness.
    
    Args:
        policy_loss: Standard PPO policy loss
        value_loss: Value function loss
        group_losses: Dict of losses by demographic group
        robustness_weight: λ parameter controlling fairness-efficiency trade-off
    
    Returns:
        Combined GRPO loss
    """
    # Standard PPO loss
    standard_loss = policy_loss + 0.5 * value_loss
    
    # Group robustness loss (minimax fairness)
    group_rewards = [loss['reward'] for loss in group_losses.values()]
    robust_loss = max(group_rewards) - min(group_rewards)
    
    # Combined objective
    total_loss = (1 - robustness_weight) * standard_loss + robustness_weight * robust_loss
    
    return total_loss
```

### 3.2 Group Assignment Function

```python
def assign_demographic_group(patient_state):
    """
    Assign patients to demographic groups based on age and condition.
    
    Args:
        patient_state: Dictionary containing patient information
    
    Returns:
        Group identifier: 'pediatric', 'adult', 'elderly', or 'critical'
    """
    age = patient_state['age']
    severity = patient_state['severity']
    
    # Critical care patients form separate group regardless of age
    if severity == 'critical':
        return 'critical'
    
    # Age-based grouping
    if age < 18:
        return 'pediatric'
    elif age < 65:
        return 'adult'
    else:
        return 'elderly'
```

### 3.3 Training Protocol

```python
def train_grpo_agent(scenario_config, robustness_weight, num_episodes=400):
    """
    Train GRPO agent with specified configuration.
    
    Args:
        scenario_config: Hospital scenario configuration
        robustness_weight: λ parameter for robustness-efficiency trade-off
        num_episodes: Number of training episodes
    
    Returns:
        Trained agent and training history
    """
    # Initialize environment and agent
    env = HospitalEnvironment(scenario_config)
    agent = GRPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        robustness_weight=robustness_weight
    )
    
    # Training loop with group-specific tracking
    training_history = []
    for episode in range(num_episodes):
        episode_data = run_episode(env, agent, train=True)
        training_history.append(episode_data)
        
        # Log progress every 50 episodes
        if episode % 50 == 0:
            log_training_progress(episode, episode_data)
    
    return agent, training_history
```

## 4. Evaluation Protocol

### 4.1 Evaluation Metrics

#### 4.1.1 Primary Fairness Metrics

```python
def compute_demographic_parity(group_outcomes):
    """
    Compute demographic parity score.
    
    Args:
        group_outcomes: Dict of treatment outcomes by demographic group
    
    Returns:
        Demographic parity score (0-1, higher is more fair)
    """
    treatment_rates = {}
    for group, outcomes in group_outcomes.items():
        treatment_rates[group] = sum(outcomes) / len(outcomes)
    
    # Calculate parity as 1 minus coefficient of variation
    rates = list(treatment_rates.values())
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    
    if mean_rate == 0:
        return 1.0
    
    cv = std_rate / mean_rate
    parity_score = max(0, 1 - cv)
    
    return parity_score
```

#### 4.1.2 Wait Time Equity

```python
def compute_wait_time_equity(wait_times_by_group):
    """
    Compute wait time equity across demographic groups.
    
    Args:
        wait_times_by_group: Dict of wait times by demographic group
    
    Returns:
        Wait time equity score (0-1, higher is more equitable)
    """
    mean_wait_times = {}
    for group, wait_times in wait_times_by_group.items():
        mean_wait_times[group] = np.mean(wait_times)
    
    # Calculate coefficient of variation
    means = list(mean_wait_times.values())
    overall_mean = np.mean(means)
    overall_std = np.std(means)
    
    if overall_mean == 0:
        return 1.0
    
    cv = overall_std / overall_mean
    equity_score = max(0, 1 - cv)
    
    return equity_score
```

#### 4.1.3 Resource Utilization Efficiency

```python
def compute_resource_utilization(resource_usage_history):
    """
    Compute resource utilization efficiency.
    
    Args:
        resource_usage_history: Time series of resource usage
    
    Returns:
        Dict of utilization rates by resource type
    """
    utilization_rates = {}
    
    for resource_type, usage_history in resource_usage_history.items():
        total_capacity = usage_history['capacity']
        used_capacity = usage_history['used']
        
        utilization_rate = np.mean(used_capacity) / total_capacity
        utilization_rates[resource_type] = utilization_rate
    
    return utilization_rates
```

### 4.2 Statistical Analysis Protocol

#### 4.2.1 Confidence Interval Calculation

```python
def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval.
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_bound, upper_bound
```

#### 4.2.2 Hypothesis Testing

```python
def test_fairness_improvement(baseline_scores, treatment_scores, alpha=0.05):
    """
    Test statistical significance of fairness improvement.
    
    Args:
        baseline_scores: Fairness scores for baseline (λ=0)
        treatment_scores: Fairness scores for treatment (λ>0)
        alpha: Significance level
    
    Returns:
        Dict with test results
    """
    from scipy import stats
    
    # Two-tailed t-test
    t_stat, p_value = stats.ttest_ind(treatment_scores, baseline_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(treatment_scores) - 1) * np.var(treatment_scores) + 
                         (len(baseline_scores) - 1) * np.var(baseline_scores)) / 
                        (len(treatment_scores) + len(baseline_scores) - 2))
    
    cohens_d = (np.mean(treatment_scores) - np.mean(baseline_scores)) / pooled_std
    
    # Interpretation
    is_significant = p_value < alpha
    effect_size_interpretation = interpret_effect_size(abs(cohens_d))
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'cohens_d': cohens_d,
        'effect_size': effect_size_interpretation,
        'confidence_interval': bootstrap_confidence_interval(treatment_scores - baseline_scores)
    }
```

## 5. Experimental Execution

### 5.1 Single Experiment Protocol

```python
def run_single_experiment(scenario, robustness_weight, seed):
    """
    Run a single GRPO experiment with specified parameters.
    
    Args:
        scenario: Hospital scenario name
        robustness_weight: Lambda parameter
        seed: Random seed for reproducibility
    
    Returns:
        Experiment results dictionary
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load scenario configuration
    config = get_scenario_config(scenario)
    
    # Initialize and train agent
    agent, training_history = train_grpo_agent(config, robustness_weight)
    
    # Evaluate trained agent
    eval_results = evaluate_agent(agent, config, num_episodes=100)
    
    # Compute metrics
    fairness_metrics = compute_fairness_metrics(eval_results)
    efficiency_metrics = compute_efficiency_metrics(eval_results)
    
    # Package results
    experiment_results = {
        'experiment_id': f'exp_{scenario}_w{robustness_weight}_seed{seed}',
        'scenario': scenario,
        'robustness_weight': robustness_weight,
        'seed': seed,
        'training_history': training_history,
        'evaluation_results': eval_results,
        'fairness_metrics': fairness_metrics,
        'efficiency_metrics': efficiency_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    return experiment_results
```

### 5.2 Comprehensive Experiment Suite

```python
def run_comprehensive_experiments():
    """
    Run complete experimental suite across all scenarios and parameters.
    
    Returns:
        Complete experimental results
    """
    scenarios = ['urban_hospital', 'rural_hospital', 'pediatric_hospital', 'emergency_surge']
    robustness_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = [42, 123, 456, 789, 321]
    
    all_results = []
    
    for scenario in scenarios:
        for weight in robustness_weights:
            for seed in seeds:
                print(f"Running experiment: {scenario}, λ={weight}, seed={seed}")
                
                try:
                    result = run_single_experiment(scenario, weight, seed)
                    all_results.append(result)
                    
                    # Save intermediate results
                    save_experiment_result(result)
                    
                except Exception as e:
                    print(f"Experiment failed: {e}")
                    continue
    
    # Save complete results
    save_comprehensive_results(all_results)
    
    return all_results
```

## 6. Data Collection and Storage

### 6.1 Data Structure

```python
EXPERIMENT_DATA_SCHEMA = {
    'experiment_id': 'string',
    'scenario': 'string',
    'robustness_weight': 'float',
    'seed': 'int',
    'training_data': {
        'episode_rewards': 'list[float]',
        'episode_fairness_scores': 'list[float]',
        'group_specific_rewards': 'dict[string, list[float]]',
        'resource_utilization': 'list[dict]',
        'convergence_metrics': 'dict'
    },
    'evaluation_data': {
        'mean_reward': 'float',
        'std_reward': 'float',
        'mean_fairness': 'float',
        'std_fairness': 'float',
        'demographic_outcomes': 'dict[string, dict]',
        'wait_time_distributions': 'dict[string, list[float]]',
        'resource_utilization_final': 'dict[string, float]'
    },
    'statistical_tests': {
        'fairness_improvement_test': 'dict',
        'efficiency_retention_test': 'dict',
        'confidence_intervals': 'dict'
    },
    'metadata': {
        'timestamp': 'string',
        'duration_seconds': 'float',
        'system_info': 'dict'
    }
}
```

### 6.2 Data Validation

```python
def validate_experiment_data(experiment_result):
    """
    Validate experiment data against schema and quality checks.
    
    Args:
        experiment_result: Experiment result dictionary
    
    Returns:
        Validation result with any issues identified
    """
    issues = []
    
    # Schema validation
    required_fields = ['experiment_id', 'scenario', 'robustness_weight', 'seed']
    for field in required_fields:
        if field not in experiment_result:
            issues.append(f"Missing required field: {field}")
    
    # Data quality checks
    if 'evaluation_data' in experiment_result:
        eval_data = experiment_result['evaluation_data']
        
        # Check for reasonable values
        if eval_data.get('mean_reward', 0) > 0:
            issues.append("Mean reward should be negative (cost minimization)")
        
        if not (0 <= eval_data.get('mean_fairness', -1) <= 10):
            issues.append("Fairness score outside expected range [0, 10]")
    
    # Statistical validation
    if 'statistical_tests' in experiment_result:
        tests = experiment_result['statistical_tests']
        for test_name, test_result in tests.items():
            if 'p_value' in test_result and not (0 <= test_result['p_value'] <= 1):
                issues.append(f"Invalid p-value in {test_name}: {test_result['p_value']}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }
```

## 7. Reproducibility Requirements

### 7.1 Environment Setup

```bash
# Create conda environment
conda create -n grpo-healthcare python=3.8
conda activate grpo-healthcare

# Install dependencies
pip install torch==1.10.0
pip install numpy==1.21.0
pip install scipy==1.7.0
pip install matplotlib==3.4.2
pip install pandas==1.3.0
pip install scikit-learn==1.0.0
pip install gym==0.18.3

# Install project in development mode
pip install -e .
```

### 7.2 Reproducibility Checklist

- [ ] **Random Seeds**: All experiments use fixed seeds for reproducibility
- [ ] **Environment Versions**: All package versions pinned in requirements.txt
- [ ] **Hardware Specifications**: Document CPU/GPU specifications used
- [ ] **Data Storage**: All experimental data saved with timestamps and metadata
- [ ] **Code Versioning**: Git commit hashes recorded for each experiment
- [ ] **Configuration Files**: All hyperparameters saved in version-controlled config files
- [ ] **Documentation**: Complete methodology documentation (this document)
- [ ] **Validation**: Independent reproduction by second researcher

### 7.3 Experiment Replication

```python
def replicate_experiment(experiment_id, target_results):
    """
    Replicate a specific experiment and validate results match.
    
    Args:
        experiment_id: ID of experiment to replicate
        target_results: Expected results from original experiment
    
    Returns:
        Replication validation results
    """
    # Parse experiment parameters from ID
    params = parse_experiment_id(experiment_id)
    
    # Run replication
    replicated_results = run_single_experiment(
        scenario=params['scenario'],
        robustness_weight=params['robustness_weight'],
        seed=params['seed']
    )
    
    # Compare results
    comparison = compare_experimental_results(target_results, replicated_results)
    
    return {
        'experiment_id': experiment_id,
        'original_results': target_results,
        'replicated_results': replicated_results,
        'comparison': comparison,
        'replication_successful': comparison['all_metrics_match']
    }
```

## 8. Quality Assurance

### 8.1 Experimental Validation

1. **Sanity Checks**: Verify baseline behavior matches expected patterns
2. **Convergence Validation**: Ensure training curves show appropriate convergence
3. **Statistical Power**: Verify adequate sample sizes for statistical significance
4. **Effect Size Validation**: Confirm practically significant effect sizes

### 8.2 Peer Review Protocol

1. **Code Review**: Independent review of algorithm implementation
2. **Methodology Review**: Validation of experimental design
3. **Statistical Review**: Verification of statistical analysis methods
4. **Results Review**: Independent interpretation of experimental outcomes

### 8.3 Documentation Standards

1. **Complete Methodology**: All procedures documented in detail
2. **Parameter Specification**: All hyperparameters and configurations recorded
3. **Assumption Documentation**: All modeling assumptions explicitly stated
4. **Limitation Discussion**: Known limitations and potential biases acknowledged

## 9. Ethical Considerations

### 9.1 Bias Mitigation

- **Demographic Representation**: Ensure balanced representation across groups
- **Algorithmic Fairness**: Explicit optimization for demographic equity
- **Validation Across Groups**: Separate validation for each demographic group
- **Continuous Monitoring**: Ongoing fairness assessment during deployment

### 9.2 Privacy Protection

- **Data Anonymization**: All patient data anonymized and aggregated
- **Minimal Data Collection**: Only collect data necessary for fairness assessment
- **Secure Storage**: Encrypted storage with access controls
- **Retention Policies**: Clear data retention and deletion policies

### 9.3 Safety Considerations

- **Simulation Validation**: Extensive testing in simulation before real-world deployment
- **Gradual Rollout**: Phased deployment with safety monitoring
- **Human Oversight**: Maintain human oversight and intervention capabilities
- **Fallback Procedures**: Clear procedures for reverting to human decision-making

## 10. Conclusion

This experimental methodology provides a comprehensive, reproducible framework for evaluating Group Robust Policy Optimization in healthcare settings. The methodology ensures:

- **Scientific Rigor**: Proper experimental design with statistical validation
- **Reproducibility**: Complete documentation and version control
- **Fairness Focus**: Explicit measurement and optimization of demographic equity
- **Practical Applicability**: Realistic healthcare scenarios and constraints

Following this methodology enables researchers to:
1. Replicate the GRPO healthcare experiments
2. Extend the work to new healthcare scenarios
3. Validate fairness improvements with statistical confidence
4. Ensure ethical and responsible AI development

The methodology serves as a foundation for advancing fairness-aware reinforcement learning in healthcare and other critical applications where demographic equity is essential.

---

*This methodology document ensures reproducible, rigorous, and ethical evaluation of AI fairness in healthcare applications.*