"""
Fairness metrics and analysis tools for healthcare GRPO implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class FairnessReport:
    """Comprehensive fairness analysis report."""
    demographic_parity: float
    equalized_odds: float
    individual_fairness: float
    treatment_rate_disparity: float
    wait_time_disparity: float
    statistical_significance: Dict[str, float]
    overall_fairness_score: float


class HealthcareFairnessAnalyzer:
    """
    Analyzer for healthcare-specific fairness metrics in GRPO.
    """
    
    def __init__(self):
        self.demographic_groups = ['PEDIATRIC', 'ADULT', 'ELDERLY', 'CRITICAL']
        self.fairness_thresholds = {
            'demographic_parity': 0.8,  # Minimum acceptable parity
            'wait_time_disparity': 2.0,  # Maximum acceptable ratio
            'treatment_rate_disparity': 1.5,  # Maximum acceptable ratio
            'statistical_significance': 0.05  # p-value threshold
        }
    
    def calculate_demographic_parity(self, treatment_rates: Dict[str, float]) -> float:
        """
        Calculate demographic parity score.
        Perfect parity = 1.0, higher is better.
        """
        if not treatment_rates or len(treatment_rates) < 2:
            return 1.0
        
        rates = list(treatment_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)
        
        if max_rate == 0:
            return 1.0
        
        # Parity score: how close the worst group is to the best group
        parity_score = min_rate / max_rate
        return parity_score
    
    def calculate_wait_time_disparity(self, wait_times: Dict[str, List[float]]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate wait time disparity across demographic groups.
        Returns disparity ratio and statistical significance tests.
        """
        if not wait_times:
            return 1.0, {}
        
        # Calculate average wait times per group
        avg_wait_times = {}
        for group, times in wait_times.items():
            if times:
                avg_wait_times[group] = np.mean(times)
        
        if len(avg_wait_times) < 2:
            return 1.0, {}
        
        # Calculate disparity ratio
        min_wait = min(avg_wait_times.values())
        max_wait = max(avg_wait_times.values())
        disparity_ratio = max_wait / max(min_wait, 1e-6)
        
        # Statistical significance testing
        significance_tests = {}
        groups = list(wait_times.keys())
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                if wait_times[group1] and wait_times[group2]:
                    # Perform t-test
                    _, p_value = stats.ttest_ind(wait_times[group1], wait_times[group2])
                    significance_tests[f"{group1}_vs_{group2}"] = p_value
        
        return disparity_ratio, significance_tests
    
    def calculate_treatment_rate_disparity(self, 
                                         patients_treated: Dict[str, int], 
                                         patients_arrived: Dict[str, int]) -> float:
        """Calculate treatment rate disparity across demographic groups."""
        if not patients_treated or not patients_arrived:
            return 1.0
        
        treatment_rates = {}
        for group in patients_treated:
            if patients_arrived.get(group, 0) > 0:
                treatment_rates[group] = patients_treated[group] / patients_arrived[group]
        
        if len(treatment_rates) < 2:
            return 1.0
        
        min_rate = min(treatment_rates.values())
        max_rate = max(treatment_rates.values())
        
        if min_rate == 0:
            return float('inf')
        
        return max_rate / min_rate
    
    def calculate_individual_fairness(self, 
                                    patient_outcomes: List[Dict],
                                    similarity_threshold: float = 0.1) -> float:
        """
        Calculate individual fairness metric.
        Similar patients should receive similar treatment.
        """
        if len(patient_outcomes) < 2:
            return 1.0
        
        fairness_violations = 0
        total_comparisons = 0
        
        for i in range(len(patient_outcomes)):
            for j in range(i + 1, len(patient_outcomes)):
                patient1 = patient_outcomes[i]
                patient2 = patient_outcomes[j]
                
                # Calculate patient similarity (simplified)
                age_diff = abs(patient1.get('age', 0) - patient2.get('age', 0)) / 100
                severity_diff = abs(patient1.get('severity', 0) - patient2.get('severity', 0)) / 2
                
                similarity = 1.0 - (age_diff + severity_diff) / 2
                
                if similarity > similarity_threshold:
                    # Patients are similar, check if treatment was similar
                    wait_diff = abs(patient1.get('wait_time', 0) - patient2.get('wait_time', 0))
                    treatment_diff = abs(patient1.get('treatment_quality', 0) - patient2.get('treatment_quality', 0))
                    
                    outcome_similarity = 1.0 - (wait_diff / 24 + treatment_diff) / 2
                    
                    if outcome_similarity < similarity_threshold:
                        fairness_violations += 1
                    
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 1.0
        
        individual_fairness = 1.0 - (fairness_violations / total_comparisons)
        return max(0.0, individual_fairness)
    
    def generate_fairness_report(self, 
                               training_results: List[Dict],
                               detailed: bool = True) -> FairnessReport:
        """Generate comprehensive fairness report from training results."""
        
        # Aggregate data across all agents
        all_wait_times = {group: [] for group in self.demographic_groups}
        all_treatment_rates = {group: 0 for group in self.demographic_groups}
        all_patients_arrived = {group: 0 for group in self.demographic_groups}
        
        for result in training_results:
            # Extract demographic metrics
            for metric in result.get('demographic_metrics', []):
                wait_times = metric.get('avg_wait_times', {})
                for group, wait_time in wait_times.items():
                    if wait_time > 0:
                        all_wait_times[group].append(wait_time)
        
        # Calculate fairness metrics
        demographic_parity = self.calculate_demographic_parity(
            {group: len(times) for group, times in all_wait_times.items()}
        )
        
        wait_disparity, significance_tests = self.calculate_wait_time_disparity(all_wait_times)
        
        treatment_disparity = self.calculate_treatment_rate_disparity(
            all_treatment_rates, all_patients_arrived
        )
        
        # Individual fairness (simplified for aggregated data)
        individual_fairness = 0.8  # Placeholder - would need patient-level data
        
        # Calculate overall fairness score
        fairness_components = [
            demographic_parity,
            min(2.0 / wait_disparity, 1.0),  # Normalize wait disparity
            min(1.5 / treatment_disparity, 1.0),  # Normalize treatment disparity
            individual_fairness
        ]
        overall_fairness = np.mean(fairness_components)
        
        report = FairnessReport(
            demographic_parity=demographic_parity,
            equalized_odds=0.0,  # Would need additional data
            individual_fairness=individual_fairness,
            treatment_rate_disparity=treatment_disparity,
            wait_time_disparity=wait_disparity,
            statistical_significance=significance_tests,
            overall_fairness_score=overall_fairness
        )
        
        if detailed:
            self._print_detailed_report(report, all_wait_times)
        
        return report
    
    def _print_detailed_report(self, report: FairnessReport, wait_times: Dict[str, List[float]]):
        """Print detailed fairness analysis report."""
        print("\n" + "=" * 60)
        print("HEALTHCARE FAIRNESS ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\n📊 OVERALL FAIRNESS SCORE: {report.overall_fairness_score:.3f}")
        
        # Fairness interpretation
        if report.overall_fairness_score >= 0.8:
            fairness_level = "EXCELLENT ✅"
        elif report.overall_fairness_score >= 0.6:
            fairness_level = "GOOD ⚠️"
        elif report.overall_fairness_score >= 0.4:
            fairness_level = "NEEDS IMPROVEMENT 🔸"
        else:
            fairness_level = "POOR ❌"
        
        print(f"Fairness Level: {fairness_level}")
        
        print("\n📈 DETAILED METRICS:")
        print("-" * 40)
        print(f"Demographic Parity: {report.demographic_parity:.3f}")
        print(f"Wait Time Disparity Ratio: {report.wait_time_disparity:.2f}")
        print(f"Treatment Rate Disparity: {report.treatment_rate_disparity:.2f}")
        print(f"Individual Fairness: {report.individual_fairness:.3f}")
        
        print("\n⏱️ WAIT TIME ANALYSIS:")
        print("-" * 40)
        for group, times in wait_times.items():
            if times:
                mean_wait = np.mean(times)
                std_wait = np.std(times)
                print(f"{group}: {mean_wait:.1f} ± {std_wait:.1f} hours ({len(times)} patients)")
        
        print("\n🧪 STATISTICAL SIGNIFICANCE:")
        print("-" * 40)
        for comparison, p_value in report.statistical_significance.items():
            significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
            print(f"{comparison}: p={p_value:.4f} ({significance})")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        if report.demographic_parity < 0.8:
            print("• Increase group robustness weight to improve demographic parity")
        
        if report.wait_time_disparity > 2.0:
            print("• Implement priority queuing for underserved demographic groups")
        
        if report.treatment_rate_disparity > 1.5:
            print("• Review resource allocation policies for fairness")
        
        if report.overall_fairness_score < 0.6:
            print("• Consider additional fairness constraints in the reward function")
            print("• Implement regular fairness audits during training")
        
        print("\n" + "=" * 60)
    
    def create_fairness_visualization(self, 
                                    training_results: List[Dict],
                                    save_path: Optional[str] = None):
        """Create comprehensive fairness visualization."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Healthcare GRPO Fairness Analysis', fontsize=16, fontweight='bold')
        
        # 1. Fairness Score by Agent
        ax1 = axes[0, 0]
        agent_fairness = [r.get('final_avg_fairness', 0) for r in training_results]
        scenarios = [r.get('scenario', f'Agent {r.get("agent_id", i)}') for i, r in enumerate(training_results)]
        
        bars = ax1.bar(scenarios, agent_fairness, color='skyblue', alpha=0.7)
        ax1.set_title('Fairness Score by Hospital Scenario')
        ax1.set_ylabel('Fairness Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, agent_fairness):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Reward vs Fairness Trade-off
        ax2 = axes[0, 1]
        rewards = [r.get('final_avg_reward', 0) for r in training_results]
        fairness = [r.get('final_avg_fairness', 0) for r in training_results]
        robustness_weights = [r.get('config', {}).get('group_robustness_weight', 0) for r in training_results]
        
        scatter = ax2.scatter(rewards, fairness, c=robustness_weights, cmap='viridis', 
                            s=100, alpha=0.7)
        ax2.set_xlabel('Average Reward')
        ax2.set_ylabel('Fairness Score')
        ax2.set_title('Reward vs Fairness Trade-off')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Robustness Weight')
        
        # 3. Robustness Weight Impact
        ax3 = axes[1, 0]
        weight_impact = {}
        for result in training_results:
            weight = result.get('config', {}).get('group_robustness_weight', 0)
            if weight not in weight_impact:
                weight_impact[weight] = {'rewards': [], 'fairness': []}
            weight_impact[weight]['rewards'].append(result.get('final_avg_reward', 0))
            weight_impact[weight]['fairness'].append(result.get('final_avg_fairness', 0))
        
        weights = sorted(weight_impact.keys())
        avg_rewards = [np.mean(weight_impact[w]['rewards']) for w in weights]
        avg_fairness = [np.mean(weight_impact[w]['fairness']) for w in weights]
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(weights, avg_rewards, 'b-o', label='Avg Reward')
        line2 = ax3_twin.plot(weights, avg_fairness, 'r-s', label='Avg Fairness')
        
        ax3.set_xlabel('Robustness Weight')
        ax3.set_ylabel('Average Reward', color='b')
        ax3_twin.set_ylabel('Average Fairness', color='r')
        ax3.set_title('Impact of Robustness Weight')
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        # 4. Performance Distribution
        ax4 = axes[1, 1]
        all_rewards = []
        all_fairness = []
        scenario_labels = []
        
        for result in training_results:
            scenario = result.get('scenario', f'Agent {result.get("agent_id", 0)}')
            all_rewards.extend([result.get('final_avg_reward', 0)])
            all_fairness.extend([result.get('final_avg_fairness', 0)])
            scenario_labels.extend([scenario])
        
        # Create performance distribution plot
        performance_data = pd.DataFrame({
            'Reward': all_rewards,
            'Fairness': all_fairness,
            'Scenario': scenario_labels
        })
        
        sns.boxplot(data=performance_data.melt(id_vars=['Scenario'], 
                                              value_vars=['Reward', 'Fairness']),
                   x='Scenario', y='value', hue='variable', ax=ax4)
        ax4.set_title('Performance Distribution by Scenario')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fairness visualization saved to: {save_path}")
        
        plt.show()


def analyze_training_results(results: List[Dict]) -> FairnessReport:
    """Analyze training results and generate fairness report."""
    analyzer = HealthcareFairnessAnalyzer()
    
    # Generate comprehensive fairness report
    report = analyzer.generate_fairness_report(results, detailed=True)
    
    # Create visualizations
    analyzer.create_fairness_visualization(results)
    
    return report