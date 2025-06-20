#!/usr/bin/env python3
"""
Create sample visualizations for GRPO Healthcare project without requiring training.
This demonstrates the visualization capabilities without needing CUDA or long training times.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_training_results():
    """Create realistic sample training results for visualization."""
    
    # Simulate 4 hospital scenarios with different configurations
    scenarios = ["urban_hospital", "rural_hospital", "pediatric_hospital", "emergency_surge"]
    robustness_weights = [0.3, 0.4, 0.2, 0.5]
    
    sample_results = []
    
    for i, (scenario, robustness_weight) in enumerate(zip(scenarios, robustness_weights)):
        # Simulate realistic performance metrics
        # Higher robustness weight generally improves fairness but may reduce reward
        base_reward = np.random.normal(150, 20)
        fairness_boost = robustness_weight * 0.3
        reward_penalty = robustness_weight * 15
        
        final_reward = base_reward - reward_penalty + np.random.normal(0, 5)
        final_fairness = 0.6 + fairness_boost + np.random.normal(0, 0.05)
        final_fairness = np.clip(final_fairness, 0, 1)
        
        # Generate training curves
        episodes = 600
        training_rewards = []
        fairness_scores = []
        
        for ep in range(episodes):
            # Simulate learning curve
            progress = ep / episodes
            reward = final_reward * (0.3 + 0.7 * progress) + np.random.normal(0, 10)
            fairness = final_fairness * (0.4 + 0.6 * progress) + np.random.normal(0, 0.03)
            fairness = np.clip(fairness, 0, 1)
            
            training_rewards.append(reward)
            fairness_scores.append(fairness)
        
        result = {
            'agent_id': i,
            'scenario': scenario,
            'config': {'group_robustness_weight': robustness_weight},
            'training_rewards': training_rewards,
            'fairness_scores': fairness_scores,
            'final_avg_reward': final_reward,
            'final_avg_fairness': final_fairness,
            'eval_reward_mean': final_reward + np.random.normal(0, 3),
            'eval_reward_std': np.random.uniform(5, 15),
            'eval_fairness_mean': final_fairness + np.random.normal(0, 0.02),
            'eval_fairness_std': np.random.uniform(0.01, 0.05)
        }
        
        sample_results.append(result)
    
    return sample_results


def create_healthcare_dashboard(results):
    """Create comprehensive healthcare GRPO visualization dashboard."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('🏥 Healthcare GRPO Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Training Progress (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    for i, result in enumerate(results):
        episodes = range(len(result['training_rewards']))
        smoothed_rewards = pd.Series(result['training_rewards']).rolling(window=20).mean()
        
        ax1.plot(episodes, smoothed_rewards, label=f"{result['scenario'].replace('_', ' ').title()}", 
                linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('🚀 Training Progress Across Hospital Scenarios')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Final Performance Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    scenarios = [r['scenario'].replace('_', ' ').title() for r in results]
    final_rewards = [r['final_avg_reward'] for r in results]
    
    bars = ax2.bar(range(len(scenarios)), final_rewards, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.set_ylabel('Final Reward')
    ax2.set_title('🏆 Final Performance')
    
    # Add value labels
    for bar, value in zip(bars, final_rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Fairness vs Reward Trade-off
    ax3 = fig.add_subplot(gs[1, 0])
    
    rewards = [r['final_avg_reward'] for r in results]
    fairness = [r['final_avg_fairness'] for r in results]
    weights = [r['config']['group_robustness_weight'] for r in results]
    
    scatter = ax3.scatter(rewards, fairness, c=weights, s=200, cmap='viridis', 
                         alpha=0.8, edgecolors='black', linewidth=1)
    
    for i, scenario in enumerate(scenarios):
        ax3.annotate(scenario.split()[0], (rewards[i], fairness[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Average Reward')
    ax3.set_ylabel('Fairness Score')
    ax3.set_title('⚖️ Reward vs Fairness Trade-off')
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Robustness Weight')
    
    # 4. Robustness Weight Impact
    ax4 = fig.add_subplot(gs[1, 1])
    
    weights = [r['config']['group_robustness_weight'] for r in results]
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(weights, rewards, 'bo-', linewidth=2, markersize=8, label='Reward')
    line2 = ax4_twin.plot(weights, fairness, 'rs-', linewidth=2, markersize=8, label='Fairness')
    
    ax4.set_xlabel('Group Robustness Weight')
    ax4.set_ylabel('Average Reward', color='blue')
    ax4_twin.set_ylabel('Fairness Score', color='red')
    ax4.set_title('📊 Robustness Weight Impact')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # 5. Fairness Score Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    
    fairness_data = []
    scenario_labels = []
    
    for result in results:
        # Simulate fairness distribution over training
        fair_scores = result['fairness_scores'][-100:]  # Last 100 episodes
        fairness_data.extend(fair_scores)
        scenario_labels.extend([result['scenario'].replace('_', ' ').title()] * len(fair_scores))
    
    df = pd.DataFrame({'Fairness': fairness_data, 'Scenario': scenario_labels})
    
    ax5.violinplot([df[df['Scenario'] == scenario]['Fairness'].values 
                   for scenario in scenarios], positions=range(len(scenarios)))
    ax5.set_xticks(range(len(scenarios)))
    ax5.set_xticklabels(scenarios, rotation=45, ha='right')
    ax5.set_ylabel('Fairness Score')
    ax5.set_title('🎻 Fairness Distribution')
    
    # 6. Demographic Wait Time Analysis (simulated)
    ax6 = fig.add_subplot(gs[2, :])
    
    demographics = ['Pediatric', 'Adult', 'Elderly', 'Critical']
    hospital_types = scenarios
    
    # Simulate wait time data showing fairness improvement
    wait_times = np.random.uniform(2, 8, (len(demographics), len(hospital_types)))
    
    # Add some realistic patterns
    wait_times[0, :] *= 0.8  # Pediatric slightly lower
    wait_times[2, :] *= 1.1  # Elderly slightly higher
    wait_times[3, :] *= 0.7  # Critical much lower (priority)
    
    # Apply robustness weight effect (more fair distribution)
    for i, weight in enumerate(weights):
        fairness_factor = 1 - (weight * 0.3)  # Higher weight = more equal wait times
        wait_times[:, i] = wait_times[:, i] * fairness_factor + np.mean(wait_times[:, i]) * weight * 0.3
    
    # Create heatmap
    im = ax6.imshow(wait_times, cmap='RdYlBu_r', aspect='auto')
    
    ax6.set_xticks(range(len(hospital_types)))
    ax6.set_xticklabels(hospital_types)
    ax6.set_yticks(range(len(demographics)))
    ax6.set_yticklabels(demographics)
    ax6.set_title('⏱️ Average Wait Times by Demographic (Hours)')
    
    # Add text annotations
    for i in range(len(demographics)):
        for j in range(len(hospital_types)):
            text = ax6.text(j, i, f'{wait_times[i, j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Wait Time (Hours)')
    
    plt.tight_layout()
    return fig


def create_fairness_metrics_summary(results):
    """Create a summary table of fairness metrics."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create summary data
    summary_data = []
    for result in results:
        scenario = result['scenario'].replace('_', ' ').title()
        robustness_weight = result['config']['group_robustness_weight']
        final_reward = result['final_avg_reward']
        final_fairness = result['final_avg_fairness']
        
        # Simulate additional metrics
        demographic_parity = final_fairness * 0.9 + np.random.normal(0, 0.02)
        wait_time_disparity = 2.5 - (final_fairness * 1.2) + np.random.normal(0, 0.1)
        
        summary_data.append([
            scenario,
            f"{robustness_weight:.1f}",
            f"{final_reward:.0f}",
            f"{final_fairness:.3f}",
            f"{demographic_parity:.3f}",
            f"{wait_time_disparity:.2f}"
        ])
    
    columns = ['Hospital Scenario', 'Robustness\nWeight', 'Final\nReward', 
               'Fairness\nScore', 'Demographic\nParity', 'Wait Time\nDisparity']
    
    # Create table
    table = ax.table(cellText=summary_data, colLabels=columns, 
                    cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code performance
    for i in range(1, len(summary_data) + 1):
        fairness_score = float(summary_data[i-1][3])
        if fairness_score >= 0.8:
            color = '#96CEB4'  # Green for good fairness
        elif fairness_score >= 0.6:
            color = '#FFEAA7'  # Yellow for moderate
        else:
            color = '#FFB3BA'  # Red for poor
        
        table[(i, 3)].set_facecolor(color)
    
    ax.axis('off')
    ax.set_title('📊 Healthcare GRPO Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    return fig


def main():
    """Generate all sample visualizations."""
    
    print("🎨 Creating sample GRPO Healthcare visualizations...")
    print("   (No CUDA required - runs on CPU!)")
    
    # Create output directory
    output_dir = "demo/sample_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    results = create_sample_training_results()
    
    print(f"\n📊 Generated sample results for {len(results)} hospital scenarios:")
    for result in results:
        print(f"   • {result['scenario'].replace('_', ' ').title()}: "
              f"Reward={result['final_avg_reward']:.0f}, "
              f"Fairness={result['final_avg_fairness']:.3f}")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # 1. Main dashboard
    print("   • Healthcare Dashboard...")
    fig1 = create_healthcare_dashboard(results)
    fig1.savefig(f"{output_dir}/healthcare_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Performance summary
    print("   • Performance Summary Table...")
    fig2 = create_fairness_metrics_summary(results)
    fig2.savefig(f"{output_dir}/performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Training curves detail
    print("   • Detailed Training Curves...")
    fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig3.suptitle('🏥 Individual Hospital Scenario Analysis', fontsize=16, fontweight='bold')
    
    for i, (result, ax) in enumerate(zip(results, axes.flat)):
        episodes = range(len(result['training_rewards']))
        
        # Plot both reward and fairness
        ax_twin = ax.twinx()
        
        line1 = ax.plot(episodes, pd.Series(result['training_rewards']).rolling(20).mean(), 
                       'b-', linewidth=2, label='Reward')
        line2 = ax_twin.plot(episodes, pd.Series(result['fairness_scores']).rolling(20).mean(), 
                           'r-', linewidth=2, label='Fairness')
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward', color='blue')
        ax_twin.set_ylabel('Fairness', color='red')
        ax.set_title(f"{result['scenario'].replace('_', ' ').title()}\n"
                    f"(λ={result['config']['group_robustness_weight']})")
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower right')
    
    plt.tight_layout()
    fig3.savefig(f"{output_dir}/individual_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"\n✅ Visualizations saved to: {output_dir}/")
    print("   📁 Files created:")
    print("      • healthcare_dashboard.png - Main analysis dashboard")
    print("      • performance_summary.png - Fairness metrics table") 
    print("      • individual_training_curves.png - Detailed training analysis")
    
    print(f"\n🎯 Key Insights from Sample Data:")
    print("   • Higher robustness weights improve fairness but may reduce raw performance")
    print("   • Emergency surge scenario benefits most from fairness optimization")
    print("   • Pediatric hospital maintains good performance with lower robustness weight")
    print("   • All scenarios achieve acceptable fairness levels (>0.6)")
    
    print(f"\n💡 To run actual training (requires dependencies):")
    print("   pip install -r requirements.txt")
    print("   python examples/hospital_scheduling.py")
    
    return output_dir


if __name__ == "__main__":
    output_dir = main()