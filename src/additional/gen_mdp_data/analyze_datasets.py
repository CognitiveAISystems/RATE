"""
MDP Dataset Analysis Script

This script analyzes MDP datasets and prints detailed statistics about
observation spaces, action spaces, and reward distributions.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any
import gymnasium as gym

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.envs_datasets.mdp_dataset import analyze_mdp_dataset


def analyze_single_environment(env_name: str, data_dir: str) -> Dict[str, Any]:
    """Analyze a single MDP environment dataset."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {env_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return {}
    
    try:
        # Use the mdp_dataset analysis function
        metadata = analyze_mdp_dataset(data_dir, env_name)
        
        # Additional environment-specific analysis
        print(f"\nEnvironment Information:")
        env = gym.make(env_name)
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        if hasattr(env, 'spec') and env.spec:
            print(f"  Max episode steps: {env.spec.max_episode_steps}")
            print(f"  Reward threshold: {env.spec.reward_threshold}")
        
        env.close()
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error analyzing {env_name}: {e}")
        return {}


def create_summary_report(all_results: Dict[str, Dict], output_dir: str):
    """Create a comprehensive summary report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary table
    summary_data = []
    
    for env_name, metadata in all_results.items():
        if not metadata or 'statistics' not in metadata:
            continue
            
        stats = metadata['statistics']
        
        summary_data.append({
            'Environment': env_name,
            'Episodes': stats.get('n_episodes', 'N/A'),
            'Total Steps': stats.get('total_steps', 'N/A'),
            'Mean Return': f"{stats.get('mean_return', 0):.3f}",
            'Std Return': f"{stats.get('std_return', 0):.3f}",
            'Min Return': f"{stats.get('min_return', 0):.3f}",
            'Max Return': f"{stats.get('max_return', 0):.3f}",
            'Mean Length': f"{stats.get('mean_length', 0):.1f}",
            'State Dim': len(stats.get('obs_mean', [])),
            'Action Type': 'Discrete' if 'action_distribution' in stats else 'Continuous'
        })
    
    # Save summary as JSON
    with open(output_path / "analysis_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*120}")
    print("DATASET SUMMARY")
    print(f"{'='*120}")
    
    if summary_data:
        # Print header
        headers = list(summary_data[0].keys())
        print(" | ".join(f"{h:>12}" for h in headers))
        print("-" * (13 * len(headers) + len(headers) - 1))
        
        # Print data rows
        for row in summary_data:
            print(" | ".join(f"{str(row[h]):>12}" for h in headers))
    else:
        print("No valid datasets found for analysis.")
    
    print(f"\nDetailed analysis saved to: {output_path / 'analysis_summary.json'}")


def plot_dataset_statistics(all_results: Dict[str, Dict], output_dir: str):
    """Create visualization plots for dataset statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    env_names = []
    mean_returns = []
    std_returns = []
    episode_counts = []
    mean_lengths = []
    
    for env_name, metadata in all_results.items():
        if not metadata or 'statistics' not in metadata:
            continue
            
        stats = metadata['statistics']
        env_names.append(env_name.replace('-v', '\n-v'))  # Break long names
        mean_returns.append(stats.get('mean_return', 0))
        std_returns.append(stats.get('std_return', 0))
        episode_counts.append(stats.get('n_episodes', 0))
        mean_lengths.append(stats.get('mean_length', 0))
    
    if not env_names:
        print("No data available for plotting.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MDP Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Returns with Error Bars
    axes[0, 0].bar(env_names, mean_returns, yerr=std_returns, capsize=5, alpha=0.7)
    axes[0, 0].set_title('Mean Episode Returns', fontweight='bold')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Counts
    axes[0, 1].bar(env_names, episode_counts, alpha=0.7)
    axes[0, 1].set_title('Number of Episodes', fontweight='bold')
    axes[0, 1].set_ylabel('Episodes')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean Episode Lengths
    axes[1, 0].bar(env_names, mean_lengths, alpha=0.7)
    axes[1, 0].set_title('Mean Episode Length', fontweight='bold')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Return Distribution (Box plot style)
    return_ranges = []
    for env_name, metadata in all_results.items():
        if metadata and 'statistics' in metadata:
            stats = metadata['statistics']
            return_ranges.append([
                stats.get('min_return', 0),
                stats.get('mean_return', 0) - stats.get('std_return', 0),
                stats.get('mean_return', 0),
                stats.get('mean_return', 0) + stats.get('std_return', 0),
                stats.get('max_return', 0)
            ])
    
    if return_ranges:
        return_ranges = np.array(return_ranges)
        for i, env_name in enumerate(env_names):
            # Plot min-max range
            axes[1, 1].plot([i, i], [return_ranges[i, 0], return_ranges[i, 4]], 
                           'k-', alpha=0.3, linewidth=2)
            # Plot std range
            axes[1, 1].plot([i, i], [return_ranges[i, 1], return_ranges[i, 3]], 
                           'b-', alpha=0.7, linewidth=4)
            # Plot mean
            axes[1, 1].plot(i, return_ranges[i, 2], 'ro', markersize=6)
    
    axes[1, 1].set_title('Return Distribution', fontweight='bold')
    axes[1, 1].set_ylabel('Return')
    axes[1, 1].set_xticks(range(len(env_names)))
    axes[1, 1].set_xticklabels(env_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "dataset_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "dataset_analysis.pdf", bbox_inches='tight')
    print(f"Plots saved to: {output_path / 'dataset_analysis.png'}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze MDP datasets")
    parser.add_argument("--env", type=str, default="all",
                       choices=["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                               "Acrobot-v1", "Pendulum-v1", "all"],
                       help="Environment to analyze ('all' for all environments)")
    parser.add_argument("--data-base-dir", type=str, default="data/MDP",
                       help="Base directory containing MDP datasets")
    parser.add_argument("--output-dir", type=str, default="src/additional/gen_mdp_data/analysis",
                       help="Directory to save analysis results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # List of environments to analyze
    if args.env == "all":
        envs_to_analyze = ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", 
                          "Acrobot-v1", "Pendulum-v1"]
    else:
        envs_to_analyze = [args.env]
    
    # Analyze each environment
    all_results = {}
    
    for env_name in envs_to_analyze:
        data_dir = os.path.join(args.data_base_dir, env_name)
        result = analyze_single_environment(env_name, data_dir)
        if result:
            all_results[env_name] = result
    
    # Create summary report
    create_summary_report(all_results, args.output_dir)
    
    # Generate plots if requested
    if args.plot:
        try:
            plot_dataset_statistics(all_results, args.output_dir)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    print(f"\nAnalysis completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
