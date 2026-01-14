"""
Analysis helper: Compare multiple experiment results
Use this after running several experiments to generate comparison plots
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_all_results(results_dir='results'):
    """Load all JSON result files."""
    results_dir = Path(results_dir)
    results = []
    
    for json_file in sorted(results_dir.glob('results_*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['filename'] = json_file.name
            results.append(data)
    
    return results


def plot_comparison(results_list, save_path='results/comparison.png'):
    """Plot all experiments on one figure."""
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, result in enumerate(results_list):
        corruption_type = result.get('corruption_type', 'unknown')
        
        # Plot each model configuration
        for j, (model_name, data) in enumerate(result['results'].items()):
            levels = [d['level'] for d in data]
            accs = [d['accuracy'] for d in data]
            
            label = f"{model_name} ({corruption_type})"
            plt.plot(levels, accs, marker='o', label=label, 
                    color=colors[i*2 + j % 2], linewidth=2, alpha=0.8)
    
    plt.xlabel('Corruption Level', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Robustness Comparison Across All Experiments', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {save_path}")
    plt.show()


def compute_area_under_curve(data):
    """Compute area under accuracy curve (robustness metric)."""
    levels = [d['level'] for d in data]
    accs = [d['accuracy'] for d in data]
    return np.trapz(accs, levels)


def generate_summary_table(results_list):
    """Print summary statistics table."""
    print("\n" + "="*80)
    print("SUMMARY TABLE: Area Under Curve (higher = more robust)")
    print("="*80)
    print(f"{'Model':<40} {'Corruption':<15} {'AUC':>10} {'Final Acc':>10}")
    print("-"*80)
    
    for result in results_list:
        corruption_type = result.get('corruption_type', 'unknown')
        
        for model_name, data in result['results'].items():
            auc = compute_area_under_curve(data)
            final_acc = data[-1]['accuracy']
            print(f"{model_name:<40} {corruption_type:<15} {auc:>10.4f} {final_acc:>10.2%}")
    
    print("="*80 + "\n")


def plot_variance_comparison(results_list, save_path='results/variance_comparison.png'):
    """Compare prediction variance across experiments."""
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, result in enumerate(results_list):
        corruption_type = result.get('corruption_type', 'unknown')
        
        for j, (model_name, data) in enumerate(result['results'].items()):
            if 'Parallel' not in model_name:
                continue  # Skip single models
                
            levels = [d['level'] for d in data]
            variances = [d['variance'] for d in data]
            
            if sum(variances) == 0:
                continue  # Skip if no variance data
                
            label = f"{model_name} ({corruption_type})"
            plt.plot(levels, variances, marker='s', label=label,
                    color=colors[i*2 + j % 2], linewidth=2, alpha=0.8)
    
    plt.xlabel('Corruption Level', fontsize=12)
    plt.ylabel('Prediction Variance', fontsize=12)
    plt.title('Ensemble Uncertainty Across Experiments', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved variance comparison: {save_path}")
    plt.show()


def main():
    print("Loading all experiment results...")
    results = load_all_results('results')
    
    if not results:
        print("No results found in results/ directory.")
        print("Run PCN_robustness_experiment.py first to generate data.")
        return
    
    print(f"Found {len(results)} experiment(s)\n")
    
    # Generate summary
    generate_summary_table(results)
    
    # Generate comparison plots
    plot_comparison(results, 'results/all_experiments_comparison.png')
    plot_variance_comparison(results, 'results/all_variance_comparison.png')
    
    print("\n✓ Analysis complete! Check results/ for comparison plots.")
    print("\nTips for your report:")
    print("  - Use AUC values to quantify robustness")
    print("  - Highlight which diversity type performs best")
    print("  - Show variance increases with corruption (uncertainty calibration)")


if __name__ == "__main__":
    main()
