"""
Quick sweep script to run multiple robustness experiments
Run this to collect data for your report
"""

import subprocess
import sys

# Define experiment configurations
experiments = [
    # Baseline: Single vs Parallel with different ensemble sizes
    {
        'name': 'gaussian_noise_ensemble_size',
        'configs': [
            {'n_models': 1, 'corruption_type': 'gaussian', 'diversity_type': 'init'},
            {'n_models': 3, 'corruption_type': 'gaussian', 'diversity_type': 'init'},
            {'n_models': 5, 'corruption_type': 'gaussian', 'diversity_type': 'init'},
        ]
    },
    
    # Different diversity types
    {
        'name': 'diversity_types',
        'configs': [
            {'n_models': 3, 'corruption_type': 'gaussian', 'diversity_type': 'init'},
            {'n_models': 3, 'corruption_type': 'gaussian', 'diversity_type': 'dynamics'},
            {'n_models': 3, 'corruption_type': 'gaussian', 'diversity_type': 'architecture'},
        ]
    },
    
    # Different corruption types
    {
        'name': 'corruption_types',
        'configs': [
            {'n_models': 3, 'corruption_type': 'gaussian', 'diversity_type': 'init'},
            {'n_models': 3, 'corruption_type': 'salt_pepper', 'diversity_type': 'init'},
            {'n_models': 3, 'corruption_type': 'occlude', 'diversity_type': 'init'},
        ]
    },
]


def run_experiment(config):
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Running: {config}")
    print(f"{'='*70}\n")
    
    # Build command (you can modify PCN_robustness_experiment.py to accept CLI args,
    # or just run it with default config and collect results)
    cmd = [sys.executable, "PCN_robustness_experiment.py"]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Completed: {config}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {config}")
        print(f"Error: {e}")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  Parallel PCN Robustness Experiment Sweep                      ║
    ║  This will run multiple experiments for your report            ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nNOTE: This basic sweep runner executes the default config.")
    print("For custom sweeps, modify PCN_robustness_experiment.py config at bottom of file.\n")
    print("Recommended experiments to run manually:")
    print("\n1. Baseline comparison (vary n_models: 1, 3, 5)")
    print("   - Shows ensemble size effect")
    print("\n2. Diversity types (vary diversity_type: 'init', 'dynamics', 'architecture')")
    print("   - Shows best parallelism strategy")
    print("\n3. Corruption types (vary corruption_type: 'gaussian', 'salt_pepper', 'occlude')")
    print("   - Shows robustness across noise types")
    print("\n4. Ensemble methods (vary ensemble_method: 'average', 'vote', 'max')")
    print("   - Shows best combination strategy")
    
    print("\n" + "="*70)
    print("To run experiments, edit the config section at the bottom of")
    print("PCN_robustness_experiment.py and run it multiple times with different settings.")
    print("="*70 + "\n")
