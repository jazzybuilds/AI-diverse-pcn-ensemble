"""
Test clean data performance comparison across diversity mechanisms
Compare baseline performance without any corruption
"""
import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pypc import utils, datasets, optim
from pypc.models import PCModel, ParallelPCModel

print("="*70)
print("TESTING CLEAN DATA PERFORMANCE")
print("="*70)

# Fixed seed for reproducibility
base_seed = 42
utils.seed(base_seed)

# Dataset
print("\nLoading dataset...")
train_dataset = datasets.MNIST(train=True, normalize=False, size=2000)
test_dataset = datasets.MNIST(train=False, normalize=False, size=1000)

# Experiment parameters
batch_size = 640
test_size = 1000
n_epochs = 5
n_train_iters = 100
nodes = [784, 300, 100, 10]

# Train all diversity types
configs = {
    'Single PCN': None,
    'Parallel (init only)': 'init',
    'Parallel (dynamics only)': 'dynamics',
    'Parallel (architecture only)': 'architecture',
    'Parallel (MIXED)': 'mixed'
}

models = {}
print("\nTraining models with controlled randomness...")

for name, diversity_type in configs.items():
    print(f"\n{name}...")
    
    # Reset seed before each model
    utils.seed(base_seed)
    train_loader = datasets.get_dataloader(train_dataset, batch_size)
    
    if diversity_type is None:
        model = PCModel(nodes=nodes, mu_dt=0.01, act_fn=utils.Tanh(), use_bias=True)
        opt = optim.get_optim(model.params, "Adam", 1e-3, grad_clip=50)
    else:
        model = ParallelPCModel(
            n_models=10, nodes=nodes, mu_dt=0.01, act_fn=utils.Tanh(),
            use_bias=True, diversity_type=diversity_type,
            mu_dt_range=(0.005, 0.03)
        )
        opt = optim.get_optim(model.params, "Adam", 1e-3, grad_clip=50)
    
    with torch.no_grad():
        for epoch in range(1, n_epochs + 1):
            for batch_id, (img, lbl) in enumerate(train_loader):
                if isinstance(model, ParallelPCModel):
                    model.train_batch_supervised(img, lbl, n_train_iters)
                else:
                    model.train_batch_supervised(img, lbl, n_train_iters)
                opt.step(epoch, batch_id, len(train_loader), img.size(0))
    
    models[name] = model
    print(f"  ✓ Done")

# Test on clean data
print("\n" + "="*70)
print("Testing clean data performance...")
print("="*70)
utils.seed(base_seed)
test_loader = datasets.get_dataloader(test_dataset, test_size)

results = {}

for name, model in models.items():
    acc = 0
    
    for img, lbl in test_loader:
        if isinstance(model, ParallelPCModel):
            pred = model.test_batch_supervised(img, 'average')
        else:
            pred = model.test_batch_supervised(img)
        
        acc += datasets.accuracy(pred, lbl)
    
    accuracy = acc / len(test_loader)
    results[name] = accuracy
    print(f"  {name:30s} {accuracy:.2%}")

# Plot 1: Bar chart comparison
fig1 = plt.figure(figsize=(10, 6))
ax = fig1.add_subplot(111)
ax.set_title('Clean Data Performance Comparison', fontsize=14, weight='bold')
ax.set_ylabel('Accuracy', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

colors = ['red', 'blue', 'green', 'purple', 'orange']
model_names = list(results.keys())
accuracies = list(results.values())

x_pos = np.arange(len(model_names))
bars = ax.bar(x_pos, accuracies, color=colors[:len(model_names)], alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
ax.set_ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/clean_performance_bar.pdf', bbox_inches='tight')
print("\n✓ Panel 1 saved to results/clean_performance_bar.pdf")
plt.close(fig1)

# Plot 2: Improvement over baseline
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)
ax2.set_title('Ensemble Benefit on Clean Data', fontsize=14, weight='bold')
ax2.set_ylabel('Improvement over Single PCN (%)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

baseline_acc = results['Single PCN']
improvements = [(results[name] - baseline_acc) * 100 for name in model_names if name != 'Single PCN']
ensemble_names = [name for name in model_names if name != 'Single PCN']

x_pos2 = np.arange(len(ensemble_names))
bars2 = ax2.bar(x_pos2, improvements, color=colors[1:len(ensemble_names)+1], alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xticks(x_pos2)
ax2.set_xticklabels(ensemble_names, rotation=15, ha='right', fontsize=10)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Single PCN baseline')

# Add value labels on bars
for bar, imp in zip(bars2, improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:+.1f}%', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=10, fontweight='bold')

ax2.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('results/clean_performance_improvement.pdf', bbox_inches='tight')
print("✓ Panel 2 saved to results/clean_performance_improvement.pdf")
plt.close(fig2)

# Summary
print("\n" + "="*70)
print("SUMMARY - Clean Data Performance")
print("="*70)
print(f"\nBaseline (Single PCN): {baseline_acc:.2%}")
print("\nEnsemble improvements:")
for name in ensemble_names:
    improvement = (results[name] - baseline_acc) * 100
    print(f"  {name:30s} {improvement:+.2f}%")

print("\nKey Findings:")
best_ensemble = max(ensemble_names, key=lambda x: results[x])
print(f"  Best ensemble: {best_ensemble} ({results[best_ensemble]:.2%})")
print(f"  Improvement: {(results[best_ensemble] - baseline_acc)*100:+.2f}%")
print("\nConclusion:")
if max(results.values()) - baseline_acc > 0.01:  # >1% improvement
    print("  ✓ Ensembles provide meaningful clean-data performance boost")
    print("  → Diversity helps even without corruption")
else:
    print("  ≈ Ensembles match baseline on clean data")
    print("  → Primary benefit is robustness to corruption (not baseline accuracy)")
