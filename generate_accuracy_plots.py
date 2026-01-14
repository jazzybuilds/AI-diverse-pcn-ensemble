"""
Generate comprehensive accuracy plots for PCN robustness experiments
Shows clean accuracy, corrupted accuracy, degradation, and improvements
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
print("GENERATING COMPREHENSIVE ACCURACY PLOTS")
print("="*70)

# Experiment parameters
base_seed = 42
utils.seed(base_seed)

train_dataset = datasets.MNIST(train=True, normalize=False, size=5000)
test_dataset = datasets.MNIST(train=False, normalize=False, size=2000)

batch_size = 640
test_size = 2000
n_epochs = 10
n_train_iters = 100
nodes = [784, 300, 100, 10]

# Define corruption types and levels to test
corruption_configs = {
    'Gaussian Noise': {
        'type': 'gaussian',
        'levels': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    },
    'Salt & Pepper': {
        'type': 'salt_pepper',
        'levels': [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2]
    },
    'Occlusion': {
        'type': 'occlude',
        'levels': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
}

# Train both models once
print("\n[1/3] Training models...")
utils.seed(base_seed)
train_loader = datasets.get_dataloader(train_dataset, batch_size)

print("  Training Single PCN...")
single_model = PCModel(nodes=nodes, mu_dt=0.01, act_fn=utils.Tanh(), use_bias=True)
single_opt = optim.get_optim(single_model.params, "Adam", 1e-3, grad_clip=50)

with torch.no_grad():
    for epoch in range(1, n_epochs + 1):
        for batch_id, (img, lbl) in enumerate(train_loader):
            single_model.train_batch_supervised(img, lbl, n_train_iters)
            single_opt.step(epoch, batch_id, len(train_loader), img.size(0))
print("  ✓ Single PCN trained")

utils.seed(base_seed)
train_loader = datasets.get_dataloader(train_dataset, batch_size)

print("  Training Parallel PCN (3 models, architecture diversity)...")
parallel_model = ParallelPCModel(
    n_models=3, nodes=nodes, mu_dt=0.01, act_fn=utils.Tanh(),
    use_bias=True, diversity_type='architecture',
    mu_dt_range=(0.005, 0.02)
)
parallel_opt = optim.get_optim(parallel_model.params, "Adam", 1e-3, grad_clip=50)

with torch.no_grad():
    for epoch in range(1, n_epochs + 1):
        for batch_id, (img, lbl) in enumerate(train_loader):
            parallel_model.train_batch_supervised(img, lbl, n_train_iters)
            parallel_opt.step(epoch, batch_id, len(train_loader), img.size(0))
print("  ✓ Parallel PCN trained")

# Test both models on all corruption types
print("\n[2/3] Testing robustness across all corruption types...")
all_results = {'Single PCN': {}, 'Parallel PCN': {}}

for corruption_name, config in corruption_configs.items():
    print(f"\n  Testing {corruption_name}...")
    corruption_type = config['type']
    corruption_levels = config['levels']
    
    single_accs = []
    parallel_accs = []
    
    for level in corruption_levels:
        utils.seed(base_seed)
        test_loader = datasets.get_dataloader(test_dataset, test_size)
        
        single_acc = 0
        parallel_acc = 0
        
        for img, lbl in test_loader:
            corrupted = utils.corrupt_images(img, corruption_type, level) if level > 0 else img
            
            single_pred = single_model.test_batch_supervised(corrupted)
            single_acc += datasets.accuracy(single_pred, lbl)
            
            parallel_pred = parallel_model.test_batch_supervised(corrupted, 'average')
            parallel_acc += datasets.accuracy(parallel_pred, lbl)
        
        single_accs.append(single_acc / len(test_loader))
        parallel_accs.append(parallel_acc / len(test_loader))
    
    all_results['Single PCN'][corruption_name] = {
        'levels': corruption_levels,
        'accuracy': single_accs
    }
    all_results['Parallel PCN'][corruption_name] = {
        'levels': corruption_levels,
        'accuracy': parallel_accs
    }
    
    print(f"    Clean: Single {single_accs[0]:.1%}, Parallel {parallel_accs[0]:.1%}")
    print(f"    Max corruption: Single {single_accs[-1]:.1%}, Parallel {parallel_accs[-1]:.1%}")
    print(f"    Improvement: +{(parallel_accs[-1] - single_accs[-1])*100:.1f}%")

# Generate comprehensive plots
print("\n[3/3] Generating plots...")

# PLOT 1: Individual accuracy curves for each corruption type
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle('Robustness Comparison: Single vs Parallel PCN', 
              fontsize=16, weight='bold', y=1.02)

corruption_names = list(corruption_configs.keys())
for idx, (ax, corruption_name) in enumerate(zip(axes, corruption_names)):
    single_data = all_results['Single PCN'][corruption_name]
    parallel_data = all_results['Parallel PCN'][corruption_name]
    
    ax.plot(single_data['levels'], single_data['accuracy'],
            marker='o', linewidth=2.5, markersize=7, 
            color='red', label='Single PCN')
    ax.plot(parallel_data['levels'], parallel_data['accuracy'],
            marker='s', linewidth=2.5, markersize=7,
            color='blue', label='Parallel PCN (n=3)')
    
    ax.set_title(corruption_name, fontsize=13, weight='bold')
    ax.set_xlabel(f'{corruption_name} Level', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('results/accuracy_all_corruptions.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: results/accuracy_all_corruptions.png")

# PLOT 2: Accuracy degradation comparison
fig2, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Accuracy Degradation: Clean vs Maximally Corrupted',
             fontsize=14, weight='bold')

x_pos = np.arange(len(corruption_names))
width = 0.35

single_clean = [all_results['Single PCN'][name]['accuracy'][0] 
                for name in corruption_names]
single_corrupt = [all_results['Single PCN'][name]['accuracy'][-1] 
                  for name in corruption_names]
parallel_clean = [all_results['Parallel PCN'][name]['accuracy'][0] 
                  for name in corruption_names]
parallel_corrupt = [all_results['Parallel PCN'][name]['accuracy'][-1] 
                    for name in corruption_names]

single_degradation = [c - cor for c, cor in zip(single_clean, single_corrupt)]
parallel_degradation = [c - cor for c, cor in zip(parallel_clean, parallel_corrupt)]

bars1 = ax.bar(x_pos - width/2, single_degradation, width, 
               label='Single PCN', color='red', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, parallel_degradation, width,
               label='Parallel PCN', color='blue', alpha=0.8)

ax.set_ylabel('Accuracy Degradation', fontsize=12)
ax.set_xlabel('Corruption Type', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(corruption_names)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/accuracy_degradation.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: results/accuracy_degradation.png")

# PLOT 3: Improvement over single PCN
fig3, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Parallel PCN Improvement Over Single PCN',
             fontsize=14, weight='bold')

for corruption_name in corruption_names:
    single_data = all_results['Single PCN'][corruption_name]
    parallel_data = all_results['Parallel PCN'][corruption_name]
    
    improvement = [(p - s) * 100 for s, p in 
                   zip(single_data['accuracy'], parallel_data['accuracy'])]
    
    ax.plot(single_data['levels'], improvement,
            marker='o', linewidth=2.5, markersize=7,
            label=corruption_name)

ax.set_xlabel('Corruption Level', fontsize=12)
ax.set_ylabel('Accuracy Improvement (%)', fontsize=12)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('results/accuracy_improvement.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: results/accuracy_improvement.png")

# PLOT 4: Summary bar chart
fig4, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Summary: Accuracy at Maximum Corruption',
             fontsize=14, weight='bold')

x_pos = np.arange(len(corruption_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, 
               [all_results['Single PCN'][name]['accuracy'][-1] for name in corruption_names],
               width, label='Single PCN', color='red', alpha=0.8)
bars2 = ax.bar(x_pos + width/2,
               [all_results['Parallel PCN'][name]['accuracy'][-1] for name in corruption_names],
               width, label='Parallel PCN (n=3)', color='blue', alpha=0.8)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Corruption Type', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(corruption_names)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/accuracy_summary.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: results/accuracy_summary.png")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for corruption_name in corruption_names:
    print(f"\n{corruption_name}:")
    single_data = all_results['Single PCN'][corruption_name]
    parallel_data = all_results['Parallel PCN'][corruption_name]
    
    single_clean = single_data['accuracy'][0]
    single_corrupt = single_data['accuracy'][-1]
    parallel_clean = parallel_data['accuracy'][0]
    parallel_corrupt = parallel_data['accuracy'][-1]
    
    print(f"  Single PCN:   {single_clean:.1%} → {single_corrupt:.1%} (Δ = {(single_clean-single_corrupt):.1%})")
    print(f"  Parallel PCN: {parallel_clean:.1%} → {parallel_corrupt:.1%} (Δ = {(parallel_clean-parallel_corrupt):.1%})")
    print(f"  Improvement:  +{(parallel_corrupt - single_corrupt)*100:.1f}% at max corruption")
    print(f"  Robustness gain: {((single_clean-single_corrupt) - (parallel_clean-parallel_corrupt))*100:.1f}% less degradation")

print("\n" + "="*70)
print("All plots saved to results/ directory")
print("="*70)
