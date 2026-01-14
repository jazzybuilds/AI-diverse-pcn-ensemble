"""
Quick experiment to verify parallel PCN improves robustness
Reduced parameters for fast testing (~5 minutes)
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
print("QUICK PARALLEL PCN VERIFICATION")
print("="*70)

utils.seed(42)

# Small dataset for speed
print("\n[1/6] Loading dataset...")
train_dataset = datasets.MNIST(train=True, normalize=False, size=5000)
test_dataset = datasets.MNIST(train=False, normalize=False, size=1000)
train_loader = datasets.get_dataloader(train_dataset, batch_size=640)
test_loader = datasets.get_dataloader(test_dataset, batch_size=1000)
print("✓ Loaded")

# Config
n_epochs = 5
n_train_iters = 100
nodes = [784, 300, 100, 10]
mu_dt = 0.01

# Train Single PCN
print(f"\n[2/6] Training Single PCN ({n_epochs} epochs)...")
single_model = PCModel(
    nodes=nodes, mu_dt=mu_dt, act_fn=utils.Tanh(), use_bias=True
)
single_opt = optim.get_optim(single_model.params, "Adam", 1e-3, grad_clip=50)

with torch.no_grad():
    for epoch in range(1, n_epochs + 1):
        for batch_id, (img_batch, label_batch) in enumerate(train_loader):
            single_model.train_batch_supervised(img_batch, label_batch, n_train_iters)
            single_opt.step(curr_epoch=epoch, curr_batch=batch_id,
                          n_batches=len(train_loader), batch_size=img_batch.size(0))
        
        if epoch % 2 == 0:
            acc = sum(datasets.accuracy(single_model.test_batch_supervised(img), lbl) 
                     for img, lbl in test_loader) / len(test_loader)
            print(f"  Epoch {epoch}: {acc:.2%}")
print("✓ Trained")

# Train Parallel PCN (3 models, dynamics diversity)
print(f"\n[3/6] Training Parallel PCN (3 models, dynamics diversity, {n_epochs} epochs)...")
parallel_model = ParallelPCModel(
    n_models=3, nodes=nodes, mu_dt=mu_dt, act_fn=utils.Tanh(),
    use_bias=True, diversity_type='dynamics', mu_dt_range=(0.005, 0.02)
)
parallel_opt = optim.get_optim(parallel_model.params, "Adam", 1e-3, grad_clip=50)

with torch.no_grad():
    for epoch in range(1, n_epochs + 1):
        for batch_id, (img_batch, label_batch) in enumerate(train_loader):
            parallel_model.train_batch_supervised(img_batch, label_batch, n_train_iters)
            parallel_opt.step(curr_epoch=epoch, curr_batch=batch_id,
                            n_batches=len(train_loader), batch_size=img_batch.size(0))
        
        if epoch % 2 == 0:
            acc = sum(datasets.accuracy(parallel_model.test_batch_supervised(img, 'average'), lbl)
                     for img, lbl in test_loader) / len(test_loader)
            print(f"  Epoch {epoch}: {acc:.2%}")
print("✓ Trained")

# Test on clean images
print("\n[4/6] Testing on clean images...")
single_clean = sum(datasets.accuracy(single_model.test_batch_supervised(img), lbl)
                   for img, lbl in test_loader) / len(test_loader)
parallel_clean = sum(datasets.accuracy(parallel_model.test_batch_supervised(img, 'average'), lbl)
                     for img, lbl in test_loader) / len(test_loader)

print(f"  Single PCN:   {single_clean:.2%}")
print(f"  Parallel PCN: {parallel_clean:.2%}")

# Test robustness to corruption
print("\n[5/6] Testing robustness to Gaussian noise...")
corruption_levels = [0.0, 0.1, 0.2, 0.3]
single_accs = []
parallel_accs = []
parallel_vars = []

for level in corruption_levels:
    single_acc = 0
    parallel_acc = 0
    parallel_var = 0
    
    for img, lbl in test_loader:
        if level > 0:
            corrupted = utils.corrupt_images(img, 'gaussian', level)
        else:
            corrupted = img
        
        single_acc += datasets.accuracy(single_model.test_batch_supervised(corrupted), lbl)
        parallel_acc += datasets.accuracy(parallel_model.test_batch_supervised(corrupted, 'average'), lbl)
        parallel_var += parallel_model.get_prediction_variance(corrupted)
    
    single_acc /= len(test_loader)
    parallel_acc /= len(test_loader)
    parallel_var /= len(test_loader)
    
    single_accs.append(single_acc)
    parallel_accs.append(parallel_acc)
    parallel_vars.append(parallel_var)
    
    print(f"  σ={level:.1f}: Single {single_acc:.2%}, Parallel {parallel_acc:.2%} (var={parallel_var:.4f})")

# Plot results
print("\n[6/6] Generating plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs corruption
ax1.plot(corruption_levels, single_accs, 'o-', linewidth=2, markersize=8,
         label='Single PCN', color='#2E86AB')
ax1.plot(corruption_levels, parallel_accs, 's-', linewidth=2, markersize=8,
         label='Parallel PCN (3 models)', color='#F18F01')
ax1.set_xlabel('Gaussian Noise Level (σ)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Robustness Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add improvement annotation
improvement = parallel_accs[-1] - single_accs[-1]
ax1.annotate(f'Parallel better\nby {improvement:+.1%}', 
            xy=(corruption_levels[-1], parallel_accs[-1]), 
            xytext=(corruption_levels[-2], (parallel_accs[-1] + single_accs[-1])/2),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green'))

# Plot 2: Variance (uncertainty)
ax2.plot(corruption_levels, parallel_vars, '^-', linewidth=2, markersize=8,
         color='#A23B72', label='Ensemble Variance')
ax2.set_xlabel('Gaussian Noise Level (σ)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prediction Variance', fontsize=12, fontweight='bold')
ax2.set_title('Uncertainty Calibration', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
results_dir = pathlib.Path("results")
results_dir.mkdir(exist_ok=True)
save_path = results_dir / "quick_verification.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.show()

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\nClean Accuracy:")
print(f"  Single PCN:   {single_clean:.2%}")
print(f"  Parallel PCN: {parallel_clean:.2%}")

print(f"\nRobustness at σ=0.3:")
print(f"  Single PCN:   {single_accs[-1]:.2%}")
print(f"  Parallel PCN: {parallel_accs[-1]:.2%}")
print(f"  Improvement:  {improvement:+.1%}")

# Area under curve
auc_single = np.trapezoid(single_accs, corruption_levels)
auc_parallel = np.trapezoid(parallel_accs, corruption_levels)
print(f"\nArea Under Curve (Overall Robustness):")
print(f"  Single PCN:   {auc_single:.3f}")
print(f"  Parallel PCN: {auc_parallel:.3f}")
print(f"  Improvement:  {auc_parallel - auc_single:+.3f}")

print("\n" + "="*70)
if improvement > 0.05:
    print("✅ SUCCESS: Parallel PCN shows significant robustness improvement!")
    print(f"   Parallel maintains {improvement:.1%} higher accuracy under noise")
elif improvement > 0:
    print("✓ Parallel PCN shows modest improvement")
else:
    print("⚠️  Parallel not better (may need more training or different diversity)")

print("\nUncertainty increases with corruption: " + 
      ("✓ Yes" if parallel_vars[-1] > parallel_vars[0] * 2 else "✗ No"))
print("="*70)
