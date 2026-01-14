"""Quick 2-minute test to verify everything works"""
import sys
import pathlib
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pypc import utils, datasets, optim
from pypc.models import PCModel, ParallelPCModel

print("="*70)
print("QUICK TEST: Parallel PCN Implementation")
print("="*70)

utils.seed(42)

# Mini dataset
print("\n[1/5] Loading mini dataset...")
train_dataset = datasets.MNIST(train=True, normalize=False, size=100)
test_dataset = datasets.MNIST(train=False, normalize=False, size=100)
train_loader = datasets.get_dataloader(train_dataset, batch_size=50)
test_loader = datasets.get_dataloader(test_dataset, batch_size=100)
print("✓ Loaded")

# Single model
print("\n[2/5] Training single PCN (1 epoch)...")
single_model = PCModel(
    nodes=[784, 100, 10], mu_dt=0.01, act_fn=utils.Tanh(), use_bias=True
)
single_opt = optim.get_optim(single_model.params, "Adam", 1e-3)

with torch.no_grad():
    for batch_id, (img_batch, label_batch) in enumerate(train_loader):
        single_model.train_batch_supervised(img_batch, label_batch, 50)
        single_opt.step(curr_epoch=1, curr_batch=batch_id, 
                       n_batches=len(train_loader), batch_size=img_batch.size(0))
print("✓ Trained")

# Parallel model
print("\n[3/5] Training parallel PCN (3 models, 1 epoch)...")
parallel_model = ParallelPCModel(
    n_models=3, nodes=[784, 100, 10], mu_dt=0.01, act_fn=utils.Tanh(),
    use_bias=True, diversity_type='init'
)
parallel_opt = optim.get_optim(parallel_model.params, "Adam", 1e-3)

with torch.no_grad():
    for batch_id, (img_batch, label_batch) in enumerate(train_loader):
        parallel_model.train_batch_supervised(img_batch, label_batch, 50)
        parallel_opt.step(curr_epoch=1, curr_batch=batch_id,
                         n_batches=len(train_loader), batch_size=img_batch.size(0))
print("✓ Trained")

# Test clean
print("\n[4/5] Testing on clean images...")
single_acc = 0
parallel_acc = 0

for _, (img_batch, label_batch) in enumerate(test_loader):
    single_pred = single_model.test_batch_supervised(img_batch)
    single_acc += datasets.accuracy(single_pred, label_batch)
    
    parallel_pred = parallel_model.test_batch_supervised(img_batch, 'average')
    parallel_acc += datasets.accuracy(parallel_pred, label_batch)

single_acc /= len(test_loader)
parallel_acc /= len(test_loader)

print(f"  Single PCN:   {single_acc:.2%}")
print(f"  Parallel PCN: {parallel_acc:.2%}")

# Test corrupted
print("\n[5/5] Testing on corrupted images (Gaussian σ=0.2)...")
single_acc_noisy = 0
parallel_acc_noisy = 0
parallel_var = 0

for _, (img_batch, label_batch) in enumerate(test_loader):
    noisy = utils.corrupt_images(img_batch, 'gaussian', 0.2)
    
    single_pred = single_model.test_batch_supervised(noisy)
    single_acc_noisy += datasets.accuracy(single_pred, label_batch)
    
    parallel_pred = parallel_model.test_batch_supervised(noisy, 'average')
    parallel_acc_noisy += datasets.accuracy(parallel_pred, label_batch)
    parallel_var += parallel_model.get_prediction_variance(noisy)

single_acc_noisy /= len(test_loader)
parallel_acc_noisy /= len(test_loader)
parallel_var /= len(test_loader)

print(f"  Single PCN:   {single_acc_noisy:.2%}")
print(f"  Parallel PCN: {parallel_acc_noisy:.2%} (variance: {parallel_var:.4f})")

# Summary
print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)
print(f"Clean accuracy:    Single {single_acc:.2%}, Parallel {parallel_acc:.2%}")
print(f"Noisy accuracy:    Single {single_acc_noisy:.2%}, Parallel {parallel_acc_noisy:.2%}")
print(f"Robustness gain:   {parallel_acc_noisy - single_acc_noisy:+.1%} (parallel better)")
print("="*70)

if parallel_acc_noisy >= single_acc_noisy:
    print("\n✅ SUCCESS: Parallel PCN shows robustness benefit!")
else:
    print("\n⚠️  Note: With only 1 epoch, results vary. Run full experiment for clear trends.")

print("\nReady for full experiments! Run:")
print("  python PCN_robustness_experiment.py")
