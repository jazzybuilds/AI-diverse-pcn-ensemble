"""
Verify init vs dynamics diversity performance
Run with different random seed to check consistency
"""
import sys
import pathlib
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pypc import utils, datasets, optim
from pypc.models import ParallelPCModel

# Test both diversity types with 2 different seeds
seeds = [42, 123]
results = {'init': [], 'dynamics': []}

for seed in seeds:
    print(f"\n{'='*70}")
    print(f"Testing with seed={seed}")
    print('='*70)
    
    utils.seed(seed)
    
    train_dataset = datasets.MNIST(train=True, normalize=False, size=3000)
    test_dataset = datasets.MNIST(train=False, normalize=False, size=1000)
    train_loader = datasets.get_dataloader(train_dataset, batch_size=640)
    test_loader = datasets.get_dataloader(test_dataset, batch_size=1000)
    
    for diversity_type in ['init', 'dynamics']:
        print(f"\nTraining 3-model Parallel PCN ({diversity_type})...")
        
        model = ParallelPCModel(
            n_models=3, nodes=[784, 300, 100, 10], mu_dt=0.01,
            act_fn=utils.Tanh(), use_bias=True,
            diversity_type=diversity_type,
            mu_dt_range=(0.005, 0.02)
        )
        opt = optim.get_optim(model.params, "Adam", 1e-3, grad_clip=50)
        
        # Quick training (3 epochs)
        with torch.no_grad():
            for epoch in range(1, 4):
                for batch_id, (img, lbl) in enumerate(train_loader):
                    model.train_batch_supervised(img, lbl, 100)
                    opt.step(epoch, batch_id, len(train_loader), img.size(0))
        
        # Test at high noise
        acc = 0
        for img, lbl in test_loader:
            corrupted = utils.corrupt_images(img, 'gaussian', 0.3)
            pred = model.test_batch_supervised(corrupted, 'average')
            acc += datasets.accuracy(pred, lbl)
        acc /= len(test_loader)
        
        results[diversity_type].append(acc)
        print(f"  Accuracy @ σ=0.3: {acc:.2%}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY: Init vs Dynamics Diversity")
print('='*70)
print(f"\nInit diversity:     {np.mean(results['init']):.2%} ± {np.std(results['init']):.2%}")
print(f"Dynamics diversity: {np.mean(results['dynamics']):.2%} ± {np.std(results['dynamics']):.2%}")

diff = np.mean(results['init']) - np.mean(results['dynamics'])
print(f"\nDifference: {diff:+.2%} in favor of {'init' if diff > 0 else 'dynamics'}")

if diff > 0.02:
    print("\n✓ Init diversity consistently outperforms dynamics!")
    print("  Possible reasons:")
    print("  - Simpler diversity is more effective with limited training")
    print("  - Different random seeds explore different local minima")
    print("  - Dynamics diversity may need more careful tuning of mu_dt_range")
elif diff < -0.02:
    print("\n✓ Dynamics diversity consistently outperforms init!")
else:
    print("\n≈ Both strategies perform similarly (within margin of error)")

print('='*70)
