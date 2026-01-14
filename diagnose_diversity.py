"""
Diagnose why MIXED doesn't improve much over Architecture
Check individual model performance within each ensemble
"""
import sys
import pathlib
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pypc import utils, datasets, optim
from pypc.models import PCModel, ParallelPCModel

base_seed = 42
utils.seed(base_seed)

# Same setup as test_combined_diversity
train_dataset = datasets.MNIST(train=True, normalize=False, size=2000)
test_dataset = datasets.MNIST(train=False, normalize=False, size=1000)
batch_size = 640
n_epochs = 5
n_train_iters = 100
nodes = [784, 300, 100, 10]

print("="*70)
print("DIAGNOSING DIVERSITY MECHANISMS")
print("="*70)

# Train architecture and mixed models
models_to_test = {
    'Architecture': 'architecture',
    'MIXED': 'mixed'
}

trained_models = {}

for name, diversity_type in models_to_test.items():
    print(f"\nTraining {name}...")
    utils.seed(base_seed)
    train_loader = datasets.get_dataloader(train_dataset, batch_size)
    
    model = ParallelPCModel(
        n_models=10, nodes=nodes, mu_dt=0.01, act_fn=utils.Tanh(),
        use_bias=True, diversity_type=diversity_type,
        mu_dt_range=(0.001, 0.1)
    )
    opt = optim.get_optim(model.params, "Adam", 1e-3, grad_clip=50)
    
    with torch.no_grad():
        for epoch in range(1, n_epochs + 1):
            for batch_id, (img, lbl) in enumerate(train_loader):
                model.train_batch_supervised(img, lbl, n_train_iters)
                opt.step(epoch, batch_id, len(train_loader), img.size(0))
    
    trained_models[name] = model
    print(f"  ✓ Done")

# Test each individual model within the ensembles
print("\n" + "="*70)
print("INDIVIDUAL MODEL PERFORMANCE @ σ=0.3")
print("="*70)

corruption_level = 0.3
utils.seed(base_seed)
test_loader = datasets.get_dataloader(test_dataset, 1000)

for ensemble_name, parallel_model in trained_models.items():
    print(f"\n{ensemble_name} Ensemble:")
    
    # Get predictions from each individual model
    individual_accs = []
    
    for model_idx in range(parallel_model.n_models):
        acc = 0
        
        for img, lbl in test_loader:
            corrupted = utils.corrupt_images(img, 'gaussian', corruption_level)
            
            # Test just this one model
            single_model = parallel_model.models[model_idx]
            pred = single_model.test_batch_supervised(corrupted)
            acc += datasets.accuracy(pred, lbl)
        
        model_acc = acc / len(test_loader)
        individual_accs.append(model_acc)
        
        # Show mu_dt if it varies
        if hasattr(single_model, 'mu_dt'):
            mu_dt = single_model.mu_dt
        else:
            mu_dt = 0.01
        
        print(f"  Model {model_idx+1:2d}: {model_acc:.1%}  (mu_dt={mu_dt:.4f})")
    
    # Ensemble accuracy
    utils.seed(base_seed)
    test_loader = datasets.get_dataloader(test_dataset, 1000)
    ensemble_acc = 0
    for img, lbl in test_loader:
        corrupted = utils.corrupt_images(img, 'gaussian', corruption_level)
        pred = parallel_model.test_batch_supervised(corrupted, 'average')
        ensemble_acc += datasets.accuracy(pred, lbl)
    ensemble_acc = ensemble_acc / len(test_loader)
    
    print(f"  ---")
    print(f"  Individual mean: {np.mean(individual_accs):.1%} ± {np.std(individual_accs):.1%}")
    print(f"  Ensemble (avg):  {ensemble_acc:.1%}")
    print(f"  Benefit:         +{(ensemble_acc - np.mean(individual_accs))*100:.1f}%")
    print(f"  Best individual: {max(individual_accs):.1%}")
    print(f"  Worst individual: {min(individual_accs):.1%}")
    print(f"  Spread (max-min): {(max(individual_accs) - min(individual_accs))*100:.1f}%")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("\nIf MIXED has:")
print("- Similar mean to Architecture → diversity mechanisms similar strength")
print("- Higher spread → more variance but averaging cancels benefits") 
print("- Worse worst-model → some models unstable (mu_dt too large)")
print("- Similar ensemble → diminishing returns from additional diversity")
