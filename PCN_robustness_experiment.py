"""
Parallel Predictive Coding Networks for Robust Sensory Inference
Experiment: Compare single vs parallel PCN under various input corruptions
"""

import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from datetime import datetime

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel, ParallelPCModel


def run_robustness_test(model, test_loader, corruption_type, corruption_levels, 
                        is_parallel=False, ensemble_method='average'):
    """
    Test model accuracy across corruption levels.
    Returns: list of (corruption_level, accuracy, variance) tuples
    """
    results = []
    
    for level in corruption_levels:
        acc = 0
        var_sum = 0
        n_batches = 0
        
        for _, (img_batch, label_batch) in enumerate(test_loader):
            # Apply corruption
            if level > 0:
                corrupted = utils.corrupt_images(img_batch, corruption_type, level)
            else:
                corrupted = img_batch
            
            # Get predictions
            if is_parallel:
                label_preds = model.test_batch_supervised(corrupted, ensemble_method)
                variance = model.get_prediction_variance(corrupted)
                var_sum += variance
            else:
                label_preds = model.test_batch_supervised(corrupted)
            
            acc += datasets.accuracy(label_preds, label_batch)
            n_batches += 1
        
        avg_acc = acc / n_batches
        avg_var = var_sum / n_batches if is_parallel else 0
        results.append((level, avg_acc, avg_var))
        
        print(f"  {corruption_type} level {level:.3f}: Accuracy {avg_acc:.2%}, Variance {avg_var:.4f}")
    
    return results


def plot_robustness_curves(results_dict, corruption_type, save_path):
    """Plot accuracy vs corruption level for different models."""
    plt.figure(figsize=(10, 6))
    
    for model_name, results in results_dict.items():
        levels = [r[0] for r in results]
        accs = [r[1] for r in results]
        plt.plot(levels, accs, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel(f'{corruption_type.capitalize()} Corruption Level', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Robustness to {corruption_type.capitalize()} Noise', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close()


def plot_variance_curves(results_dict, corruption_type, save_path):
    """Plot prediction variance vs corruption level."""
    plt.figure(figsize=(10, 6))
    
    for model_name, results in results_dict.items():
        if 'Parallel' in model_name:  # Only plot variance for parallel models
            levels = [r[0] for r in results]
            variances = [r[2] for r in results]
            plt.plot(levels, variances, marker='s', label=model_name, linewidth=2)
    
    plt.xlabel(f'{corruption_type.capitalize()} Corruption Level', fontsize=12)
    plt.ylabel('Prediction Variance', fontsize=12)
    plt.title(f'Ensemble Disagreement under {corruption_type.capitalize()} Noise', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close()


def save_results_json(results_dict, corruption_type, save_path):
    """Save numerical results to JSON."""
    data = {
        'corruption_type': corruption_type,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for model_name, results in results_dict.items():
        data['results'][model_name] = [
            {'level': r[0], 'accuracy': r[1], 'variance': r[2]} 
            for r in results
        ]
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results: {save_path}")


def main(cf):
    print(f"\n{'='*70}")
    print(f"Parallel Predictive Coding Networks for Robust Sensory Inference")
    print(f"{'='*70}")
    print(f"Seed: {cf.seed} | Device: {utils.DEVICE}")
    print(f"Corruption: {cf.corruption_type} | Levels: {cf.corruption_levels}")
    print(f"{'='*70}\n")
    
    utils.seed(cf.seed)

    # Load datasets
    train_dataset = datasets.MNIST(train=True, scale=cf.label_scale, size=cf.train_size, 
                                   normalize=cf.normalize)
    test_dataset = datasets.MNIST(train=False, scale=cf.label_scale, size=cf.test_size, 
                                  normalize=cf.normalize)
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.test_size)

    # ========== Train Single PCN (Baseline) ==========
    print("\n[1/4] Training Single PCN (Baseline)...")
    single_model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, 
        use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    single_optimizer = optim.get_optim(
        single_model.params, cf.optim, cf.lr, batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip, weight_decay=cf.weight_decay
    )

    with torch.no_grad():
        for epoch in range(1, cf.n_epochs + 1):
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                single_model.train_batch_supervised(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train
                )
                single_optimizer.step(
                    curr_epoch=epoch, curr_batch=batch_id,
                    n_batches=len(train_loader), batch_size=img_batch.size(0)
                )
            
            if epoch % cf.test_every == 0:
                acc = 0
                for _, (img_batch, label_batch) in enumerate(test_loader):
                    label_preds = single_model.test_batch_supervised(img_batch)
                    acc += datasets.accuracy(label_preds, label_batch)
                print(f"  Epoch {epoch}: Clean Accuracy {acc / len(test_loader):.2%}")

    # ========== Train Parallel PCN (Ensemble) ==========
    print(f"\n[2/4] Training Parallel PCN ({cf.n_models} models, diversity={cf.diversity_type})...")
    parallel_model = ParallelPCModel(
        n_models=cf.n_models, nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn,
        use_bias=cf.use_bias, kaiming_init=cf.kaiming_init,
        diversity_type=cf.diversity_type, mu_dt_range=cf.mu_dt_range
    )
    parallel_optimizer = optim.get_optim(
        parallel_model.params, cf.optim, cf.lr, batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip, weight_decay=cf.weight_decay
    )

    with torch.no_grad():
        for epoch in range(1, cf.n_epochs + 1):
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                parallel_model.train_batch_supervised(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train
                )
                parallel_optimizer.step(
                    curr_epoch=epoch, curr_batch=batch_id,
                    n_batches=len(train_loader), batch_size=img_batch.size(0)
                )
            
            if epoch % cf.test_every == 0:
                acc = 0
                for _, (img_batch, label_batch) in enumerate(test_loader):
                    label_preds = parallel_model.test_batch_supervised(
                        img_batch, ensemble_method=cf.ensemble_method
                    )
                    acc += datasets.accuracy(label_preds, label_batch)
                print(f"  Epoch {epoch}: Clean Accuracy {acc / len(test_loader):.2%}")

    # ========== Robustness Testing ==========
    print(f"\n[3/4] Testing robustness to {cf.corruption_type} corruption...")
    
    results_dict = {}
    
    print("\n  Testing Single PCN:")
    single_results = run_robustness_test(
        single_model, test_loader, cf.corruption_type, cf.corruption_levels,
        is_parallel=False
    )
    results_dict['Single PCN'] = single_results
    
    print(f"\n  Testing Parallel PCN ({cf.n_models} models, {cf.ensemble_method}):")
    parallel_results = run_robustness_test(
        parallel_model, test_loader, cf.corruption_type, cf.corruption_levels,
        is_parallel=True, ensemble_method=cf.ensemble_method
    )
    results_dict[f'Parallel PCN (n={cf.n_models})'] = parallel_results

    # ========== Save Results and Plots ==========
    print(f"\n[4/4] Generating plots and saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Plot accuracy curves
    plot_path = results_dir / f"robustness_{cf.corruption_type}_{timestamp}.png"
    plot_robustness_curves(results_dict, cf.corruption_type, plot_path)
    
    # Plot variance curves
    var_plot_path = results_dir / f"variance_{cf.corruption_type}_{timestamp}.png"
    plot_variance_curves(results_dict, cf.corruption_type, var_plot_path)
    
    # Save JSON
    json_path = results_dir / f"results_{cf.corruption_type}_{timestamp}.json"
    save_results_json(results_dict, cf.corruption_type, json_path)
    
    print(f"\n{'='*70}")
    print(f"Experiment complete! Results saved to {results_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    cf = utils.AttrDict()
    
    # Experiment params
    cf.seed = 42
    cf.n_epochs = 10
    cf.test_every = 5
    
    # Dataset params
    cf.train_size = None
    cf.test_size = 2000  # Smaller for faster testing
    cf.label_scale = None
    cf.normalize = False
    
    # Optimizer params
    cf.optim = "Adam"
    cf.lr = 1e-3
    cf.batch_size = 640
    cf.batch_scale = False
    cf.grad_clip = 50
    cf.weight_decay = None
    
    # Inference params
    cf.mu_dt = 0.01
    cf.n_train_iters = 200
    cf.fixed_preds_train = False
    
    # Model params
    cf.use_bias = True
    cf.kaiming_init = False
    cf.nodes = [784, 300, 100, 10]
    cf.act_fn = utils.Tanh()
    
    # Parallel PCN params
    cf.n_models = 3  # Number of parallel streams
    cf.diversity_type = 'init'  # 'init', 'dynamics', 'architecture', or 'mixed'
    cf.mu_dt_range = (0.005, 0.02)  # For 'dynamics' diversity
    cf.ensemble_method = 'average'  # 'average', 'vote', or 'max'
    
    # Robustness test params
    cf.corruption_type = 'gaussian'  # 'gaussian', 'salt_pepper', or 'occlude'
    cf.corruption_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    # Run experiment
    main(cf)
    
    print("\nTo run different experiments, modify the config:")
    print("  - cf.corruption_type: 'gaussian', 'salt_pepper', 'occlude'")
    print("  - cf.n_models: 1, 2, 3, 5 (number of parallel streams)")
    print("  - cf.diversity_type: 'init', 'dynamics', 'architecture'")
    print("  - cf.ensemble_method: 'average', 'vote', 'max'")
