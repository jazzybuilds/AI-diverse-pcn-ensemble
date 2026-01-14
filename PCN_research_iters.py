import sys
import pathlib
import matplotlib.pyplot as plt
import torch
import numpy as np

# Add parent directory to path to import pypc
sys.path.append(str(pathlib.Path(__file__).parent))

from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel

def run_experiment(n_iters, cf):
    print(f"\n--- Running Experiment with n_train_iters = {n_iters} ---")
    utils.seed(cf.seed)

    # Load Data
    train_dataset = datasets.MNIST(
        train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize
    )
    test_dataset = datasets.MNIST(
        train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize
    )
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.test_size)

    # Initialize Model
    model = PCModel(
        nodes=cf.nodes,
        mu_dt=cf.mu_dt,
        act_fn=cf.act_fn,
        use_bias=cf.use_bias,
        kaiming_init=cf.kaiming_init,
    )
    optimizer = optim.get_optim(
        model.params,
        cf.optim,
        cf.lr,
        batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip,
        weight_decay=cf.weight_decay,
    )

    # Training Loop
    acc_history = []
    with torch.no_grad():
        for epoch in range(1, cf.n_epochs + 1):
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                # Use the variable n_iters here
                model.train_batch_supervised(
                    img_batch, label_batch, n_iters, fixed_preds=cf.fixed_preds_train
                )
                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )
            
            # Testing
            if epoch % cf.test_every == 0:
                acc = 0
                for _, (img_batch, label_batch) in enumerate(test_loader):
                    label_preds = model.test_batch_supervised(img_batch)
                    acc += datasets.accuracy(label_preds, label_batch)
                current_acc = acc / len(test_loader)
                acc_history.append(current_acc)
                print(f"Epoch {epoch} | Accuracy: {current_acc:.2%}")
                
    return acc_history

if __name__ == "__main__":
    # Configuration
    cf = utils.AttrDict()
    cf.seed = 1
    cf.n_epochs = 5  # Reduced for quick testing
    cf.test_every = 1
    cf.train_size = None
    cf.test_size = 10000
    cf.label_scale = None
    cf.normalize = False
    cf.optim = "Adam"
    cf.lr = 1e-3
    cf.batch_size = 640
    cf.batch_scale = False
    cf.grad_clip = 50
    cf.weight_decay = None
    
    # Model Params
    cf.nodes = [784, 300, 100, 10]
    cf.mu_dt = 0.05
    cf.act_fn = utils.ReLU()
    cf.use_bias = True
    cf.kaiming_init = True
    cf.fixed_preds_train = False

    # --- RESEARCH VARIABLE ---
    iteration_values = [10, 20, 50] # Compare these values
    results = {}

    for iters in iteration_values:
        acc_hist = run_experiment(iters, cf)
        results[iters] = acc_hist

    # Plotting Results
    plt.figure(figsize=(10, 6))
    for iters, acc_hist in results.items():
        plt.plot(range(1, cf.n_epochs + 1), acc_hist, label=f'n_iters={iters}')
    
    plt.title("Impact of Inference Iterations on Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("research_results.png")
    print("\nExperiment complete. Results saved to research_results.png")
