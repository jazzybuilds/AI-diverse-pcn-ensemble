import numpy as np
import torch

from pypc import utils
from pypc.layers import FCLayer


class PCModel(object):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False):
        self.nodes = nodes
        self.mu_dt = mu_dt

        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1

        self.layers = []
        for l in range(self.n_layers):
            _act_fn = utils.Linear() if (l == self.n_layers - 1) else act_fn

            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            self.layers.append(layer)

    def reset(self):
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            )

    def set_input(self, inp):
        self.mus[0] = inp.clone()

    def set_target(self, target):
        self.mus[-1] = target.clone()

    def forward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def propagate_mu(self):
        for l in range(1, self.n_layers):
            self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_batch_supervised(self, img_batch, label_batch, n_iters, fixed_preds=False):
        self.reset()
        self.set_input(img_batch)
        self.propagate_mu()
        self.set_target(label_batch)
        self.train_updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads()

    def train_batch_generative(self, img_batch, label_batch, n_iters, fixed_preds=False):
        self.reset()
        self.set_input(label_batch)
        self.propagate_mu()
        self.set_target(img_batch)
        self.train_updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads()

    def test_batch_supervised(self, img_batch):
        return self.forward(img_batch)

    def test_batch_generative(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.test_updates(n_iters, fixed_preds=fixed_preds)
        return self.mus[0]

    def train_updates(self, n_iters, fixed_preds=False):
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for itr in range(n_iters):
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
                
    def test_updates(self, n_iters, fixed_preds):
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for itr in range(n_iters):
            delta = self.layers[0].backward(self.errs[1])
            self.mus[0] = self.mus[0] + self.mu_dt * delta
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

    def update_grads(self):
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def get_target_loss(self):
        return torch.sum(self.errs[-1] ** 2).item()

    @property
    def params(self):
        return self.layers


class ParallelPCModel(object):
    """Ensemble of multiple PCModels for robust parallel inference."""
    
    def __init__(self, n_models, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False, 
                 diversity_type='init', mu_dt_range=None):
        """
        Args:
            n_models: number of parallel PCN streams
            nodes, act_fn, use_bias, kaiming_init: same as PCModel
            mu_dt: base step size (or list of step sizes if diversity_type='dynamics')
            diversity_type: 
                'init' - same arch/dynamics, different random init only
                'dynamics' - different inference step sizes (mu_dt)
                'architecture' - different layer configurations
                'mixed' - combines dynamics + architecture diversity (recommended!)
            mu_dt_range: (min, max) for dynamics diversity, e.g., (0.005, 0.05)
        """
        self.n_models = n_models
        self.diversity_type = diversity_type
        self.models = []
        
        # Save initial random state to control init diversity
        # Only 'init' and 'mixed' should have different random initializations
        import random
        initial_torch_state = torch.get_rng_state()
        initial_np_state = np.random.get_state()
        initial_python_state = random.getstate()
        
        for i in range(n_models):
            # Control initialization diversity based on diversity_type
            # 'dynamics' and 'architecture' should have IDENTICAL initial weights
            if diversity_type == 'dynamics' or diversity_type == 'architecture':
                # Reset to same initial state for all models
                torch.set_rng_state(initial_torch_state)
                np.random.set_state(initial_np_state)
                random.setstate(initial_python_state)
            # 'init' and 'mixed' allow random state to advance naturally (different inits)
            
            # Determine mu_dt (dynamics diversity)
            if diversity_type in ['dynamics', 'mixed'] and mu_dt_range is not None:
                # Spread step sizes across the range
                model_mu_dt = mu_dt_range[0] + (mu_dt_range[1] - mu_dt_range[0]) * i / max(1, n_models - 1)
            elif isinstance(mu_dt, list):
                model_mu_dt = mu_dt[i] if i < len(mu_dt) else mu_dt[-1]
            else:
                model_mu_dt = mu_dt
            
            # Determine architecture (architectural diversity)
            if diversity_type in ['architecture', 'mixed']:
                # Vary architecture while keeping params roughly constant
                model_nodes = self._vary_architecture(nodes, i, n_models)
            else:
                model_nodes = nodes
                
            model = PCModel(
                nodes=model_nodes,
                mu_dt=model_mu_dt,
                act_fn=act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init
            )
            self.models.append(model)
    
    def _vary_architecture(self, base_nodes, idx, n_models):
        """Create architectural diversity (e.g., deep-narrow vs shallow-wide)."""
        if idx == 0:
            return base_nodes  # keep one baseline
        elif idx % 2 == 1:
            # Make deeper and narrower
            new_nodes = [base_nodes[0]] + [int(n * 0.7) for n in base_nodes[1:-1]] + [base_nodes[-1]]
            # Add an extra layer
            new_nodes.insert(2, int((new_nodes[1] + new_nodes[2]) / 2))
            return new_nodes
        else:
            # Make shallower and wider
            new_nodes = [base_nodes[0]] + [int(n * 1.3) for n in base_nodes[1:-1]] + [base_nodes[-1]]
            return new_nodes
    
    def train_batch_supervised(self, img_batch, label_batch, n_iters, fixed_preds=False):
        """Train all models in parallel."""
        for model in self.models:
            model.train_batch_supervised(img_batch, label_batch, n_iters, fixed_preds)
    
    def test_batch_supervised(self, img_batch, ensemble_method='average'):
        """
        Test with ensemble inference.
        Args:
            ensemble_method: 'average' (mean of outputs), 'vote' (majority), or 'max' (max confidence)
        """
        predictions = []
        for model in self.models:
            pred = model.test_batch_supervised(img_batch)
            predictions.append(pred)
        
        # Stack predictions: [n_models, batch_size, n_classes]
        predictions = torch.stack(predictions, dim=0)
        
        if ensemble_method == 'average':
            return torch.mean(predictions, dim=0)
        elif ensemble_method == 'vote':
            # Majority vote on argmax
            votes = torch.argmax(predictions, dim=2)  # [n_models, batch_size]
            batch_size = votes.size(1)
            n_classes = predictions.size(2)
            result = torch.zeros(batch_size, n_classes).to(predictions.device)
            for b in range(batch_size):
                vote_counts = torch.bincount(votes[:, b], minlength=n_classes)
                result[b, torch.argmax(vote_counts)] = 1.0
            return result
        elif ensemble_method == 'max':
            # Take max confidence across models
            return torch.max(predictions, dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def get_prediction_variance(self, img_batch):
        """Measure disagreement/variance across parallel models."""
        predictions = []
        for model in self.models:
            pred = model.test_batch_supervised(img_batch)
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=0)  # [n_models, batch, classes]
        return torch.var(predictions, dim=0).mean().item()
    
    @property
    def params(self):
        """Return all parameters from all models."""
        all_params = []
        for model in self.models:
            all_params.extend(model.params)
        return all_params
