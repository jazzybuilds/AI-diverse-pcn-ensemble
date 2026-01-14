import random
import json
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Activation(object):
    def forward(self, inp):
        raise NotImplementedError

    def deriv(self, inp):
        raise NotImplementedError

    def __call__(self, inp):
        return self.forward(inp)


class Linear(Activation):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return set_tensor(torch.ones((1,)))


class ReLU(Activation):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out


class Tanh(Activation):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0


def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_tensor(tensor):
    return tensor.to(DEVICE).float()


def flatten_array(array):
    return torch.flatten(torch.cat(array, dim=1))


def save_json(obj, path):
    with open(path, "w") as file:
        json.dump(obj, file)


def load_json(path):
    with open(path) as file:
        return json.load(file)

# Corruption utilities for robustness experiments
def add_gaussian_noise(images, std=0.1):
    """Add Gaussian noise to images."""
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)


def add_salt_pepper_noise(images, prob=0.05):
    """Add salt-and-pepper noise to images."""
    mask = torch.rand_like(images)
    noisy = images.clone()
    noisy[mask < prob/2] = 0  # pepper
    noisy[mask > 1 - prob/2] = 1  # salt
    return noisy


def occlude_images(images, occlude_fraction=0.3, img_size=28):
    """Randomly occlude a fraction of the image."""
    occluded = images.clone()
    batch_size = images.size(0)
    n_pixels = img_size * img_size
    n_occlude = int(n_pixels * occlude_fraction)
    
    for b in range(batch_size):
        indices = torch.randperm(n_pixels)[:n_occlude]
        occluded[b, indices] = 0
    
    return occluded


def corrupt_images(images, corruption_type='gaussian', level=0.1, img_size=28):
    """
    Apply corruption to images.
    Args:
        corruption_type: 'gaussian', 'salt_pepper', 'occlude'
        level: corruption strength (std for gaussian, prob for salt_pepper, fraction for occlude)
    """
    if corruption_type == 'gaussian':
        return add_gaussian_noise(images, std=level)
    elif corruption_type == 'salt_pepper':
        return add_salt_pepper_noise(images, prob=level)
    elif corruption_type == 'occlude':
        return occlude_images(images, occlude_fraction=level, img_size=img_size)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")