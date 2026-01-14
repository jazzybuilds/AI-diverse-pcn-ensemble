"""
Visualize what different corruption types look like on MNIST
Run this to see examples for your report
"""

import sys
import pathlib
import matplotlib.pyplot as plt
import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pypc import utils, datasets


def visualize_corruptions():
    """Show examples of different corruption types."""
    # Load a few MNIST images
    test_dataset = datasets.MNIST(train=False, normalize=False, size=100)
    test_loader = datasets.get_dataloader(test_dataset, batch_size=8)
    img_batch, label_batch = next(iter(test_loader))
    
    # Define corruption levels to show
    gaussian_levels = [0.0, 0.1, 0.2, 0.3]
    salt_pepper_levels = [0.0, 0.05, 0.1, 0.2]
    occlude_levels = [0.0, 0.1, 0.3, 0.5]
    
    # Show first image under different corruptions
    sample_img = img_batch[0:1]  # Take first image
    sample_label = torch.argmax(label_batch[0]).item()
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Row 1: Gaussian noise
    for i, level in enumerate(gaussian_levels):
        corrupted = utils.corrupt_images(sample_img, 'gaussian', level)
        img = corrupted[0].cpu().numpy().reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Gaussian σ={level}', fontsize=10)
        axes[0, i].axis('off')
    
    # Row 2: Salt-and-pepper
    for i, level in enumerate(salt_pepper_levels):
        corrupted = utils.corrupt_images(sample_img, 'salt_pepper', level)
        img = corrupted[0].cpu().numpy().reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Salt-Pepper p={level}', fontsize=10)
        axes[1, i].axis('off')
    
    # Row 3: Occlusion
    for i, level in enumerate(occlude_levels):
        corrupted = utils.corrupt_images(sample_img, 'occlude', level, img_size=28)
        img = corrupted[0].cpu().numpy().reshape(28, 28)
        axes[2, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Occlude {int(level*100)}%', fontsize=10)
        axes[2, i].axis('off')
    
    fig.suptitle(f'Corruption Types on MNIST Digit: {sample_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "corruption_examples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()
    
    # Show multiple digits under one corruption level
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    corruption_level = 0.2
    
    # Top row: clean
    for i in range(8):
        img = img_batch[i].cpu().numpy().reshape(28, 28)
        label = torch.argmax(label_batch[i]).item()
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Clean: {label}', fontsize=10)
        axes[0, i].axis('off')
    
    # Bottom row: corrupted
    corrupted_batch = utils.corrupt_images(img_batch, 'gaussian', corruption_level)
    for i in range(8):
        img = corrupted_batch[i].cpu().numpy().reshape(28, 28)
        label = torch.argmax(label_batch[i]).item()
        axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Noisy: {label}', fontsize=10)
        axes[1, i].axis('off')
    
    fig.suptitle(f'Clean vs Gaussian Noise (σ={corruption_level})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = results_dir / "corruption_batch_example.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved batch visualization: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("Generating corruption visualizations for report...")
    visualize_corruptions()
    print("\n✓ Done! Check results/ folder for images to include in your report.")
