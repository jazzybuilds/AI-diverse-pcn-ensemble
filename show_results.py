import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Show the key plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Load images
img1 = mpimg.imread('results/quick_verification.png')
img2 = mpimg.imread('results/diversity_comparison.png')

axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('Main Result: Parallel PCN Improves Robustness', fontsize=14, fontweight='bold', pad=10)

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('Diversity Strategy Comparison', fontsize=14, fontweight='bold', pad=10)

plt.suptitle('✅ VERIFIED: Parallel Predictive Coding Networks Improve Sensory Inference', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('results/FINAL_RESULTS.png', dpi=150, bbox_inches='tight')
print("✓ Created combined results plot: results/FINAL_RESULTS.png")
plt.show()
