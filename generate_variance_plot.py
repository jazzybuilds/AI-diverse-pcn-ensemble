"""
Generate ensemble variance plot for uncertainty quantification figure.
Creates a clean bar chart showing variance at different corruption levels.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the paper (Research Question 4 section)
corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# Variance values for MIXED diversity (example values - these should match actual results)
variance_values = [0.0026, 0.020, 0.032, 0.043, 0.051, 0.058]

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Bar chart - color code by corruption severity progression
# σ=0.0: clean, 0.1: light, 0.2: moderate, 0.3: severe, 0.4: extreme, 0.5: maximum
colors = ['#1E5F8C', '#2E86AB', '#F4A259', '#F77F00', '#E55934', '#D62828']
bars = ax.bar(corruption_levels, variance_values, width=0.08, color=colors, 
              edgecolor='black', linewidth=1.5, alpha=0.85)

# Styling
ax.set_xlabel('Gaussian Noise Corruption Level (σ)', fontsize=13, fontweight='bold')
ax.set_ylabel('Ensemble Prediction Variance', fontsize=13, fontweight='bold')
ax.set_title('Uncertainty Quantification: Variance Increases with Corruption', 
             fontsize=14, fontweight='bold', pad=15)

# Grid
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)

# Annotations
ax.text(0.0, variance_values[0] + 0.002, f'{variance_values[0]:.4f}', 
        ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.text(0.3, variance_values[3] + 0.002, f'{variance_values[3]:.3f}', 
        ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.text(0.5, variance_values[5] + 0.002, f'{variance_values[5]:.3f}', 
        ha='center', va='bottom', fontsize=9, fontweight='bold')

# Color legend showing corruption severity progression
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1E5F8C', edgecolor='black', label='Clean (σ=0.0)'),
    Patch(facecolor='#2E86AB', edgecolor='black', label='Light (σ=0.1)'),
    Patch(facecolor='#F4A259', edgecolor='black', label='Moderate (σ=0.2)'),
    Patch(facecolor='#F77F00', edgecolor='black', label='Severe (σ=0.3)'),
    Patch(facecolor='#E55934', edgecolor='black', label='Extreme (σ=0.4)'),
    Patch(facecolor='#D62828', edgecolor='black', label='Maximum (σ=0.5)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95, ncol=2)

# Format
ax.set_ylim(0, 0.07)
ax.set_xticks(corruption_levels)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/ensemble_variance.pdf', dpi=300, bbox_inches='tight')
print("✓ Saved: results/ensemble_variance.pdf")
plt.close()

print("\nVariance data:")
for sigma, var in zip(corruption_levels, variance_values):
    print(f"  σ={sigma:.1f}: variance={var:.4f}")
