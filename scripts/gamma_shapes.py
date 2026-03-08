import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Create time points
t = np.linspace(0, 10, 1000)

# Fixed rate parameter
lambda_rate = 1.0

# Different shape parameters (r values)
r_values = [1, 2, 3, 5, 10]

# Create plot
plt.figure(figsize=(12, 8))

for r in r_values:
    # Gamma distribution with shape r and rate lambda
    # scipy uses (a, scale) where a=r and scale=1/lambda
    pdf = gamma.pdf(t, a=r, scale=1/lambda_rate)
    
    label = f'r={r}'
    if r == 1:
        label += ' (Exponential!)'
    
    plt.plot(t, pdf, linewidth=2.5, label=label)

plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('Gamma Distribution Shapes for Different r Values (λ=1)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.tight_layout()

# Add annotations
plt.text(0.5, 0.9, 'r=1: L-shaped\n(exponential)', transform=plt.gca().transAxes, 
         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.text(0.5, 0.75, 'r increases:\n→ peak shifts right\n→ more symmetric', 
         transform=plt.gca().transAxes, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.show()

print("\n📊 Key observations:")
print("- r=1 (Exponential): Starts high, decays immediately")
print("- r=2,3: Peak emerges, shifts right")
print("- r=5: Clear bell-like shape, still right-skewed")
print("- r=10: Nearly symmetric, approaching Normal distribution")
print(f"\nPeak location ≈ (r-1)/λ")
