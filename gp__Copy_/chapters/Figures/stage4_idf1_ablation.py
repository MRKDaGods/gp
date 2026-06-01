import os
import numpy as np
import matplotlib.pyplot as plt

# Data from docs/findings.md Stage 4 ablation table (w_tertiary -> IDF1)
labels = [
    '0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.40', '0.50', '0.60', '0.70'
]
idf1 = [
    0.7842, 0.7846, 0.7846, 0.7851, 0.7840, 0.7840, 0.7853, 0.7851, 0.7857, 0.7916, 0.7909
]

# Convert to percent
idf1_percent = [v * 100 for v in idf1]

# Styling similar to provided example: horizontal bars, blue gradient, value labels on right
fig, ax = plt.subplots(figsize=(9, 5))

# create a blue gradient for the bars
cmap = plt.get_cmap('Blues')
colors = [cmap(0.4 + 0.5 * (i / (len(labels) - 1))) for i in range(len(labels))]

bars = ax.barh(labels, idf1_percent, color=colors, edgecolor='black')

# annotate values at right end of bars
for bar, value in zip(bars, idf1_percent):
    ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2,
            f'{value:.2f}', va='center', fontsize=11, color='#222222')

# Title and labels
ax.set_xlabel('IDF1 (%)', fontsize=13)
ax.set_ylabel('w_tertiary', fontsize=13)
ax.set_title('Ablation Study — IDF1 Sensitivity to Association hyperparameters', fontsize=16, pad=12)

# Tight x-limits to match visual emphasis
xmin = min(idf1_percent) - 0.4
xmax = max(idf1_percent) + 0.8
ax.set_xlim(xmin, xmax)

# Light grid and subtle spines like the example
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Footer note similar to example
plt.figtext(0.02, 0.02, 'Note: values from docs/findings.md (w_tertiary sweep).', fontsize=9, color='#666666')

plt.tight_layout()

# Save image next to script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'stage4_idf1_ablation_style.png')
plt.savefig(save_path, dpi=200)
plt.show()
