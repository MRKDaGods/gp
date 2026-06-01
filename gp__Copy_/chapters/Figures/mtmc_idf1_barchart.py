
import os
import matplotlib.pyplot as plt

# IDF1 values (as percentages)
labels = [
    'Ours (2026)',
    'AIC22 1st',
    'AIC22 2nd',
    'AIC21 1st'
]
values = [
    77.936,   # Ours (0.77936 * 100)
    84.86,    # AIC22 1st (0.8486 * 100)
    84.37,    # AIC22 2nd
    84.10     # AIC21 1st
]


# Use a single blue color for all bars
blue = '#4C72B0'
plt.figure(figsize=(7, 5))
bars = plt.bar(labels, values, color=blue, edgecolor='black')

# Annotate values on top of bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{value:.2f}%', ha='center', va='bottom', fontsize=12)

plt.ylabel('IDF1 (%)', fontsize=13)
plt.title('MTMC IDF1 Comparison: Ours vs. AIC Challenge Winners', fontsize=14)
plt.ylim(75, 87)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
# Save image in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "mtmc_idf1_barchart.png")
plt.savefig(save_path, dpi=200)
plt.show()
