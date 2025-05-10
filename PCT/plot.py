import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Set the checkpoints directory
checkpoint_root = "checkpoints"

# Experiment types and prefixes
exp_prefixes = ["test_pct_", "test_pctxyz_", "test_pctgrid_", "test_pctrope_"]
results = {
    "test_pct": [],
    "test_pctxyz": [],
    "test_pctgrid": [],
    "test_pctrope": []
}

# Load results
for folder in os.listdir(checkpoint_root):
    for prefix in exp_prefixes:
        if folder.startswith(prefix):
            if "pctgrid" in folder:
                exp_type = "test_pctgrid"
            elif "pctxyz" in folder:
                exp_type = "test_pctxyz"
            elif "pctrope" in folder:
                exp_type = "test_pctrope"
            else:
                exp_type = "test_pct"

            try:
                num_points = int(folder.split('_')[-1])
            except ValueError:
                continue

            log_path = os.path.join(checkpoint_root, folder, "run.log")
            if not os.path.isfile(log_path):
                continue

            with open(log_path, "r") as f:
                content = f.read()
                match = re.search(r"test acc:\s*([0-9.]+)", content)
                if match:
                    acc = float(match.group(1))
                    results[exp_type].append((num_points, acc))
            break

# Sort results by number of input points
for key in results:
    results[key].sort()

# Labels and colors for plot
labels = {
    "test_pct": "pct (no PE)",
    "test_pctxyz": "pct-xyz",
    "test_pctgrid": "pct-grid",
    "test_pctrope": "pct-rope-axial"
}

colors = {
    "test_pct": "#1f77b4",   # blue
    "test_pctxyz": "#2ca02c", # green
    "test_pctgrid": "#ff7f0e", # orange
    "test_pctrope": "#d62728"  # red
}

# Plot
plt.figure(figsize=(10, 6))
for key, data in results.items():
    if data:
        x, y = zip(*data)
        plt.plot(x, y, marker='o', label=labels[key], color=colors[key], linewidth=2)

# Reference line at 512 input points
plt.axvline(x=512, color='gold', linestyle=':', linewidth=2, label="input size: 512 Points")

plt.xlabel("Number of Points")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Number of Input Points (PCT Variants)")
plt.grid(True)
plt.legend()
plt.xticks(range(256, 2048 + 1, 128))
plt.tight_layout()
plt.savefig("test_accuracy_vs_points.png", dpi=300)
print("Figure saved as test_accuracy_vs_points.png")

# Generate Markdown table
all_points = sorted(set(pt for values in results.values() for pt, _ in values))
table = pd.DataFrame(index=all_points)

for key, data in results.items():
    label = labels[key]
    for pt, acc in data:
        table.at[pt, label] = acc

# Print Markdown table
print("\n### 📊 Accuracy Table (Markdown Format)")
print(table.to_markdown())