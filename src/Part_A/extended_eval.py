import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

os.makedirs("../assets/evaluation", exist_ok=True)

models = []

# Load Gen 1
with open("../assets/generation_1/results.json", "r") as f:
    gen1 = json.load(f)
    models.append({"Model": "DNN Baseline", "Recall": gen1["recall"], "F1": gen1["f1_score"], "Time": gen1["training_time_adam_sec"]})

# Load Gen 2
with open("../assets/generation_2/results.json", "r") as f:
    gen2 = json.load(f)
    for key in ["LSTM", "GRU", "BiLSTM"]:
        models.append({"Model": key, "Recall": gen2[key]["recall"], "F1": gen2[key]["f1"], "Time": gen2[key]["train_time_sec"]})

# Load Gen 3
with open("../assets/generation_3/results.json", "r") as f:
    gen3 = json.load(f)
    for key in ["FrozenBase", "FullFineTune"]:
        models.append({"Model": gen3[key]["model"], "Recall": gen3[key]["recall"], "F1": gen3[key]["f1"], "Time": gen3[key]["train_time_sec"]})

names = [m["Model"] for m in models]
recalls = [m["Recall"] for m in models]
f1s = [m["F1"] for m in models]
times = [m["Time"] for m in models]

sns.set_style("whitegrid")

# 1. Trade-off Scatter Plot (Time vs Recall)
plt.figure(figsize=(10, 6))
# Colors based on generation
colors = ['red'] + ['orange']*3 + ['green']*2
plt.scatter(times, recalls, s=200, c=colors, edgecolors='black', alpha=0.8)

for i, txt in enumerate(names):
    plt.annotate(txt, (times[i], recalls[i]), xytext=(10, -5), textcoords='offset points', fontsize=11, fontweight='bold')

plt.xscale('log')
plt.xlabel('Training Time in Seconds (Log Scale)', fontsize=12)
plt.ylabel('Recall (Sensitivity)', fontsize=12)
plt.title('Performance Trade-off: Training Time vs. Recall', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("../assets/evaluation/tradeoff_plot.png", dpi=300)
print("Saved tradeoff_plot.png")

# 2. F1 Score Comparison Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(names, f1s, color=colors, edgecolor='black')
plt.ylabel('F1 Score', fontsize=12)
plt.title('F1 Score Comparison Across Generations', fontsize=14, fontweight='bold')
plt.xticks(rotation=25, ha='right')

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("../assets/evaluation/f1_comparison.png", dpi=300)
print("Saved f1_comparison.png")
