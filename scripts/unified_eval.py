import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("../assets/evaluation", exist_ok=True)

# 1. Load Data
models = []

# --- Generation 1 ---
with open("../assets/generation_1/results.json", "r") as f:
    gen1 = json.load(f)
    models.append({
        "Model": "DNN (Baseline)",
        "Accuracy": gen1["accuracy"],
        "Precision": gen1["precision"],
        "Recall": gen1["recall"],
        "F1 Score": gen1["f1_score"],
        "Training Time": f"{gen1['training_time_adam_sec']:.2f}s"
    })

# --- Generation 2 ---
with open("../assets/generation_2/results.json", "r") as f:
    gen2 = json.load(f)
    for key in ["LSTM", "BiLSTM", "GRU"]:
        models.append({
            "Model": key,
            "Accuracy": gen2[key]["accuracy"],
            "Precision": gen2[key]["precision"],
            "Recall": gen2[key]["recall"],
            "F1 Score": gen2[key]["f1"],
            "Training Time": f"{gen2[key]['train_time_sec']:.2f}s"
        })

# --- Generation 3 ---
with open("../assets/generation_3/results.json", "r") as f:
    gen3 = json.load(f)
    for key in ["FrozenBase", "FullFineTune"]:
        models.append({
            "Model": gen3[key]["model"],
            "Accuracy": gen3[key]["accuracy"],
            "Precision": gen3[key]["precision"],
            "Recall": gen3[key]["recall"],
            "F1 Score": gen3[key]["f1"],
            "Training Time": f"{gen3[key]['train_time_sec']:.2f}s"
        })

# 2. Generate LaTeX Table
df = pd.DataFrame(models)

latex_table = df.to_latex(index=False, float_format="%.4f")
with open("../assets/evaluation/unified_metrics.tex", "w") as f:
    f.write(latex_table)
print("Saved unified_metrics.tex")

# 3. Reconstruct Confusion Matrices
def reconstruct_cm(accuracy, precision, recall, is_gen3=False, gen3_data=None):
    if is_gen3:
        return np.array([
            [gen3_data['tn'], gen3_data['fp']],
            [gen3_data['fn'], gen3_data['tp']]
        ])
    
    # For Gen 1 and Gen 2, test support: Positives = 1410, Negatives = 16933
    P = 1410
    N = 16933
    
    tp = recall * P
    fn = P - tp
    # FP from precision
    fp = (tp / precision) - tp if precision > 0 else 0
    tn = N - fp
    
    return np.array([
        [int(tn), int(fp)],
        [int(fn), int(tp)]
    ])

cms = []
# Gen 1
cms.append(reconstruct_cm(gen1["accuracy"], gen1["precision"], gen1["recall"]))
# Gen 2
for key in ["LSTM", "BiLSTM", "GRU"]:
    cms.append(reconstruct_cm(gen2[key]["accuracy"], gen2[key]["precision"], gen2[key]["recall"]))
# Gen 3
cms.append(reconstruct_cm(0, 0, 0, is_gen3=True, gen3_data=gen3["FrozenBase"]))
cms.append(reconstruct_cm(0, 0, 0, is_gen3=True, gen3_data=gen3["FullFineTune"]))

model_names = [m["Model"] for m in models]

# 4. Plot Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (cm, name) in enumerate(zip(cms, model_names)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
plt.savefig("../assets/evaluation/unified_cm.png", dpi=300)
print("Saved unified_cm.png")
