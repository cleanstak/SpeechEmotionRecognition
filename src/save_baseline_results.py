import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DIRECTORIES
# ============================================================

RESULTS_DIR = r".\results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# BASELINE METRICS
# ============================================================

results = pd.DataFrame({
    "model": ["CNN", "RNN", "LSTM"],
    "parameters": [101636, 63300, 227652],
    "accuracy": [0.5089, 0.2857, 0.3571],
    "macro_precision": [0.5582, 0.0714, 0.3727],
    "macro_recall": [0.5156, 0.2500, 0.4375],
    "macro_f1": [0.4660, 0.1111, 0.3240],
})

results.to_csv(
    os.path.join(RESULTS_DIR, "baseline_results.csv"),
    index=False
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

class_names = [
    "Angry",
    "Happy",
    "Neutral",
    "Sad"
]

confusion_matrices = {

    "CNN": np.array([
        [30,  2,  0,  0],
        [18, 13,  0,  1],
        [ 0,  7,  9,  0],
        [ 4, 11, 12,  5]
    ]),

    "RNN": np.array([
        [0, 0, 0, 32],
        [0, 0, 0, 32],
        [0, 0, 0, 16],
        [0, 0, 0, 32]
    ]),

    "LSTM": np.array([
        [12,  8, 12,  0],
        [ 3, 12, 17,  0],
        [ 0,  0, 16,  0],
        [ 1,  3, 28,  0]
    ])
}


# ============================================================
# CONFUSION MATRIX PLOTS
# ============================================================

for model_name, cm in confusion_matrices.items():

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm)

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Emotion"
    )

    ax.set_ylabel(
        "Actual Emotion"
    )

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(4):
        for j in range(4):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

metrics = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1"
]

metric_labels = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

x = np.arange(len(results["model"]))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 6))

for i, metric in enumerate(metrics):

    values = results[metric]

    ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=metric_labels[i]
    )

ax.set_title(
    "Baseline Speech Emotion Recognition Model Comparison"
)

ax.set_ylabel(
    "Score"
)

ax.set_xlabel(
    "Model"
)

ax.set_xticks(x)
ax.set_xticklabels(results["model"])

ax.set_ylim(0, 1)

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "baseline_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("BASELINE RESULTS SAVED")
print("=" * 70)

print(results.to_string(index=False))

print()
print("Files created:")

for filename in sorted(os.listdir(RESULTS_DIR)):
    print(" -", os.path.join(RESULTS_DIR, filename))

print()