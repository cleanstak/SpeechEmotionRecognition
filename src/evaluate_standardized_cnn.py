from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .dataset_standardized import (
    StandardizedSpeechEmotionDataset,
)

from .models import get_model


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = Path(
    "data/metadata/dataset_split.csv"
)

MODEL_PATH = Path(
    "models/cnn_standardized_best.pth"
)

STATS_PATH = Path(
    "models/mfcc_standardization_stats.npz"
)

BATCH_SIZE = 16

CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print(f"Device: {device}")
print("Model: STANDARDIZED CNN")
print(
    "Feature set: MFCC + training-set standardization"
)
print(f"Model path: {MODEL_PATH}")


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

stats = np.load(STATS_PATH)

mean = stats["mean"]
std = stats["std"]

print(
    f"Mean shape: {mean.shape}"
)

print(
    f"Std shape:  {std.shape}"
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = StandardizedSpeechEmotionDataset(
    METADATA_FILE,
    split="test",
    mean=mean,
    std=std,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(
    f"Test samples: {len(test_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

model = get_model("cnn")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        outputs = model(features)

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZED CNN TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"Macro F1:  {macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print(
    "\nRows = Actual emotion"
)

print(
    "Columns = Predicted emotion"
)

print(
    "\nClass order:"
)

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{i} = {name}"
    )