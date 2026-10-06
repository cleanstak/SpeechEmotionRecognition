import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.dataset_enhanced import EnhancedSpeechEmotionDataset
from src.models import get_model


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = r".\data\metadata\dataset_split.csv"

MODEL_PATH = r".\models\cnn_enhanced_best.pth"

BATCH_SIZE = 16

CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Model: ENHANCED CNN")
print("Feature set: MFCC + Delta + Delta-Delta")
print("Model path:", MODEL_PATH)


# ============================================================
# TEST DATA
# ============================================================

test_dataset = EnhancedSpeechEmotionDataset(
    METADATA_FILE,
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# MODEL
# ============================================================

model = get_model("cnn")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
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
        ).cpu()

        all_predictions.extend(
            predictions.tolist()
        )

        all_labels.extend(
            labels.tolist()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("ENHANCED CNN TEST RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print()
print("Rows = Actual emotion")
print("Columns = Predicted emotion")

print()
print("Class order:")
print("0 = angry")
print("1 = happy")
print("2 = neutral")
print("3 = sad")