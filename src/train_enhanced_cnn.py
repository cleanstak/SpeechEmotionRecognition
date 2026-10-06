import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.dataset_enhanced import EnhancedSpeechEmotionDataset
from src.models import get_model


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = r".\data\metadata\dataset_split.csv"

MODEL_NAME = "cnn"
MODEL_PATH = r".\models\cnn_enhanced_best.pth"

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 6

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Model:", MODEL_NAME.upper())
print("Feature set: MFCC + Delta + Delta-Delta")


# ============================================================
# DATASETS
# ============================================================

train_dataset = EnhancedSpeechEmotionDataset(
    METADATA_FILE,
    "train"
)

validation_dataset = EnhancedSpeechEmotionDataset(
    METADATA_FILE,
    "validation"
)

test_dataset = EnhancedSpeechEmotionDataset(
    METADATA_FILE,
    "test"
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = get_model(MODEL_NAME)
model = model.to(device)

print(
    "Number of parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# LOSS
# ============================================================

class_weights = torch.tensor(
    [1.0, 1.0, 2.0, 1.0],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

best_validation_accuracy = 0.0
epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            features.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in validation_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += (
                loss.item() *
                features.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    validation_loss = validation_loss / total
    validation_accuracy = correct / total


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        validation_accuracy
    )

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Acc: {validation_accuracy:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    MODEL_NAME,

                "feature_type":
                    "mfcc_delta_delta",

                "validation_accuracy":
                    best_validation_accuracy,

                "epoch":
                    epoch + 1
            },
            MODEL_PATH
        )

        print(
            f"  -> Best model saved: "
            f"{MODEL_PATH}"
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("ENHANCED CNN TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.4f}"
)

print(
    f"Best model: {MODEL_PATH}"
)