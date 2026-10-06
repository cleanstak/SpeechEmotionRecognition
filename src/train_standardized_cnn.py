from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset_standardized import (
    StandardizedSpeechEmotionDataset,
    calculate_training_statistics,
)

from .models import get_model


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 6

METADATA_FILE = Path(
    "data/metadata/dataset_split.csv"
)

MODEL_DIR = Path("models")

MODEL_PATH = (
    MODEL_DIR /
    "cnn_standardized_best.pth"
)

STATS_PATH = (
    MODEL_DIR /
    "mfcc_standardization_stats.npz"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("EXPERIMENT 3: STANDARDIZED MFCC + CNN")
print("=" * 60)

print(f"Device: {device}")
print("Model: CNN")
print(
    "Feature set: MFCC + training-set standardization"
)


# ============================================================
# LOAD BASE TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

train_base = StandardizedSpeechEmotionDataset(
    METADATA_FILE,
    split="train",
)

print(
    f"Training samples: {len(train_base)}"
)


# ============================================================
# CALCULATE TRAINING STATISTICS
# ============================================================

mean, std = calculate_training_statistics(
    train_base
)

print("\nTraining-set statistics calculated.")

print(f"Mean shape: {mean.shape}")
print(f"Std shape:  {std.shape}")

print(
    f"Mean range: "
    f"{mean.min():.4f} to {mean.max():.4f}"
)

print(
    f"Std range:  "
    f"{std.min():.4f} to {std.max():.4f}"
)


# ============================================================
# SAVE STATISTICS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez(
    STATS_PATH,
    mean=mean,
    std=std,
)

print(
    f"\nNormalization statistics saved to:"
    f"\n{STATS_PATH}"
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = StandardizedSpeechEmotionDataset(
    METADATA_FILE,
    split="train",
    mean=mean,
    std=std,
)

val_dataset = StandardizedSpeechEmotionDataset(
    METADATA_FILE,
    split="validation",
    mean=mean,
    std=std,
)

test_dataset = StandardizedSpeechEmotionDataset(
    METADATA_FILE,
    split="test",
    mean=mean,
    std=std,
)


print("\nDataset sizes:")
print(f"Train:      {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test:       {len(test_dataset)}")


# ============================================================
# VERIFY FEATURE SHAPE
# ============================================================

sample_features, sample_label = train_dataset[0]

print("\nFeature verification:")
print(
    f"Feature shape: "
    f"{sample_features.shape}"
)

print(
    f"Feature dtype: "
    f"{sample_features.dtype}"
)

print(f"Label: {sample_label}")

print(
    f"Label dtype: "
    f"{sample_label.dtype}"
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

model = get_model("cnn")
model = model.to(device)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"\nTrainable parameters: "
    f"{num_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

class_weights = torch.tensor(
    [1.0, 1.0, 2.0, 1.0],
    dtype=torch.float32,
    device=device,
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0
epochs_without_improvement = 0

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(
    1,
    EPOCHS + 1,
):

    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
        )
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
    )

    scheduler.step(
        val_accuracy
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "best_val_accuracy":
                    best_val_accuracy,

                "feature_set":
                    "MFCC + training-set standardization",

                "mean": mean,

                "std": std,
            },
            MODEL_PATH,
        )

        epochs_without_improvement = 0

        print(
            f"  --> Best model saved "
            f"(Val Acc: "
            f"{best_val_accuracy:.4f})"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"\nEarly stopping at "
            f"epoch {epoch}."
        )

        break


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Model saved to:"
    f"\n{MODEL_PATH}"
)

print(
    f"Normalization statistics saved to:"
    f"\n{STATS_PATH}"
)