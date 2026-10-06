from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_audio


LABEL_MAP = {
    "angry": 0,
    "happy": 1,
    "neutral": 2,
    "sad": 3,
}


class StandardizedSpeechEmotionDataset(Dataset):
    """
    RAVDESS speech emotion dataset using MFCC features.

    The dataset uses the existing speaker-independent split
    stored in data/metadata/dataset_split.csv.

    MFCC features:
        Original shape: (40, 251)
        Returned shape: (251, 40)

    If mean/std are supplied, features are standardized using
    statistics calculated from the TRAINING SET ONLY.
    """

    def __init__(
        self,
        metadata_file,
        split,
        mean=None,
        std=None,
    ):
        self.metadata_file = Path(metadata_file)
        self.split = split

        df = pd.read_csv(self.metadata_file)

        # Keep only the requested split
        df = df[df["split"] == split].reset_index(drop=True)

        self.samples = df

        self.mean = mean
        self.std = std

        print(
            f"Loaded {split} dataset: "
            f"{len(self.samples)} samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        row = self.samples.iloc[idx]

        file_path = Path(row["file_path"])
        emotion = row["emotion"]

        # Handle paths consistently from project root
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path

        _, mfcc = preprocess_audio(
            str(file_path)
        )

        # Original:
        # (40, 251)
        #
        # Required model input:
        # (251, 40)
        mfcc = mfcc.T.astype(np.float32)

        # Apply training-set standardization
        if self.mean is not None and self.std is not None:

            mfcc = (
                mfcc - self.mean
            ) / (
                self.std + 1e-8
            )

        features = torch.tensor(
            mfcc,
            dtype=torch.float32,
        )

        label = torch.tensor(
            LABEL_MAP[emotion],
            dtype=torch.long,
        )

        return features, label


def calculate_training_statistics(dataset):
    """
    Calculate feature-wise mean and standard deviation
    using ONLY the training dataset.

    No validation or test samples are used.
    """

    all_features = []

    print(
        "\nCalculating MFCC statistics "
        "from TRAINING SET ONLY..."
    )

    for i in range(len(dataset)):

        row = dataset.samples.iloc[i]

        file_path = Path(row["file_path"])

        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path

        _, mfcc = preprocess_audio(
            str(file_path)
        )

        # (40, 251) -> (251, 40)
        mfcc = mfcc.T.astype(np.float32)

        all_features.append(mfcc)

    # Combine all time frames from all training utterances
    all_features = np.concatenate(
        all_features,
        axis=0,
    )

    # Feature-wise statistics
    mean = all_features.mean(
        axis=0
    ).astype(np.float32)

    std = all_features.std(
        axis=0
    ).astype(np.float32)

    # Avoid division by zero
    std[std < 1e-8] = 1.0

    return mean, std