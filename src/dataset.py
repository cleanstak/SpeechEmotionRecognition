import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import preprocess_audio


class SpeechEmotionDataset(Dataset):
    """
    PyTorch Dataset for speech emotion recognition.
    """

    LABEL_MAP = {
        "angry": 0,
        "happy": 1,
        "neutral": 2,
        "sad": 3,
    }

    def __init__(self, metadata_file, split):
        self.data = pd.read_csv(metadata_file)

        # Keep only the requested split
        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        file_path = row["file_path"]
        emotion = row["emotion"]

        # Preprocess audio
        _, mfcc = preprocess_audio(file_path)

        # Convert:
        # (40, 251)
        # to:
        # (251, 40)
        mfcc = mfcc.T

        # Convert to PyTorch tensor
        features = torch.tensor(
            mfcc,
            dtype=torch.float32
        )

        label = self.LABEL_MAP[emotion]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return features, label