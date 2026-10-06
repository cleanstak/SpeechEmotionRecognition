import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing_enhanced import preprocess_audio_enhanced


class EnhancedSpeechEmotionDataset(Dataset):

    LABEL_MAP = {
        "angry": 0,
        "happy": 1,
        "neutral": 2,
        "sad": 3,
    }

    def __init__(self, metadata_file, split):

        self.data = pd.read_csv(
            metadata_file
        )

        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        file_path = row["file_path"]

        emotion = row["emotion"]

        _, features = preprocess_audio_enhanced(
            file_path
        )

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        label = torch.tensor(
            self.LABEL_MAP[emotion],
            dtype=torch.long
        )

        return features, label