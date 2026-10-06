from src.dataset_enhanced import EnhancedSpeechEmotionDataset


METADATA_FILE = r".\data\metadata\dataset_split.csv"


dataset = EnhancedSpeechEmotionDataset(
    METADATA_FILE,
    "train"
)

features, label = dataset[0]

print("Number of training samples:", len(dataset))
print("Feature shape:", features.shape)
print("Feature dtype:", features.dtype)
print("Label:", label.item())
print("Label dtype:", label.dtype)

assert features.shape == (251, 120)

print()
print("Enhanced feature test passed.")