from pathlib import Path
import pandas as pd


# ============================================================
# RAVDESS METADATA CREATION
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Output directory
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# Create metadata directory if it does not exist
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# RAVDESS emotion codes
EMOTION_MAP = {
    1: "neutral",
    3: "happy",
    4: "sad",
    5: "angry",
}


records = []

# Find every WAV file
audio_files = sorted(DATA_DIR.rglob("*.wav"))

print(f"Total WAV files found: {len(audio_files)}")


for audio_path in audio_files:

    # Example filename:
    # 03-01-03-01-01-01-07.wav

    filename = audio_path.stem

    parts = filename.split("-")

    # RAVDESS filenames contain 7 components
    if len(parts) != 7:
        print(f"Skipping invalid filename: {audio_path.name}")
        continue

    try:
        modality = int(parts[0])
        channel = int(parts[1])
        emotion_code = int(parts[2])
        intensity = int(parts[3])
        statement = int(parts[4])
        repetition = int(parts[5])
        actor_id = int(parts[6])
    except ValueError:
        print(f"Skipping invalid filename: {audio_path.name}")
        continue

    # Keep only our four target emotions
    if emotion_code not in EMOTION_MAP:
        continue

    emotion = EMOTION_MAP[emotion_code]

    records.append(
        {
            "file_path": str(audio_path.relative_to(PROJECT_ROOT)),
            "actor_id": actor_id,
            "emotion": emotion,
            "emotion_code": emotion_code,
        }
    )


# Create DataFrame
df = pd.DataFrame(records)

# Sort by actor and emotion
df = df.sort_values(
    by=["actor_id", "emotion", "file_path"]
).reset_index(drop=True)


# Save metadata
output_file = METADATA_DIR / "ravdess_metadata.csv"

df.to_csv(output_file, index=False)


print("\nMetadata creation complete.")
print(f"Total selected audio files: {len(df)}")
print(f"Metadata saved to: {output_file}")

print("\nEmotion distribution:")
print(df["emotion"].value_counts())

print("\nActor distribution:")
print(df["actor_id"].value_counts().sort_index())

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))