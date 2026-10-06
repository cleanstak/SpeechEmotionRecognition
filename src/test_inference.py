from pathlib import Path

from .inference import predict_emotion


# Use an existing RAVDESS test-set file
AUDIO_FILE = Path(
    "data/raw/Actor_01/"
    "03-01-05-01-01-01-01.wav"
)


print("=" * 60)
print("FINAL CNN INFERENCE TEST")
print("=" * 60)

print(f"Audio file: {AUDIO_FILE}")


emotion, confidence, probabilities = (
    predict_emotion(AUDIO_FILE)
)


print("\nPrediction:")
print(
    f"Emotion: {emotion}"
)

print(
    f"Confidence: {confidence:.4f}"
)

print("\nProbabilities:")

for emotion_name, probability in (
    probabilities.items()
):

    print(
        f"  {emotion_name:8s}: "
        f"{probability:.4f}"
    )