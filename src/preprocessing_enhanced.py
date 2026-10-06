import numpy as np
import librosa


SAMPLE_RATE = 16000
DURATION = 4
TARGET_SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256


def load_and_normalize_audio(file_path):
    """
    Load an audio file, resample to 16 kHz, and normalize amplitude.
    """

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Fix the duration to exactly 4 seconds
    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]

    elif len(audio) < TARGET_SAMPLES:
        padding = TARGET_SAMPLES - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    # Per-utterance amplitude normalization
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude > 0:
        audio = audio / max_amplitude

    return audio


def preprocess_audio_enhanced(file_path):
    """
    Extract MFCC, delta MFCC and delta-delta MFCC.

    Output shape:
        (251, 120)

    40 MFCC
    + 40 delta MFCC
    + 40 delta-delta MFCC
    = 120 features
    """

    audio = load_and_normalize_audio(file_path)

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # First-order temporal derivative
    # --------------------------------------------------------

    delta_mfcc = librosa.feature.delta(
        mfcc
    )

    # --------------------------------------------------------
    # Second-order temporal derivative
    # --------------------------------------------------------

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    features = np.concatenate(
        [
            mfcc,
            delta_mfcc,
            delta2_mfcc
        ],
        axis=0
    )

    # librosa format:
    # (features, time)
    #
    # Neural network format:
    # (time, features)

    features = features.T

    return audio, features