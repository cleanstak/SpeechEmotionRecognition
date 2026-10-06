import numpy as np
import librosa


# =========================
# Audio configuration
# =========================

SAMPLE_RATE = 16000
DURATION = 4.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256


def load_audio(file_path):
    """
    Load an audio file as mono at the target sample rate.
    """

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return audio


def normalize_audio(audio):
    """
    Normalize audio amplitude to approximately [-1, 1].
    """

    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    return audio


def fix_audio_length(audio):
    """
    Trim or zero-pad audio to exactly 4 seconds.
    """

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    elif len(audio) < MAX_SAMPLES:
        padding = MAX_SAMPLES - len(audio)
        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    return audio


def extract_mfcc(audio):
    """
    Extract MFCC features from the audio signal.
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return mfcc


def preprocess_audio(file_path):
    """
    Complete preprocessing pipeline.

    Returns:
        audio: fixed-length normalized waveform
        mfcc: MFCC feature matrix
    """

    # 1. Load audio
    audio = load_audio(file_path)

    # 2. Normalize amplitude
    audio = normalize_audio(audio)

    # 3. Fix duration
    audio = fix_audio_length(audio)

    # 4. Extract MFCC
    mfcc = extract_mfcc(audio)

    return audio, mfcc