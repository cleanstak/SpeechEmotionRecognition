from pathlib import Path

import torch

from .models import get_model
from .preprocessing import preprocess_audio


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/cnn_best.pth")

CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
]

DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = get_model("cnn")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # The baseline checkpoint contains the model state dict.
    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREDICT EMOTION
# ============================================================

def predict_emotion(
    file_path,
    model=None,
):
    """
    Predict the emotion of a speech audio file.

    Parameters
    ----------
    file_path : str or Path
        Path to WAV/audio file.

    model : optional
        Loaded CNN model. If None, model is loaded automatically.

    Returns
    -------
    emotion : str
        Predicted emotion.

    confidence : float
        Softmax confidence of predicted class.

    probabilities : dict
        Probability for every emotion.
    """

    if model is None:
        model = load_model()

    # --------------------------------------------------------
    # Preprocess audio
    # --------------------------------------------------------

    _, mfcc = preprocess_audio(
        str(file_path)
    )

    # Original MFCC:
    # (40, 251)
    #
    # CNN expects:
    # (batch, 251, 40)

    features = torch.tensor(
        mfcc.T,
        dtype=torch.float32,
    )

    features = features.unsqueeze(0)

    features = features.to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(features)

        probabilities_tensor = torch.softmax(
            outputs,
            dim=1,
        )[0]

    predicted_index = int(
        torch.argmax(
            probabilities_tensor
        ).item()
    )

    predicted_emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities_tensor[
            predicted_index
        ].item()
    )

    probabilities = {
        CLASS_NAMES[i]:
        float(
            probabilities_tensor[i].item()
        )
        for i in range(len(CLASS_NAMES))
    }

    return (
        predicted_emotion,
        confidence,
        probabilities,
    )