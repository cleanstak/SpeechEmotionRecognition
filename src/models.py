import torch
import torch.nn as nn


# ============================================================
# 1. CNN MODEL
# ============================================================

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for speech emotion recognition.

    Input:
        (batch, 251, 40)

    Internally converted to:
        (batch, 1, 40, 251)
    """

    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second convolution block
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Third convolution block
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Adaptive pooling avoids depending on exact
            # MFCC dimensions.
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # (batch, 251, 40)
        # → (batch, 1, 40, 251)

        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# 2. VANILLA RNN MODEL
# ============================================================

class RNNModel(nn.Module):
    """
    Vanilla Recurrent Neural Network.

    Input:
        (batch, 251, 40)
    """

    def __init__(
        self,
        input_size=40,
        hidden_size=128,
        num_layers=2,
        num_classes=4
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # RNN output:
        # (batch, time, hidden)

        output, hidden = self.rnn(x)

        # Take the final time step
        x = output[:, -1, :]

        x = self.classifier(x)

        return x


# ============================================================
# 3. LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory Network.

    Input:
        (batch, 251, 40)
    """

    def __init__(
        self,
        input_size=40,
        hidden_size=128,
        num_layers=2,
        num_classes=4
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # LSTM output:
        # (batch, time, hidden)

        output, (hidden, cell) = self.lstm(x)

        # Take the final time step
        x = output[:, -1, :]

        x = self.classifier(x)

        return x


# ============================================================
# Model factory
# ============================================================

def get_model(model_name):

    model_name = model_name.lower()

    if model_name == "cnn":
        return CNNModel()

    elif model_name == "rnn":
        return RNNModel()

    elif model_name == "lstm":
        return LSTMModel()

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose cnn, rnn, or lstm."
        )