import torch

from src.models import CNNModel, RNNModel, LSTMModel


# Simulate one training batch
x = torch.randn(16, 251, 40)

print("Input shape:", x.shape)


# =========================
# CNN
# =========================

cnn = CNNModel()
cnn_output = cnn(x)

print("CNN output shape:", cnn_output.shape)


# =========================
# RNN
# =========================

rnn = RNNModel()
rnn_output = rnn(x)

print("RNN output shape:", rnn_output.shape)


# =========================
# LSTM
# =========================

lstm = LSTMModel()
lstm_output = lstm(x)

print("LSTM output shape:", lstm_output.shape)