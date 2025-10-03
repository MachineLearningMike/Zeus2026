import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from Zeus_CNN_LSTM import Zeus2
# -----------------------------
# Training parameters
# -----------------------------
batch_size = 1
C, H, W = 4, 720, 1440
n, m = 24, 24         # input sequence length n, prediction length m
num_epochs = 3
learning_rate = 1e-3
sampling_prob = 0.8    # scheduled sampling probability

# -----------------------------
# Create model
# -----------------------------
model = Zeus2(input_shape=(C, H, W),
              cnn_hidden_channels=[C, C],
              lstm_layers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Dummy dataset (list of sequences)
# -----------------------------
def generate_batch(batch_size, n, m, C, H, W, alpha=0.1):
    """
    Generate a batch of sequences where each frame is sin(alpha * t)
    Returns a list of length (n + m), each tensor shape (B, C, H, W)
    """
    seq = []
    for t in range(n + m):
        # compute scalar value sin(alpha * t)
        val = math.sin(alpha * t)
        # create a tensor of shape (B, C, H, W) filled with this value
        frame = torch.full((batch_size, C, H, W), val, device=device)
        seq.append(frame)
    return seq

# -----------------------------
# Training loop
# -----------------------------

if __name__ == "__main__":

    for epoch in range(num_epochs):
        # Example: one batch per epoch (replace with DataLoader for real data)
        x_seq = generate_batch(batch_size, n, m, C, H, W)

        optimizer.zero_grad()
        preds = model.forward(x_seq, m=m, sampling_prob=sampling_prob)

        # Compute loss (compare prediction to ground truth)
        loss = 0.0
        for i, pred in enumerate(preds):
            target = x_seq[n + i]  # ground-truth frame
            loss += criterion(pred, target)
        loss /= len(preds)

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
