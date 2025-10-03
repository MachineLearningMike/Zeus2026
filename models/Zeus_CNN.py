import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# -----------------------------
# Encoder & Decoder
# -----------------------------

class CnnPair(nn.Module):
    def __init__(self, h_index, channels, W, max_kernel=65):
        super().__init__()
        alpha = h_index / 720 * 180 - 90
        alpha = math.cos(alpha * math.pi / 180) + 1e-9

        floor = max(1, math.floor(alpha))
        ceil  = max(1, math.ceil(alpha))
        self.floor = min(floor, max_kernel)
        self.ceil  = min(ceil,  max_kernel)

        self.floorW = ceil - alpha
        self.ceilW  = alpha - floor

        self.channels = channels
        self.W = W

        # Conv1d with no padding
        self.floorCnn = nn.Conv1d(channels, channels, self.floor, stride=1, padding=0, bias=True)
        self.ceilCnn  = nn.Conv1d(channels, channels, self.ceil,  stride=1, padding=0, bias=True)

    def forward(self, x):   # (B, C, W)
        # compute padding to keep output same width as input
        pad_floor = (self.floor - 1) // 2
        pad_ceil  = (self.ceil  - 1) // 2

        xFloor = F.pad(x, (pad_floor, self.floor - 1 - pad_floor), mode="circular")
        xFloor = self.floorCnn(xFloor)

        xCeil  = F.pad(x, (pad_ceil, self.ceil - 1 - pad_ceil), mode="circular")
        xCeil  = self.ceilCnn(xCeil)

        x = self.floorW * xFloor + self.ceilW * xCeil
        return x
        
class Encoder(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.cnnPairs = nn.ModuleList([CnnPair(h_index, C, W) for h_index in range(H)])

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        slices = [cnnPair(x[:, :, h_index, :]) for h_index, cnnPair in enumerate(self.cnnPairs)]    # [(B, C, W)]
        x = torch.stack(slices, dim=2)  # (B, C, H, W)
        return x

class Decoder(Encoder):
    def __init__(self, in_C, C, H, W):
        assert in_C == C
        super().__init__(C, H, W)

# -----------------------------
# Convolutions (circular width)
# -----------------------------
class Convolutions(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, kernel_size=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.kernel_sizes = []
        prev_channels = in_channels
        for out_channels in hidden_channels_list:
            self.convs.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=kernel_size, padding=0)
            )
            self.kernel_sizes.append(kernel_size)
            prev_channels = out_channels

    def forward(self, x):
        for conv, k in zip(self.convs, self.kernel_sizes):
            pad_h = k // 2
            pad_w = k // 2

            # circular pad along width (dim=3)
            if pad_w > 0:
                x = torch.cat([x[:, :, :, -pad_w:], x, x[:, :, :, :pad_w]], dim=3)

            # zero pad along height (dim=2)
            if pad_h > 0:
                x = F.pad(x, (0, 0, pad_h, pad_h), mode='constant', value=0)

            x = conv(x)
            x = F.relu(x, inplace=True)
        return x


# -----------------------------
# ZeusCNN model
# -----------------------------
class ZeusCNN(nn.Module):
    def __init__(self, C, H, W, cnn_hidden_channels, n_frames, repeated_encoder=False):
        """
        repeated_encoder:
            - False: past n frames are flattened into channels and fed once to encoder + CNN
            - True: only y_pred at each step is fed through encoder + CNN, previous frames are cached
        """
        super().__init__()
        self.n = n_frames
        self.C, self.H, self.W = C, H, W
        self.repeated_encoder = repeated_encoder

        if not repeated_encoder:
            # Flatten past frames into channels
            in_channels = C * n_frames
        else:
            # CNN sees one frame at a time
            in_channels = C

        self.encoder = Encoder(in_channels, H, W)
        self.cnn = Convolutions(in_channels, cnn_hidden_channels)
        self.decoder = Decoder(cnn_hidden_channels[-1], C, H, W)

    def forward_step(self, x_input):
        """
        Forward a single step through encoder + CNN + decoder
        x_input: (B, C, H, W)
                 if repeated_encoder=False, this may be (B, n*C, H, W)
        """
        x_enc = self.encoder(x_input)
        x_cnn = self.cnn(x_enc)
        y_pred = self.decoder(x_cnn)
        return y_pred

    def forward(self, x_seq, n=None, m=1):
        """
        Auto-regressive prediction
        x_seq: (B, n, C, H, W)
        """
        if n is None:
            n = self.n

        B, n_frames, C, H, W = x_seq.shape
        if n_frames != n:
            raise ValueError(f"x_seq should have n={n} frames, got {n_frames}")

        preds = []

        if not self.repeated_encoder:
            # Flatten all past frames into channels
            x_window = x_seq.reshape(B, n*C, H, W)
            for t in range(m):
                y_pred = self.forward_step(x_window)
                preds.append(y_pred)
                # Slide window: remove oldest channels, append predicted frame
                # Flatten the window again: drop oldest C channels, append new
                if t < m - 1:
                    x_window = torch.cat([x_window[:, C:], y_pred], dim=1)
        else:
            # repeated_encoder=True
            # Initial window: first n frames
            x_window = x_seq.clone()  # (B, n, C, H, W)
            for t in range(m):
                # Only feed the most recent frame to encoder + CNN
                y_pred = self.forward_step(x_window[:, -1])  # (B, C, H, W)
                preds.append(y_pred)
                # Slide window: drop oldest, append predicted frame
                x_window = torch.cat([x_window[:, 1:], y_pred.unsqueeze(1)], dim=1)

        return preds  # list of m predictions (B, C, H, W)


# -----------------------------
# Sinusoidal dataset
# -----------------------------

from torch.utils.data import Dataset, DataLoader
class SinusoidalDataset(Dataset):
    def __init__(self, B, C, H, W, n, m, alpha=0.2, length=100):
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.n = n
        self.m = m
        self.alpha = alpha
        self.length = length  # number of sequences

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq_len = self.n + self.m
        x_seq = torch.zeros(self.C, seq_len, self.H, self.W)  # (C, seq_len, H, W)
        for t in range(seq_len):
            val = math.sin(self.alpha * t)
            x_seq[:, t, :, :] = val * torch.ones(self.C, self.H, self.W)
        # return shape (seq_len, C, H, W) -> we'll permute in collate
        return x_seq.permute(1, 0, 2, 3)  # (seq_len, C, H, W)

# -------------------------------
# Training loop
# -------------------------------
def train_epoch(model, dataloader, optimizer, criterion, n, m, device='cpu'):
    model.train()
    total_loss = 0.0

    for x_seq in dataloader:
        x_seq = x_seq.to(device)  # (B, n+m, C, H, W)
        optimizer.zero_grad()

        # Only feed the first n frames as input
        x_input = x_seq[:, :n, :, :, :]  # (B, n, C, H, W)
        preds = model.forward(x_input, m=m)  # list of m predictions (B, C, H, W)

        preds_tensor = torch.stack(preds, dim=1)  # (B, m, C, H, W)
        target = x_seq[:, n:, :, :, :]           # (B, m, C, H, W)
        loss = criterion(preds_tensor, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -----------------------------
    # DataLoader
    # -----------------------------
    B, C, H, W = 1, 4, 720, 1440
    n, m = 5, 3
    dataset = SinusoidalDataset(B, C, H, W, n, m, alpha=0.2, length=10)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)


    # -----------------------------
    # Model and training
    # -----------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ZeusCNN(C=C, H=H, W=W, cnn_hidden_channels=[C, C], n_frames=n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(5):
        loss = train_epoch(model, dataloader, optimizer, criterion, n, m, device)
        print(f"Epoch {epoch+1}, loss: {loss:.6f}")

