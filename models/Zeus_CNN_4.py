import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import random

# -----------------------------
# Encoder & Decoder
# -----------------------------

class CnnPair(nn.Module):
    def __init__(self, h_index, channels, W, max_kernel=65):
        super().__init__()
        alpha = h_index / W / 2 * 180 - 90  # assumption: W/2 = H
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
        self.positional_codes = nn.Parameter(torch.empty(C, H, W).uniform_(-math.sqrt(3), math.sqrt(3)))        # new over v.3

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        slices = [cnnPair(x[:, :, h_index, :]) for h_index, cnnPair in enumerate(self.cnnPairs)]    # [(B, C, W)]
        x = torch.stack(slices, dim=2)  # (B, C, H, W)
        x = x + self.positional_codes
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
        for out_channels in hidden_channels_list:       # Caller will ensure hidden_channels_list[-1] == C
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
            x = F.tanh(x)

        return x


# -----------------------------
# ZeusCNN model
# -----------------------------
class ZeusCNN(nn.Module):
    def __init__(self, C, H, W, cnn_hidden_channels, n_frames=5, kernel_size=3, device='cpu'):
        """
        Past n frames are flattened into channels (n*C) and fed once to encoder + CNN.
        """
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.n_frames = n_frames
        self.device = device

        # Encoder/Decoder always work per-frame
        self.encoder = Encoder(C, H, W).to(device)
        self.decoder = Decoder(cnn_hidden_channels[-1], C, H, W).to(device)

        in_channels = n_frames * C    # stacked frames
        self.cnn = Convolutions(in_channels, cnn_hidden_channels, kernel_size=kernel_size).to(device)
        self.sampling_prob = 0.0
    
    def update_sampling_prob(self, step, total_steps):
        self.sampling_prob = max(min(step / total_steps, 1.0), 0.0)

    def forward(self, x, m_frames, y=None):  # y_seq != None signals training
        """
        Auto-regressive prediction
        x_seq: (B, n, C, H, W)
        """        
        B, n_frames, C, H, W = x.shape
        assert n_frames == self.n_frames, f"Expected {self.n_frames} input frames, got {n_frames}"

        x = x.to(self.device)
        if y is not None:
            assert m_frames == y.shape[1]
            y = y.to(self.device)

        x_enc = self.encoder(x.view(B * n_frames, C, H, W)).view(B, n_frames, C, H, W)

        preds = []

        # Initial window of encoded frames
        x_window = x_enc.reshape(B, n_frames*C, H, W)

        for t in range(m_frames):
            y_pred = self.cnn(x_window)     # From (B, n_frames * C, H, W) to (B, C, H, W)  =========== # This is why the last of cnn_hidden_channels should C.
            y_pred = self.decoder(y_pred)
            preds.append(y_pred)

            if t < m_frames - 1:
                if y is not None and random.random() > self.sampling_prob:  # self.sampling_prob rises as steps go, giving less chance to y.
                    y_next = self.encoder(y[:, t])
                else:
                    y_next = y_pred
                                        
                x_window = torch.cat([x_window[:, C:], y_next], dim=1)
        
        preds = torch.stack(preds, dim=1)   # (B, m_frames, C, H, W)
        preds = self.decoder(preds.view(B * m_frames, C, H, W)).view(B, m_frames, C, H, W)

        return preds

    def sigle_GPU_forward_loss_backward(self, data, target, loss_fn, unnormal_fn, batch_diff, era5_data_vars):
        
        pred = self.forward(data, target.shape[1], target)

        target = unnormal_fn(target)
        pred = unnormal_fn(pred)
        lossAndMetrics = loss_fn(target, pred, batch_diff, era5_data_vars)     
        loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
        loss.backward()             # grads for last_input (i.e. act[n-2]) -> last_module -> act[n-1] -> loss_fn -> loss.
        
        grads = [p.grad.detach().clone() for p in self.parameters()]
        
        return grads, lossAndMetrics

# -----------------------------
# Sinusoidal dataset
# -----------------------------

from torch.utils.data import Dataset, DataLoader
class TemperatureGlobeDataset(Dataset):
    def __init__(self, B, C, H, W, n, m, length=100):
        """
        Simulated temperature over the globe:
        - B: batch size
        - C: number of channels (e.g., different "variables")
        - H: number of latitude points
        - W: number of longitude points
        - n: number of input frames
        - m: number of target frames
        - length: number of sequences
        """
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.n = n
        self.m = m
        self.length = length

        # Create latitude and longitude grids in radians
        lat = torch.linspace(-math.pi/2, math.pi/2, H)   # -90° to +90°
        lon = torch.linspace(-math.pi, math.pi, W)       # -180° to +180°
        self.lat_grid, self.lon_grid = torch.meshgrid(lat, lon, indexing='ij')
        self.obliquity = math.pi / 180 * 23.44

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq_len = self.n + self.m
        x_seq = torch.zeros(self.C, seq_len, self.H, self.W)  # (C, seq_len, H, W)

        for t in range(seq_len):
            temp_pattern = (torch.cos(self.lat_grid + self.obliquity) + 1.5) * (torch.cos(self.lon_grid + t * math.pi/12) + 1.5)
            for c in range(self.C):
                x_seq[c, t, :, :] = temp_pattern * math.sin(c + 1)

        return x_seq.permute(1, 0, 2, 3)  # (seq_len, C, H, W)


# -------------------------------
# Training loop
# -------------------------------
steps = 0
transient_steps = 500

def train_epoch(model, dataloader, optimizer, criterion, x_frames, y_frames, device='cpu', mean=0.0, std=1.0):
    global steps, transient_steps
    
    model.train()
    total_loss = 0.0
    
    def normalize(x):
        return (x - mean.to(x.device)) / std.to(x.device)
    def denormalize(x):
        return x * std.to(x.device) + mean.to(x.device)

    for frames in dataloader:
        frames = frames.to(device)  # (B, n+m, C, H, W)
        optimizer.zero_grad()
        
        # Only feed the first n frames as input
        x_input = frames[:, :x_frames]  # (B, n, C, H, W)
        target_org = frames[:, x_frames:]           # (B, m, C, H, W)
        steps += 1
        model.update_sampling_prob(steps, transient_steps)
        
        x_input = normalize(x_input)
        target = normalize(target_org)
        preds = model.forward(x_input, y_frames, y=target)  # list of m predictions (B, C, H, W)

        preds = denormalize(preds)
        loss = criterion(preds, target_org)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# -------------------------------
# Example usage
# -------------------------------
class LambdaDataset(Dataset):
    def __init__(self, base_dataset, fn):
        self.base_dataset = base_dataset
        self.fn = fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x = self.base_dataset[idx]
        return self.fn(x)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -----------------------------
    # DataLoader
    # -----------------------------
    factor10 = 8  #=======================
    B, C, H, W = 1, 4, 72 * factor10, 144 * factor10
    n, m = 24, 24
    epochs = 500
    length = 10
    
    cnn_hidden_channels=[n*C, C] #=========================
    
    max_radius = 1000 # km
    kernel_size = round(max_radius*72*factor10/6400/3.1416/len(cnn_hidden_channels)) // 2 * 2 + 1
    kernel_size = 9     #============== Mike!!! temporary to reproduce past experiments.
    # if kernel_size % 2 == 0: kernel_size += 1
    
    print('kernel_size, max_radius: ', kernel_size, 6400 * 3.1416 / 72 / factor10 * kernel_size * len(cnn_hidden_channels) )
    dataset = TemperatureGlobeDataset(B, C, H, W, n, m, length=length)
    
    x = dataset[0][0]    # (C, H, W)
    mean = x.mean(dim=[1,2], keepdim=True)
    std = x.std(dim=[1,2], keepdim=True)
    dataset = LambdaDataset(dataset, lambda x: ( ((x-mean)/std*30 + 10) ))
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False)
   
    x = dataset[0][0]    # (C, H, W)
    mean = x.mean(dim=[1,2], keepdim=True)
    std = x.std(dim=[1,2], keepdim=True)
   
    # -----------------------------
    # Model and training
    # -----------------------------
    lr = 1e-3   #=======================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ZeusCNN(C=C, H=H, W=W, n_frames=n, cnn_hidden_channels=cnn_hidden_channels, kernel_size=kernel_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def criterion(pred, target):
        return ((target.to(pred.device) - pred) ** 2).mean() ** 0.5

    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, n, m, device=device, mean=mean, std=std)
        print(f"Epoch {epoch+1}, loss: {loss:.6f}")

