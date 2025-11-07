import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import math
from einops import rearrange
import random

from pathlib import Path
import os, sys
from SphericalCodec import SphericalEncoder, SphericalDecoder

sys.path.append(str(Path(__file__).resolve().parent.parent))

# modules.ResNet_custom_1 is 0.131:0.090x faster, giving the same accuracy.
# from modules.ResNet50_Zeus_GroupNorm_365K import ResNetCylindrical
# from modules.ResNet50_Zeus_GroupNorm_24M import ResNetCylindrical
# from modules.ResNet50_Zeus_noStage_GroupNorm_xM import ResNetCylindrical
# from modules.ResNet50_Zeus_noStage_GroupNorm_365K import ResNetCylindrical
# from modules.ResNet50_Zeus_GroupNorm import ResNetCylindrical
from modules.ResNet_CircularBottleneckStack import CircularBottleneckStack

# -----------------------------
# ZeusCNN model
# -----------------------------
class Zeus(nn.Module):
    def __init__(self, pm, device='cpu'):
        """
        Past n frames are flattened into channels (n*C) and fed once to encoder + CNN.
        """       
        super().__init__()
        self.pm = pm

        C, H, W, self.n_frames = len(pm.variables), pm.height, pm.width, pm.nFramesIn
        blocksPerState = pm.blocksPerState
        self.device = device

        # Encoder/Decoder always work per-frame
        self.encoder = SphericalEncoder(C, H, W, pm.ObsorbHDistortion).to(device)
        self.decoder = SphericalDecoder(C, C, H, W, pm.ObsorbHDistortion).to(device)

        in_channels = self.n_frames * C    # stacked frames
        # self.cnn = Convolutions(in_channels, cnn_hidden_channels, kernel_size=kernel_size).to(device)
        print("ZeusResNet being created...")
        # self.cnn = ResNetCylindrical(nVariables=C, nBlocks=36, mid_channels=in_channels, in_channels=in_channels).to(device)    # 3753K params, from ResNet50_Zeus_GroupNorm_3753K
        # self.cnn = ResNetCylindrical(in_channels=pm.T1*C).to(device)  # 365K params, from ResNet50_Zeus_GroupNorm_365K
        # self.cnn = ResNetCylindrical(self.pm).to(device)
        self.cnn = CircularBottleneckStack(channels=pm.T1 * pm.nVariables, num_blocks=pm.nCNNBlocks, bottlenectRate=pm.bottleneckRate, dropout=pm.bottleneck_dropout).to(device)

        num_params = sum(p.numel() for p in self.cnn.parameters())
        print(f"Total parameters: {num_params:,}")

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

        # This block is replaced by its following line.
        # for t in range(m_frames):
        #     y_pred = self.cnn(x_window)     # From (B, n_frames * C, H, W) to (B, C, H, W)  =========== # This is why the last of cnn_hidden_channels should C.
        #     y_pred = self.decoder(y_pred)
        #     preds.append(y_pred)
        #     if t < m_frames - 1:
        #         if y is not None and random.random() > self.sampling_prob:  # self.sampling_prob rises as steps go, giving less chance to y.
        #             y_next = self.encoder(y[:, t])
        #         else:
        #             y_next = y_pred      
        #         x_window = torch.cat([x_window[:, C:], y_next], dim=1)        
        # preds = torch.stack(preds, dim=1)   # (B, m_frames, C, H, W)
        
        # This line replaces the above commented-out block. Faster. Larger loss reduction per hour, though smaller loss reduction per epoch.
        pred = self.cnn(x_window).view(B, n_frames, C, H, W)   # self.cnn: (B, n_frames*C, H, W) -> (B, n_frames*C, H, W)
        
        preds = self.decoder(pred.view(B * m_frames, C, H, W)).view(B, m_frames, C, H, W)

        return preds

    def single_GPU_forward_loss_backward(self, input_unnormalized, target_unnormalized, loss_fn, normalize, unnormalize, batch_diff, era5_data_vars, trainable=True):
        # Do not cast to 16bit for CNN networks.

        # with autocast(device_type=self.device_type, dtype=torch.bfloat16):
        #     with torch.set_grad_enabled(trainable):         

        # assert input.requires_grad == False
        # assert target.requires_grad == False

        input_unnormalized = input_unnormalized.to(self.device)
        target_unnormalized = target_unnormalized.to(self.device)
        
        input_normalized = normalize(input_unnormalized)
        target_normalized = normalize(target_unnormalized)

        context = torch.enable_grad if trainable else torch.no_grad       # commented out for test, as validation loss is weirdly larger than training loss.
        with context():
            pred = self(input_normalized, target_normalized.shape[1])
            lossAndMetrics = loss_fn(target_normalized, pred, batch_diff, era5_data_vars)
            loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
            if trainable:
                loss.backward()
                # grads = [p.grad.detach().clone() for p in self.parameters()]

        if isinstance(lossAndMetrics, (tuple, list)):
            detached_loss_metrics = [x.detach().cpu() if torch.is_tensor(x) else x for x in lossAndMetrics]
        else:
            detached_loss_metrics = lossAndMetrics.detach().cpu()
        
        pred = unnormalize(pred.detach())        # (B, T, C, H, W)
        mae = torch.abs(pred - target_unnormalized).mean(dim=(0, 1, 3, 4)).cpu()    # (C,)
                        
        return detached_loss_metrics, mae

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

    import time
    start = time.time(); local_step = 0
    for frames in dataloader:
        frames = frames.to(device)  # (B, n+m, C, H, W)
        optimizer.zero_grad()
        
        # Only feed the first n frames as input
        x_input = frames[:, :x_frames]  # (B, n, C, H, W)
        target_org = frames[:, x_frames:]           # (B, m, C, H, W)
        steps += 1; local_step += 1
        
        x_input = normalize(x_input)
        target = normalize(target_org)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            s = time.time()
            preds = model(x_input, y_frames, y=target)  # list of m predictions (B, C, H, W)
            e = time.time()
            print("forward pass: {:.3f}".format(e-s))

        preds = denormalize(preds)
        loss = criterion(preds, target_org)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            s = time.time()
            loss.backward()
            e = time.time()
            print("backward pass: {:.3f}".format(e-s))
        optimizer.step()
        total_loss += loss.item()
        e = time.time()
        end = time.time()
        print("steps: {}, loss {:.3f}, avg time(s) {:.3f}".format(steps, loss, (end-start)/local_step))

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
    
    import time
    # import sys
    # from pathlib import Path
    # # Get path to sibling folder
    # sibling_path = Path(__file__).resolve().parent.parent / "models"
    # sys.path.insert(0, str(sibling_path))
    from config import config
    from modules.params import Params
    
    pm = Params(config)
    B, C, H, W, n, m, = pm.batch_size, len(pm.variables), pm.height, pm.width, pm.nFramesIn, pm.nFramesOut
    print("B, C, H, W, n, m", B, C, H, W, n, m)
    
    epochs = 50
    length = 10
    
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
    model = Zeus(pm, device=device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def criterion(pred, target):
        return ((target.to(pred.device) - pred) ** 2).mean() ** 0.5

    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, n, m, device=device, mean=mean, std=std)
        # print(f"Epoch {epoch+1}, loss: {loss:.6f}")

