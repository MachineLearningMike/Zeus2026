"""
Icosahedral U-Net (PyTorch + PyTorch Geometric) template
======================================================

This file is a runnable template for training an Icosahedral (mesh) U-Net on
remapped ERA5 data. It's intentionally modular and contains placeholders where
you should plug in your environment specifics (paths, mesh files, data format).

Key assumptions
- You remapped ERA5 lat-lon data to an icosahedral mesh at one or more
  subdivision levels. For each level `L` you have:
    nodes_level_L.npz -> contains arrays: 'coords' (N_L,3) and 'edges' (E_L,2)
    features stored per time sample as numpy arrays: features_YYYYMMDDhh.npz
      each contains 'X' shaped (T_local, C, N_L) or precomputed per-day files.
- You have precomputed coarse/fine mesh correspondences for pooling/unpooling:
    pool_map_L.npz -> 'coarse_idx' array of length N_coarse mapping from
                       fine nodes to coarse node index (for unpooling reverse)

Replace path constants below with your dataset locations.

Requirements
- Python 3.9+
- torch, torchvision
- torch_geometric (PyG) + its dependencies (torch-scatter, torch-sparse, etc.)
- numpy, xarray (optional), zarr (optional)

Notes on design choices for B200 GPU
- Mixed precision training (AMP) is enabled
- Gradient accumulation used to emulate larger batch
- Use GroupNorm/InstanceNorm instead of BatchNorm (small batch sizes)
- We treat time as channels by default (fast) but show hooks for temporal convs

"""

import os
import math
import time
import glob
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# PyG imports
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GCNConv, MessagePassing

# -----------------------------
# Configuration / paths
# -----------------------------
DATA_ROOT = "d:/zeus_data/icosahedral/remapped/icosahedral/data"  # REPLACE
MESH_ROOT = "d:/zeus_data/icosahedral/mesh/levels"                # REPLACE
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Mesh/levels configuration: list of levels from fine to coarse
# e.g. levels = [4, 3, 2, 1] where level 4 is finest
MESH_LEVELS = [4, 3, 2, 1]

# Model / training hyperparams (tune as needed)
NUM_INPUT_VARS = 6
HOURS_IN = 24
HOURS_OUT = 24
INPUT_CHANNELS = NUM_INPUT_VARS * HOURS_IN  # when time-as-channels
OUTPUT_CHANNELS = NUM_INPUT_VARS * HOURS_OUT

LR = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 30
ACCUM_STEPS = 8  # gradient accumulation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Utilities for mesh loading
# -----------------------------

def load_mesh_level(level: int, mesh_root: str = MESH_ROOT) -> Dict:
    """Expect a file nodes_level_{level}.npz containing 'coords' and 'edges'.
    edges shape: (E,2) with 0-based node indices. coords shape: (N,3) or (N,2).
    """
    p = os.path.join(mesh_root, f"nodes_level_{level}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p)
    coords = d['coords']
    edges = d['edges'].astype(np.int64)
    return {
        'level': level,
        'coords': coords,
        'edges': edges,
        'num_nodes': coords.shape[0]
    }


def load_pool_map(fine_level: int, coarse_level: int, mesh_root: str = MESH_ROOT) -> np.ndarray:
    """Load a mapping that maps fine node index -> coarse node index
    File: pool_map_{fine}_{coarse}.npz with array 'coarse_idx' length N_fine.
    """
    p = os.path.join(mesh_root, f"pool_map_{fine_level}_{coarse_level}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p)
    return d['coarse_idx'].astype(np.int64)

# -----------------------------
# Dataset class
# -----------------------------
class IcosahedralERA5Dataset(InMemoryDataset):
    """Assumes per-sample feature files in DATA_ROOT, e.g. sample_YYYYMMDDhh.npz
    Each sample file should contain 'X' with shape (C, N_nodes_fine) representing
    one timeframe (or flattened times as channels). For this template we assume
    each sample is a pair (input: 24h of data flattened into channels, target: 24h next day)
    """
    def __init__(self, root: str, mesh_level: int, split: str = 'train', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.mesh_level = mesh_level
        self.split = split
        self.file_list = sorted(glob.glob(os.path.join(root, split, '*.npz')))
        # We'll load lazily in get() for memory efficiency

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # Each file: 'X_in' shape (C_in, N), 'X_out' shape (C_out, N)
        p = self.file_list[idx]
        d = np.load(p)
        X_in = d['X_in'].astype(np.float32)  # shape (C_in, N)
        X_out = d['X_out'].astype(np.float32)
        # build PyG Data
        # edges for this level
        mesh = load_mesh_level(self.mesh_level)
        edge_index = torch.from_numpy(mesh['edges'].T.copy()).long()
        data = GeometricData(x=torch.from_numpy(X_in.T).float(), y=torch.from_numpy(X_out.T).float(), edge_index=edge_index)
        # x: (N, C_in), y: (N, C_out)
        return data

# -----------------------------
# Graph U-Net building blocks
# -----------------------------
class GraphConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)
        self.conv2 = GCNConv(out_ch, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch) if out_ch>=8 else nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch) if out_ch>=8 else nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.act(x)
        return x

class PoolingLayer:
    """Simple pooling via precomputed mapping fine->coarse.
    We implement pool(x) by summing fine node features to their coarse index.
    """
    def __init__(self, pool_map: np.ndarray, num_coarse: int):
        self.pool_map = torch.from_numpy(pool_map).long()
        self.num_coarse = int(num_coarse)

    def pool(self, x: torch.Tensor):
        # x: (N_fine, C)
        idx = self.pool_map.to(x.device)
        C = x.shape[1]
        out = torch.zeros((self.num_coarse, C), device=x.device, dtype=x.dtype)
        out = out.index_add(0, idx, x)
        # average
        counts = torch.zeros(self.num_coarse, device=x.device)
        counts = counts.index_add(0, idx, torch.ones_like(idx, dtype=x.dtype))
        counts = counts.clamp_min(1.0).unsqueeze(1)
        out = out / counts
        return out

    def unpool(self, x_coarse: torch.Tensor):
        # x_coarse: (N_coarse, C) -> return (N_fine, C)
        idx = self.pool_map.to(x_coarse.device)
        return x_coarse[idx]

# -----------------------------
# Full Icosahedral U-Net model
# -----------------------------
class IcosahedralUNet(nn.Module):
    def __init__(self, mesh_levels: List[int], in_ch: int, out_ch: int, base_ch: int = 64):
        super().__init__()
        self.mesh_levels = mesh_levels
        self.levels = len(mesh_levels)
        # Build per-level meshes and pooling maps
        self.mesh_data = [load_mesh_level(l) for l in mesh_levels]
        # precompute pool maps from level i -> i+1 (fine->coarse)
        self.pool_maps = []
        for i in range(self.levels - 1):
            fine = mesh_levels[i]
            coarse = mesh_levels[i+1]
            pm = load_pool_map(fine, coarse)
            self.pool_maps.append(pm)
        # encoder blocks
        self.enc_blocks = nn.ModuleList()
        self.pools = []
        in_channels = in_ch
        ch = base_ch
        for i in range(self.levels):
            self.enc_blocks.append(GraphConvBlock(in_channels, ch))
            in_channels = ch
            ch = min(ch * 2, 512)
        # bottleneck
        self.bottleneck = GraphConvBlock(in_channels, in_channels)
        # decoder blocks (mirror)
        self.dec_blocks = nn.ModuleList()
        for i in range(self.levels - 1, -1, -1):
            ch_dec = self.enc_blocks[i].conv2.out_channels
            ch_out = ch_dec
            self.dec_blocks.append(GraphConvBlock(ch_dec * 2, ch_out))
        # final projection
        self.final_lin = nn.Linear(self.enc_blocks[0].conv2.out_channels, out_ch)
        # build pooling helper objects
        self.pool_objs = []
        for i, pm in enumerate(self.pool_maps):
            num_coarse = load_mesh_level(mesh_levels[i+1])['num_nodes']
            self.pool_objs.append(PoolingLayer(pm, num_coarse))

    def forward(self, x_in, mesh_edge_indices: List[torch.Tensor]):
        # x_in: (N_fine, C_in) on level 0
        enc_feats = []
        x = x_in
        # Encoder with pooling
        for i in range(self.levels - 1):
            edge_index = mesh_edge_indices[i]
            x = self.enc_blocks[i](x, edge_index)
            enc_feats.append(x)
            # pool to next coarse level
            x = self.pool_objs[i].pool(x)
        # last level encoder
        edge_index = mesh_edge_indices[self.levels - 1]
        x = self.enc_blocks[self.levels - 1](x, edge_index)
        x = self.bottleneck(x, edge_index)
        # Decoder: unpool & concat
        for i in range(self.levels - 1, 0, -1):
            # unpool from current coarse x to finer level
            pool_idx = i - 1
            x = self.pool_objs[pool_idx].unpool(x)
            # concat with skip connection
            skip = enc_feats[pool_idx]
            x = torch.cat([x, skip], dim=1)
            edge_index = mesh_edge_indices[pool_idx]
            x = self.dec_blocks[self.levels - 1 - pool_idx](x, edge_index)
        # final projection to out channels per node
        out = self.final_lin(x)
        return out

# -----------------------------
# Training loop
# -----------------------------

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    it = 0
    optimizer.zero_grad()
    for step, data in enumerate(dataloader):
        # data.x: (N, C_in), data.y: (N, C_out)
        data = data.to(device)
        x = data.x
        y = data.y
        # Forward, with mixed precision
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            pred = model(x, mesh_edge_indices)
            # Both pred and y: (N, C_out)
            # compute weighted MSE by latitude if coords available
            loss = F.mse_loss(pred, y)
            loss = loss / ACCUM_STEPS
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * ACCUM_STEPS
        it += 1
    return total_loss / it


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    it = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            x = data.x
            y = data.y
            pred = model(x, mesh_edge_indices)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item()
            it += 1
    return total_loss / it

# -----------------------------
# Entrypoint / example usage
# -----------------------------
if __name__ == '__main__':
    # Load meshes and build edge_index tensors per level
    mesh_edge_indices = []
    mesh_nodes = []
    for lvl in MESH_LEVELS:
        mesh = load_mesh_level(lvl)
        edge_idx = torch.from_numpy(mesh['edges'].T.copy()).long().to(DEVICE)
        mesh_edge_indices.append(edge_idx)
        mesh_nodes.append(mesh['num_nodes'])

    # Create datasets (train/val) - user must prepare remapped files under DATA_ROOT/train and /val
    train_ds = IcosahedralERA5Dataset(DATA_ROOT, mesh_level=MESH_LEVELS[0], split='train')
    val_ds = IcosahedralERA5Dataset(DATA_ROOT, mesh_level=MESH_LEVELS[0], split='val')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Build model
    model = IcosahedralUNet(mesh_levels=MESH_LEVELS, in_ch=INPUT_CHANNELS, out_ch=OUTPUT_CHANNELS, base_ch=64)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    best_val = 1e9
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE, epoch)
        val_loss = validate(model, val_loader, DEVICE)
        t1 = time.time()
        print(f"Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} epoch_time={t1-t0:.1f}s")
        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                       os.path.join(CHECKPOINT_DIR, 'best.pth'))

""
