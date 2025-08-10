import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import time
import xarray as xr
import numpy as np

from zeus_dataset import Zeus_Dataset
from zeus_tqdm import zeus_tqdm

from pathlib import Path

root_dir = Path("D:\Download\CDS")
param_path = Path("D:\Download\dataset_params.json")
split_dates_path = Path("D:\Download\split_dates.json")
variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind']   # order matters.

config = {
    # image h w c
    'height': 72,
    'width': 144,
    'channels': 4,
    # video length in and out
    'T1': 5,    # 5 later.
    'MaxT2': 5,
    'FG': 24,   # frames in a frame group
    # patch size
    'patch_size': 8,
    # transformer
    'heads': 8,
    'dim_head': 32,
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'patch_dim_to_dim' : 1,
    'dim_to_mlp_dim': 2,
    # layers
    'attLayers' : 1,
    'zeusLayers': 6,
}

data_set = Zeus_Dataset(config, root_dir, param_path, split_dates_path, variables)

pbar = zeus_tqdm(data_set, desc="Training Batches")

pbar.prefetch(1)

cnt = 1
for batch_x, batch_y in pbar:
    pbar.prefetch(cnt)
    # Simulate a training step
    output = batch_x * 2  # Replace with model(batch_x)
    loss = ((output - batch_y)**2).mean()  # Dummy loss

    cnt += 1
    pbar.set_description('loss: {:.5f}'.format(loss.item()))
    time.sleep(0.5)
