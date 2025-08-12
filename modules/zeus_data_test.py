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
    'height': 720,
    'width': 1440,
    'channels': 4,
    # video length in and out
    'T1': 3,    # 5 later.
    'MaxT2': 5,
    'FG': 24,   # frames in a frame group
    # patch size
    'patch_size': 8, # 8
    # transformer
    'heads': 8,     # 8
    'dim_head': 32,
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'patch_dim_to_dim' : 1,
    'dim_to_mlp_dim': 2,
    # layers
    'attLayers' : 1,
    'zeusLayers': 6,
    # training
    'seed' : 523,
    'epochs': 20,
    'batch_size' : 2,
    'log_step': 10,
    'optimizer': 'adam',
    'lr' : 1e-5,
    'max_lr' : 1e-3,
    'weight_decay': 0.01,
    'pct_start' : 0.3,
    'anneal_strategy' : 'linear',   # 'cos'
    'div_factor' : 25.0,
    'final_div_factor' : 1e4,
    'ex_folder' : 'D:\Download\experiments',
    'ex_name' : '1st',
}

dataset = Zeus_Dataset(config, root_dir, param_path, split_dates_path, variables)

total_days, start_day, end_day, complete, data_dates = dataset.count_data_days()
print(start_day, end_day)
print(total_days, complete)

# complete, mean, std = dataset.analyze_dataset()
# print(complete, mean, std)

mean, std = dataset.load_dataset_params()
print("mean: {}, std: {}".format(mean, std))

train_dates, vali_dates, test_dates = dataset.split_data_dates(train_pct=0.5, vali_pct=0.3, seed=523)
print("Lengths of dates:", len(train_dates), len(vali_dates), len(test_dates))
print("train_dates[:3]", train_dates[:3])

train_set, vali_set, test_set = dataset.generate_x_sets(seed=523)
print("train_set, vali_set, test_set lengths:", len(train_set), len(vali_set), len(test_set))
print("train_set[:3]", train_set[:3])

# dataset.create_difficulties(end_day, WRITE_FILES=True)
dataset.load_difficulties()
# dataset.show_difficulty("100m_u_component_of_wind")

vars = dataset.load_era5_data_vars()
print('vars ', vars)

dataset.initialize_x_set('train_set', seed=523)

pbar = zeus_tqdm(dataset, desc="Training Batches")

pbar.prefetch('train_set')

cnt = 1
for  batch in pbar: # (B, T1FG, C, H, W), (B, T2FG, C, H, W), (B, 1, C, H, W), (C,)
    if batch is None: break
    batch_x, batch_y, batch_diff, era5_data_vars = batch
    pbar.prefetch('train_set')

    print("trainer ", batch_x.shape, batch_y.shape, batch_diff.shape, era5_data_vars.shape)
    
    # Simulate a training step, taking time and processing.
    # The dataset should be working concurrently to prefetch next data batch.
    for x in range(1000000):
        for y in range(100):
            pass

    loss = torch.tensor(3)

    cnt += 1
    pbar.set_description('loss: {:.5f}'.format(loss.item()))
    # time.sleep(0.5)

pbar.shutdown()
