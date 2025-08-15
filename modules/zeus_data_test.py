import torch
from pathlib import Path

from zeus_tqdm import zeus_tqdm
from zeus_dataset import Zeus_Dataset

from models.config import config

import platform
if platform.system() == "Windows":  root_path = Path("D:/Zeus_Data")
else:   root_path = Path("/mnt/Data/Zeus_Data")

dataset = Zeus_Dataset(config, root_path)

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

dataset.select_x_set('train_set', seed=523)

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
