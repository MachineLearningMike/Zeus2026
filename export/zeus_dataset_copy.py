import numpy as np
import torch
import json
from einops import rearrange
from datetime import datetime, date, timedelta

from torch.utils.data import Dataset

class Zeus_Dataset(Dataset):
    def __init__(self, pm, dataset_path, management_path):
        super().__init__()

        self.param_path = management_path / 'dataset_params.json'
        self.mean, self.std, self.mean_ready_cpu, self.std_ready_cpu, self.mean_ready_cuda, self.std_ready_cuda = self.load_dataset_params()

    def load_dataset_params(self):
        with open(self.param_path, "r") as f:
            data = json.load(f)
            # print('------', data, torch.tensor(data['mean']).shape)
        mean, std = torch.tensor(data['mean']), torch.tensor(data['std'])
        
        std = std # * (self.pm.factor10 / 10)   # Mike!!! This multiplication make sdt deviate from 1.
        
        self.mean, self.std = mean, std
        
        # This violence may cause inconsistency, I know.
        mean_ready_cpu = rearrange(mean, 'c -> c 1 1').to("cpu")
        std_ready_cpu = rearrange(std, 'c -> c 1 1').to("cpu")
        
        mean_ready_cuda, std_ready_cuda = None, None
        if torch.cuda.is_available():
            mean_ready_cuda = rearrange(mean, 'c -> c 1 1').to("cuda")
            std_ready_cuda = rearrange(std, 'c -> c 1 1').to("cuda")
        
        return mean, std, mean_ready_cpu, std_ready_cpu, mean_ready_cuda, std_ready_cuda # (C,), (C,)

    def normalize(self, data):    # data: (... C, H, W)
        # device = data.device
        # return ( data - rearrange(self.mean, 'c -> c 1 1').to(device) ) / rearrange(self.std, 'c -> c 1 1').to(device)
        if data.is_cuda:
            return ( data - self.mean_ready_cuda ) / self.std_ready_cuda
        else:
            return ( data - self.mean_ready_cpu ) / self.std_ready_cpu
    
    def unnormalize(self, data):    # data: (... C, H, W)
        # Mike!!! I found this error Oct 23 only. 
        # Over 2 months of test training was based on this wrong formula, forcing neual networks to distort accordingly.
        # return ( data - rearrange(self.mean, 'c -> c 1 1').to(device) ) / rearrange(self.std, 'c -> c 1 1').to(device)

        # device = data.device
        # return data * rearrange(self.std, 'c -> c 1 1').to(device) + rearrange(self.mean, 'c -> c 1 1').to(device)
        if data.is_cuda:
            return data * self.std_ready_cuda + self.mean_ready_cuda
        else:
            return data * self.std_ready_cpu + self.mean_ready_cpu
                

#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":

    # src = "/mnt/data/zeus_data/CDS"           # source CDS folder
    # drive_even = "/mnt/data_even/CDS"  # target drive for even days
    # drive_odd = "/mnt/data_odd/CDS"   # target drive for odd days
    # divide_cds_by_day(src, drive_even, drive_odd)
    
    # src = "/mnt/data/zeus_data"
    # dst = "/mnt/data_even"
    # copy_except_cds(src, dst)

    from pathlib import Path
    from config_14nov25 import config
    from params_14nov25 import Params
    pm = Params(config)

    dataset = Zeus_Dataset(pm, Path(config['dataset_path']), Path(config['management_path']))

