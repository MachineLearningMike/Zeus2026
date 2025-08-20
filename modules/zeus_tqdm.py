import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import time

class zeus_tqdm(tqdm):
    def __init__(self, iterable, **kwargs):
        super(zeus_tqdm, self).__init__(iterable, **kwargs)
    
    def prefetch(self, x_set_label):
        self.iterable.__prefetch__(x_set_label)

    def generate_x_sets(self, seed=None):
        self.iterable.generate_x_sets(seed=seed)

    def no_more_data(self):
        return self.iterable.no_more_data()
    
    def start_prefetch_thread(self):
        self.iterable.start_prefetch_thread()

    def stop_prefetch_thread(self):
        self.iterable.shutdown()
