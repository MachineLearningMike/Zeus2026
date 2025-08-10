import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import time

class zeus_tqdm(tqdm):
    def __init__(self, iterable, **kwargs):
        super(zeus_tqdm, self).__init__(iterable, **kwargs)
    
    def prefetch(self, T2):
        self.iterable.__prefetch__(T2)
