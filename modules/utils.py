import os
import torch
import random
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

def check_gradient(model):
    
    for name, param in model.named_parameters():
        if (param.requires_grad is True) and (param.grad is None):
            print(f"Parameter '{name}' did NOT receive a gradient --------------------------------------------")
        else:
            pass
            # print(f"Parameter '{name}' received gradient, shape: {param.grad.shape}, device: {param.grad.device}")

def get_gpu_names():
    gpu_names = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
            if "A100" in gpu_name:
                print("  -> This is an A100")
                gpu_names.append('A100')
                
            elif "B200" in gpu_name:
                print("  -> This is a B200")
                gpu_names.append('B200')
                
    return gpu_names

            # Note: B200 identification string may vary when released