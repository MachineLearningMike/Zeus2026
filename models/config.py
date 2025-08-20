variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind']   # order matters.

config = {
    'dtype' : 32,   # 16 is complicated. deffered.
    # image h w c
    'height': 72,
    'width': 144,
    # video length in and out
    'variables': variables,
    'T1': 1,    # 2 runs out of GPU memory 5 later.
    'maxT2': 6,
    'FG': 24,   # frames in a frame group
    # data split
    'train_pct': 75,
    'vali_pct': 25,
    # patch size
    'patch_size': 8, # 8
    # transformer
    'heads': 8,     # 8
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'dim_to_mlp_dim': 2,
    # layers:   max (2, 3) on laptop
    'attLayers' : 2,
    'zeusLayers': 2,
    # training
    'hypers' : {},
    'seed' : 523,
    'include_difficulty' : False,
    'epochs': 50,
    'batch_size' : 1,
    'log_step': 10,
    'optimizer': 'adam',
    'lr' : 1e-6,
    'max_lr' : 2e-6,
    'weight_decay': 0.01,
    'pct_start' : 0.3,
    'anneal_strategy' : 'linear',   # 'cos'
    'div_factor' : 25.0,
    'final_div_factor' : 1e4,
    'epoch_patience' : 5,       # Mike!!!

    'seed': 523,
    'bayesProject' : 'production_0820_16',    # Represents an instance of a whole bayesian miximization. Mike!!!
}

from pathlib import Path
import platform
if platform.system() == "Windows":  
    root_path = Path("D:/Zeus_Data")
    config['height'] = 72
    config['width'] = 144
else:   
    root_path = Path("/mnt/Data/Zeus_Data")
    config['height'] = 720
    config['width'] = 1440
    
