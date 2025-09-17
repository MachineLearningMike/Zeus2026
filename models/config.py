variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind']   # order matters.

config = {
    'dtype' : 32,   # 16 is complicated. deffered.
    # image h w c
    'height': 720,
    'width': 1440,
    # video length in and out
    'variables': variables,
    'T1': 1,    #
    'maxT2': 6,
    'FG': 24, #24,   # frames in a frame group
    # data split
    'train_pct': 75,
    'vali_pct': 25,
    # patch size
    'patch_size': 12, #16, # 8
    # transformer
    'heads': 4, #8,     # 8
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'drop_path': 0.0,
    'dim_to_mlp_dim': 2,
    # layers:   max (2, 3) on laptop
    'attLayers' : 4,
    'zeusLayers': 4,
    # training
    'hypers' : {},
    'seed' : 523,
    'include_difficulty' : False,
    'epochs': 50,
    'steps_to_save' : 50,
    'batch_size' : 1,
    'log_step': 10,
    'optimizer': 'adam',    # 'adamw' takes more momory for optimizer.
    'lr' : 1e-12,    # -7 normal
    'max_lr' : 2e-12,
    'weight_decay': 0.0,    # disable weight_decay, in AdamW
    'pct_start' : 0.3,
    'anneal_strategy' : 'linear',   # 'cos'
    'div_factor' : 10.0,
    'final_div_factor' : 10,
    'epoch_patience' : 3,       # Mike!!!
    
    'abundant_GPUs' : False,
    'doubleZeus' : True,

    'freeze_1st_half' : True,
    'skip_2nd_half' : True,

    'seed': 523,
    'bayesProject' : 'production_0823_18' #'production_0823-18' #31_12',  #23-18  # Represents an instance of a whole bayesian miximization. Mike!!!
}


from pathlib import Path
import platform
if platform.system() == "Windows":  
    root_path = Path("D:/Zeus_Data")
    config['height'] = 72 * 2
    config['width'] = 144
    config['attLayers'] = 1
    config['zeusLayers'] = 4
    config['epochs'] = 3
    config['steps_to_save'] = 5
    config['abundant_GPUs'] = False
    config['doubleZeus'] = True
else:   
    root_path = Path("/mnt/Data/Zeus_Data")

    
