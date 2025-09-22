variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind']   # order matters.

config = {
    'dtype' : 32,   # 16 is complicated. deffered.
    # image h w c
    'height': 720,
    'width': 1440,
    'T1': 3,    # 2, 3, 4, 6, 12, ----------------------
    'FG': 1, # not 24, for a better granularity control   # frames in a frame group
    'variables': variables,
    
    'patch_size':   16, #16, # 8  ------------------------
    'heads':        256, #8,     # 8   ------------------------
    'attLayers' :   1,    #           ----------------------
    'zeusLayers':   8,    #           ----------------------

    'dropout':      0.0,
    'attn_dropout': 0.0,
    'drop_path':    0.0,
    'dim_to_mlp_dim': 4,

    # video length in and out
    'maxT2': 6,
    # data split
    'train_pct': 75,
    'vali_pct': 25,

    # training
    'hypers' : {},
    'seed' : 523,
    'include_difficulty' : False,
    'epochs': 50,
    'steps_to_save' : 50,
    'batch_size' : 1,
    'log_step': 10,
    'optimizer': 'adam',    # 'adamw' takes more momory for optimizer.
    'lr' : 1e-6,    # -7 normal
    'max_lr' : 1e-6,
    'weight_decay': 0.0,    # disable weight_decay, in AdamW
    'pct_start' : 0.3,
    'anneal_strategy' : 'linear',   # 'cos'
    'div_factor' : 10.0,
    'final_div_factor' : 10,
    'epoch_patience' : 3,       # Mike!!!
    
    'abundant_GPUs' : False,
    'doubleZeus' : False,

    'freeze_zeus1' : False,
    'skip_net_for_zeus2' : True,

    'seed': 523,
    'bayesProject' : 'production_0922_12' #'production_0823-18' #31_12',  #23-18  # Represents an instance of a whole bayesian miximization. Mike!!!
}


from pathlib import Path
import platform
if platform.system() == "Windows":  
    root_path = Path("D:/Zeus_Data")
    config['height'] = 72 * 2
    config['width'] = 144
    config['patch_size'] = 16
    config['attLayers'] = 1
    config['zeusLayers'] = 2
    config['epochs'] = 3
    config['steps_to_save'] = 5
    config['abundant_GPUs'] = False
    config['doubleZeus'] = True
else:   
    root_path = Path("/mnt/Data/Zeus_Data")

    
