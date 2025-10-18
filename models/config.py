variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind']   # order matters.

config = {
    'dtype' : 32,   # 16 is complicated. deffered.
    # image h w c
    'height': 720,
    'width': 1440,
    'FG': 1, # not 24, for a better granularity control   # frames in a frame group
    'variables': variables,    
    'T1':           24, # 24 for Chunk, 3 for 256-head CNNs,
    'T2':           24,
    
    'patch_size':   16, #16, # 8        ----------------------  optimization param,  Chunk
    'heads':        128, # (128) 8,     # 8    ----------------------    optimization param,  Chunk

    'heads_per_chunk':  2,  # (2)             ---------------------   optimization param,  Chunk
    'tokens_per_chunk':    8,    #             ---------------------   optimization param,  Chunk

    # 'attLayers' :   5, # (10)            ----------------------  optimization param,  Chunk
    'layerFFN' :    "GLU",  # "FFN", "GLU"    ----------------------  optimization param,  Chunk

    'zeusLayers':   4,    #             ---------------------- optimization param
    'crossLayers':  6,
    
    'dropout':      0.0,
    'attn_dropout': 0.0,
    'drop_path':    0.0,
    # 'dim_to_mlp_dim': 4,

    'nFrames': 24,
    'mFrames' : 24,
    'radius'  : 1000,   # in kilometers
    'factor10' : 10,    # 10 taking the full image size.
    'transient_steps' : 6000,

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
    # The bottleneck is on file reading.
    'batch_size' : 10, #(10)
    'log_step': 10,
    
    'skip_validation' : False,
    
    'data_last_years' : 100,    # The latest 'data_last_years' years of the databased years are accounted for. 

    'optimizer': 'adam',    # 'adamw' takes more momory for optimizer.
    # 'lr' : 1e-5,    # ---> changed below
    
    # Scheduler
    # (initial_lr = max_lr / div_factor, for pct_start of the total_steps) -> max_lr -> (Final LR = initial_lr/final_div_factor, for the remaining steps)
    # 'max_lr' : 1e-5,    # ----> changed below
    'total_steps' : 60000,
    'weight_decay': 0.0,    # disable weight_decay, in AdamW
    'pct_start' : 0.5,      #
    'anneal_strategy' : 'linear',   # 'cos'
    'div_factor' : 1.0,    # Initial LR = max_lr / div_factor
    'final_div_factor' : 10, # Final LR = initial_lr / final_div_factor

    'epoch_patience' : 10,       # Mike!!!
    
    'abundant_GPUs' : False,
    'doubleZeus' : False,

    # for 'doubleZeus' == True
    'freeze_zeus1' : False,
    'skip_net_for_zeus2' : True,

    'seed': 523,
    # 'bayesProject' : '16-128-8-6-GLU' # '16-128-8-6-GLU' # 'f2-k5-3', 'f4-k5-3' #'production_0923_16' #'production_0823-18' #31_12',  #23-18  # Represents an instance of a whole bayesian miximization. Mike!!!
}

from pathlib import Path
import platform
if platform.system() == "Windows":  
    
    root_path = Path("D:/Zeus_Data")

    # GPU: RTX 4070, Chunk2D
    config['factor10'] = 2  # 2 is the minimum, as 1 causes a non-float parameter error.
    config['dim_to_mlp_dim'] = 4

    config['heads'] = 128
    config['attLayers'] = 5
    config['heads_per_chunk'] = 32  # 32 == 64
    config['tokens_per_chunk'] = round(config['T1'] * 72 * config['factor10'] * 144 * config['factor10'] /config['patch_size']/config['patch_size']) // 2

    config['noPE'] = True
    
    config['batch_size'] = 5
    config['epochs'] = 100
    config['steps_to_save'] = 50
    config['abundant_GPUs'] = False
    
else:
    root_path = Path("/mnt/Data/Zeus_Data")

    # GPU: 
    config['factor10'] = 10  # 2 is the minimum, as 1 causes a non-float parameter error.
    config['tokens_per_chunk'] = round(config['T1'] * 72 * config['factor10'] * 144 * config['factor10'] /config['patch_size']/config['patch_size']) // 2

    config['epochs'] = 100
    config['steps_to_save'] = 50
    config['abundant_GPUs'] = False

n = config['nFrames']
C = len(config['variables'])

config['cnn_hidden_channels'] = [n*C, n*C, n*C, n*C, n*C, n*C, n*C, n*C, C] # the last should be C. See ZeusCNN.forward. We get one frame from n frames.
   
# config['cnn_hidden_channels'] = [2*n*C, 2*n*C, 2*n*C, 2*n*C, 2*n*C, 2*n*C, 2*n*C, 2*n*C, C]

# (-5, -4,)
config['lr'] = config['max_lr'] = 1e-3 # 1e-3 # pow(10, -(config['attLayers']-2)/2) * 1e-2

# Chunk2D and simple Chunk
if config['dim_to_mlp_dim'] == 4:
    config['bayesProject'] = "-".join([str(config['patch_size']), str(config['heads']), str(config['heads_per_chunk']), str(config['tokens_per_chunk']), str(config['attLayers']), config['layerFFN'], "noPE" if config['noPE'] else "" ])
else:
    config['bayesProject'] = "-".join([str(config['patch_size']), str(config['heads']), str(config['heads_per_chunk']), str(config['tokens_per_chunk']), str(config['attLayers']), str(config['dim_to_mlp_dim']), config['layerFFN'], "noPE" if config['noPE'] else ""])

config['bayesProject'] = "ResNet50"
config['bayesProject'] = "ResNet50-NewCodec-2"
# config['bayesProject'] = "Latent-ConvNeXtV2"

print('bayesProject:', config['bayesProject'])

# config['bayesProject'] = config['bayesProject'] + '-NoMHA'  # NoMHA