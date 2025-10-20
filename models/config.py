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
    'nHeads':        128, # (128) 8,     # 8    ----------------------    optimization param,  Chunk

    'heads_per_chunk':  2,  # (2)             ---------------------   optimization param,  Chunk
    'tokens_per_chunk':    8,    #             ---------------------   optimization param,  Chunk
    'kv_full_sequence': True,

    # 'nLayers' :   5, # (10)            ----------------------  optimization param,  Chunk
    'layerFFN' :    "GLU",  # "FFN", "GLU"    ----------------------  optimization param,  Chunk

    'zeusBlocks':   4,    #             ---------------------- optimization param
    'crossLayers':  6,
    
    'PE': 'random',    # 'random', 'sinusodial'
    'learn_PE': True,
    'add_spatial_conv': True,
    
    'dropout':      0.05,
    'attn_dropout': 0.05,
    'drop_path':    0.05,
    # 'dim_to_layerFFN_dim': 4,

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
    
    'data_years' : 100,    # The latest 'data_years' years of the databased years are accounted for. 

    'optimizer': 'adam',    # 'adamw' takes more momory for optimizer.
    # 'lr' : 1e-5,    # ---> changed below
    
    # Scheduler
    # (initial_lr = max_lr / div_factor, for pct_start of the total_steps) -> max_lr -> (Final LR = initial_lr/final_div_factor, for the remaining steps)
    # 'max_lr' : 1e-5,    # ----> changed below
    'total_steps' : 60000,
    'weight_decay': 0.0,    # disable weight_decay, in AdamW
    'pct_start' : 1.0,      #
    'anneal_strategy' : 'linear',   # 'cos'
    'div_factor' : 1.0,    # Initial LR = max_lr / div_factor
    'final_div_factor' : 1.0, # Final LR = initial_lr / final_div_factor

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
    config['dim_to_layerFFN_dim'] = 4

    config['nHeads'] = 128
    config['nLayers'] = 5
    config['heads_per_chunk'] = None
    config['tokens_per_chunk'] = None

    config['no_PE'] = True
    
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

# Chunk2D and simple Chunk
if config['dim_to_layerFFN_dim'] == 4:
    config['bayesProject'] = "-".join([str(config['patch_size']), str(config['nHeads']), str(config['heads_per_chunk']), str(config['tokens_per_chunk']), str(config['nLayers']), config['layerFFN'], "no_PE" if config['no_PE'] else "" ])
else:
    config['bayesProject'] = "-".join([str(config['patch_size']), str(config['nHeads']), str(config['heads_per_chunk']), str(config['tokens_per_chunk']), str(config['nLayers']), str(config['dim_to_layerFFN_dim']), config['layerFFN'], "no_PE" if config['no_PE'] else ""])

config['lr'] = config['max_lr'] = 1e-3
config['batch_size'] = 5 #5
config['bayesProject'] = "ResNet50-NewCodec-2"      # Current Convolution

# config['lr'] = config['max_lr'] = 5e-5
# config['batch_size'] = 1
# config['bayesProject'] = "Flash2DBlocked-1"         # Current Transformer

print('bayesProject:', config['bayesProject'])

# config['bayesProject'] = config['bayesProject'] + '-NoMHA'  # NoMHA