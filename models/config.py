
# Change

config = {
    'dtype' : 32,   # 16 is complicated. deffered.

    'variables': [],
    'height': 720,
    'width': 1440,
    'patch_size':   16, #16, # 8        ----------------------  optimization param,  Chunk

    'FG': 1, # not 24, for a better granularity control   # frames in a frame group
    'T1':           24, # 24 for Chunk, 3 for 256-head CNNs,
    'T2':           24,

    # Transformer    
    'nHeads':        128, # (128) 8,     # 8    ----------------------    optimization param,  Chunk
    'heads_per_chunk':  2,  # (2)             ---------------------   optimization param,  Chunk
    'tokens_per_chunk':    8,    #             ---------------------   optimization param,  Chunk
    'kv_full_sequence': True,
    'layerFFN' :    "GLU",  # "FFN", "GLU"    ----------------------  optimization param,  Chunk
    'nLayers' :   5, # (10)            ----------------------  optimization param,  Chunk
    'zeusBlocks':   4,    #             ---------------------- optimization param
    'crossLayers':  6,
    'PE': 'random',    # 'random', 'sinusodial'
    'learn_PE': True,

    # CNN
    'add_spatial_conv': True,
    'blocksPerState': [3, 4, 6],
    'ObsorbHDistortion' : False,
    'Zeus_Model_module' : 'Zeus_ResNet',
    'EncLayer' : 'Flash2DEncLayer_shallow', # 'Flash2DEncLayer_deep'
    'nCNNBlocks' : 20,
    'bottleneckRate' : 0.3,
    
    'dropout': 0.0,
    'dropPath': 0.0,
    # 'dim_to_layerFFN_dim': 4,

    'nFrames': 24,
    'mFrames' : 24,
    'radius'  : 1000,   # in kilometers
    'factor10' : 10,    # 10 taking the full image size.
    'transient_steps' : 6000,

    # video length in and out
    # data split
    'train_pct': 80,
    'vali_pct': 20,

    # training
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
    'lr' : 1e-5,    # ---> changed below
    
    # Scheduler
    # (initial_lr = max_lr / div_factor, for pct_start of the total_steps) -> max_lr -> (Final LR = initial_lr/final_div_factor, for the remaining steps)
    'max_lr' : 1e-5,    # ----> changed below
    'total_steps' : 60000,
    'weight_decay': 0.0,    # disable weight_decay, in AdamW
    'pct_start' : 1.0,      #
    'anneal_strategy' : 'linear',   # 'cos'
    'div_factor' : 1.0,    # Initial LR = max_lr / div_factor
    'final_div_factor' : 1.0, # Final LR = initial_lr / final_div_factor
    'accum_steps' : 10,  # the smaller batch_size is, the larger accum_steps is.
    'use_autocast' : True,

    'epoch_patience' : 10,       # Mike!!!
    'useLRLearner' : False,
    
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
    config['variables'] = variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind', "2m_dewpoint_temperature", "surface_pressure"]
    root_path = Path("D:/Zeus_Data")

    # GPU: RTX 4070, Chunk2D
    config['factor10'] = 2  # 2 is the minimum, as 1 causes a non-float parameter error.
    config['dim_to_layerFFN_dim'] = 4

    config['nHeads'] = 128
    config['nLayers'] = 5
    config['heads_per_chunk'] = None
    config['tokens_per_chunk'] = None
    
    config['use_PE'] = True
    
    config['epochs'] = 100
    config['steps_to_save'] = 50
    config['abundant_GPUs'] = False
    
else:
    config['variables'] = variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind', "2m_dewpoint_temperature", "surface_pressure"]
    root_path = Path("/mnt/data/zeus_data")
    
    # GPU: 
    config['factor10'] = 10  # 2 is the minimum, as 1 causes a non-float parameter error.
    config['tokens_per_chunk'] = round(config['T1'] * 72 * config['factor10'] * 144 * config['factor10'] /config['patch_size']/config['patch_size']) // 2

    config['epochs'] = 100
    config['steps_to_save'] = 50
    config['abundant_GPUs'] = False

n = config['nFrames']
C = len(config['variables'])
config['cnn_hidden_channels'] = [n*C, n*C, n*C, n*C, n*C, n*C, n*C, n*C, C] # the last should be C. See ZeusCNN.forward. We get one frame from n frames.
   
#-------------------------------------------------- project list ---------------
projects = {}
projects["ResNet50-24M"] = {           #
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 1,
    'lr': 1e-3,
    'max_lr': 1e-3,
    'batch_size': 2, # (?)
    'ObsorbHDistortion': True,
}
projects["ResNet50-24M-2"] = {           #
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 2,
    'lr': 1e-3,
    'max_lr': 1e-3,
    'batch_size': 1,
    'ObsorbHDistortion': True,
}
projects["ResNet50-NewCodec-2"] = {      # longest. 365K params CNN.
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 2,
    'lr': 1e-3,
    'max_lr': 1e-3,
    'batch_size': 5,
    'ObsorbHDistortion': True,
}
projects["ResNet50-3753K"] = {
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 2,
    'lr': 1e-3,
    'max_lr': 1e-4,
    'batch_size': 2,
    'ObsorbHDistortion': True,
}
projects["ResNet50-9M"] = {
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 2,
    'lr': 1e-3,
    'max_lr': 1e-4,
    'batch_size': 2,
    'blocksPerState': [3, 4, 6],
    'ObsorbHDistortion': True,
}
projects["ResNet50-9M-2"] = {
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 2,
    'lr': 1e-3,
    'max_lr': 1e-4,
    'batch_size': 2,
    'blocksPerState': [3, 4, 6],
    'ObsorbHDistortion': False,
}
projects["ResNet50-6M-2"] = {
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 2,
    'lr': 1e-3,
    'max_lr': 1e-4,
    'batch_size': 2,
    'blocksPerState': [3, 20],    #------ stc [3, 4, 6]
    'ObsorbHDistortion': False,
}

projects["Flash2DBlocked-1"] = {         # to check 
    'Zeus_Model_module' : 'Zeus_Flash2D_Blocked',
    'factor10': 2,
    'lr': 5e-5,
    'max_lr': 5e-5,
    'batch_size': 1,
    'dim_to_layerFFN_dim': 4,
    'nHeads' : 128,
    'nLayers' : 5,
    'zeusBlocks':   4,
    'heads_per_chunk' : None,
    'tokens_per_chunk' : None,
    'PE': 'random',    # 'random', 'sinusodial'
    'learn_PE': True,
    'add_spatial_conv': True,
}

projects["Flash2DBlocked-2"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    'Zeus_Model_module' : 'Zeus_Flash2D_Blocked',
    'factor10': 2,      # simulates 10
    'patch_size': 12,   # simulates 16
    'batch_size': 1,
    'nHeads' : 72,      # should be 16 for 1024 dim.
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    'zeusBlocks':  4,
    'nLayers' : 5,      # This is a major upgrade, from 4.
    'heads_per_chunk' : None,
    'tokens_per_chunk' : None,
    'use_PE' : False,
    'PE': 'sinusodial',    # 'random', 'sinusodial'
    'dim_to_layerFFN_dim': 4,
    'learn_PE': True,
    'add_spatial_conv': True,   # spatial_conv has a major improvement.
    'useLRLearner' : True,
    'accum_steps' : 10,
    'use_autocast' : False,     # autocast, to bfloat16, makes it 10x (?) slower to reduce loss, though it takes 14.5/11.2 speedier.
    'lr': 1e-4,
    'max_lr': 1e-4,
    'dropout': 0.0,
}

projects["Flash2DBlocked-3"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    'Zeus_Model_module' : 'Zeus_Flash2D_TBlocks',
    'factor10': 2,      # simulates 10
    'patch_size': 12,   # 12 -> 576 dim,  16 -> 1024 dim
    'batch_size': 1,
    'nHeads' : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    'zeusBlocks':  4,
    'nLayers' : 5,      # This is a major upgrade, from 4.
    'heads_per_chunk' : None,
    'tokens_per_chunk' : None,
    'use_PE' : False,
    'PE': 'sinusodial',    # 'random', 'sinusodial'
    'dim_to_layerFFN_dim': 4,
    'learn_PE': True,
    'add_spatial_conv': False,   # spatial_conv has a major improvement.
    'useLRLearner' : True,
    'accum_steps' : 10,
    'use_autocast' : False, # should be True as flash_attn_func requires bf16.
    'lr': 1e-6,
    'max_lr': 1e-6,
    'dropout': 0.0,
}

projects["Flash2DBlocked-4"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    'Zeus_Model_module' : 'Zeus_Flash2D_TBlocks',
    'factor10': 2,      # simulates 10
    'patch_size': 12,   # 12 -> 576 dim,  16 -> 1024 dim
    'batch_size': 1,
    'nHeads' : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    'zeusBlocks':  4,
    'nLayers' : 5,      # This is a major upgrade, from 4.
    'heads_per_chunk' : None,
    'tokens_per_chunk' : None,
    'use_PE' : True,
    'PE': 'sinusodial',    # 'random', 'sinusodial'
    'dim_to_layerFFN_dim': 4,
    'learn_PE': True,
    'add_spatial_conv': False,   # spatial_conv has a major improvement.
    'useLRLearner' : True,
    'accum_steps' : 10,
    'use_autocast' : False, # should be True as flash_attn_func requires bf16.
    'lr': 1e-6,
    'max_lr': 1e-6,
    'dropout': 0.0,
}

projects["Flash2DBlocked-5"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    'Zeus_Model_module' : 'Zeus_Flash2D_TBlocks',
    'factor10': 2,      # simulates 10
    'patch_size': 12,   # 12 -> 576 dim,  16 -> 1024 dim
    'batch_size': 1,
    'nHeads' : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    'zeusBlocks':  5,
    'nLayers' : 5,      # This is a major upgrade, from 4.
    'heads_per_chunk' : None,
    'tokens_per_chunk' : None,
    'use_PE' : True,
    'PE': 'sinusodial',    # 'random', 'sinusodial'
    'dim_to_layerFFN_dim': 4,
    'learn_PE': True,
    'add_spatial_conv': False,   # spatial_conv has a major improvement.
    'useLRLearner' : True,
    'accum_steps' : 10,
    'use_autocast' : False, # should be True as flash_attn_func requires bf16.
    'lr': 1e-6,
    'max_lr': 1e-6,
    'dropout': 0.0,
}

projects["Flash2DBlocked-6"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    'Zeus_Model_module' : 'Zeus_Flash2D_TBlocks',
    'factor10': 2,      # simulates 10
    'patch_size': 12,   # 12 -> 576 dim,  16 -> 1024 dim
    'batch_size': 1,
    'nHeads' : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    'zeusBlocks':  5,
    'nLayers' : 5,      # This is a major upgrade, from 4.
    'heads_per_chunk' : None,
    'tokens_per_chunk' : None,
    'use_PE' : True,
    'PE': 'sinusodial',    # 'random', 'sinusodial'
    'dim_to_layerFFN_dim': 4,
    'learn_PE': True,
    'add_spatial_conv': False,   # spatial_conv has a major improvement.
    'useLRLearner' : True,
    'accum_steps' : 10,
    'use_autocast' : True, # should be True as flash_attn_func requires bf16.
    'lr': 1e-6,
    'max_lr': 1e-6,
    'dropout': 0.0,
}

projects["Flash2DBlocked-7"] = projects["Flash2DBlocked-6"]
projects["Flash2DBlocked-7"].update({
    'zeusBlocks':  3,
    'nLayers' : 3,      # This is a major upgrade, from 4.
    'PE': 'random',    # 'random', 'sinusodial'
})

projects["Flash2DBlocked-8"] = projects["Flash2DBlocked-7"]
projects["Flash2DBlocked-8"].update({
    'useLRLearner' : False,
})

projects["Flash2DBlocked-9"] = projects["Flash2DBlocked-8"]
projects["Flash2DBlocked-8"].update({
    'useLRLearner' : True,
    'zeusBlocks':  5,
    'nLayers' : 5,      # This is a major upgrade, from 4.
})


projects["Flash2DBlocked-10"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    'Zeus_Model_module' : 'Zeus_Flash2D_TBlocks',
    'factor10': 10,      # simulates 10
    'patch_size': 12,   # 12 -> 576 dim,  16 -> 1024 dim
    'batch_size': 1,
    'nHeads' : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    'zeusBlocks':  5,
    'nLayers' : 5,      # This is a major upgrade, from 4.
    'heads_per_chunk' : None,
    'tokens_per_chunk' : None,
    'use_PE' : True,
    'PE': 'random',    # 'random', 'sinusodial'
    'dim_to_layerFFN_dim': 4,
    'learn_PE': True,
    'add_spatial_conv': True,   # spatial_conv has a major improvement.
    'useLRLearner' : True,
    'accum_steps' : 10,
    'use_autocast' : True, # should be True as flash_attn_func requires bf16.
    'lr': 1e-6,
    'max_lr': 1e-6,
    'dropout': 0.0,
}

projects["ResNet_NoStage_B20_BR30"] = {
    'Zeus_Model_module' : 'Zeus_ResNet',
    'factor10': 10,
    'lr': 1e-5,
    'max_lr': 1e-5,
    'batch_size': 1,
    'ObsorbHDistortion': False,
    'nCNNBlocks' : 20,
    'bottleneckRate' : 0.30,
}

#-------------------------------- select proejct -----------------------
select = "ResNet_NoStage_B20_BR30" # "ResNet50-NewCodec-2" #"ResNet50-9M" # "ResNet50-3753K" #"ResNet50-NewCodec-2"  #'ResNet50-24M-2'

config['bayesProject'] = select
config.update(projects[select])
print('bayesProject:', config['bayesProject'])
