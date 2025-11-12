
# Change

config = {
    #============================ project parameters ===============================
    'bayesProject': None,

    # training
    'seed' : 523,
    'steps_to_save' : 50,
    'log_step': 10,    
        
    # Scheduler
    # (initial_lr = max_lr / div_factor, for pct_start of the total_steps) -> max_lr -> (Final LR = initial_lr/final_div_factor, for the remaining steps)
    'GPUWaitsData' : True,
    
    'abundant_GPUs' : False,
    'doubleZeus' : False,

    'dual_drive' : True,

    # for 'doubleZeus' == True
    'freeze_zeus1' : False,
    'skip_net_for_zeus2' : True,
    
    #============================ data parameters ===========================================================================
    'dtype' : 32,   # 16 is complicated. deffered.
    'variables': ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind', "2m_dewpoint_temperature", "surface_pressure"],
    'pole_boundary_angle': 60.0,      #

    'org_height': 721,
    'org_width': 1440,
    'factor10' : 10,    # 10 taking the full image size.
    'data_years' : 100,
    'batch_size' : 10, #(10)

    'FG': 1, # not 24, for a better granularity control   # frames in a frame group
    'T1':           24, # 24 for Chunk, 3 for 256-head CNNs,
    'T2':           24,
    'nFrames': 24,
    'mFrames' : 24,

    'train_pct': 80,
    'vali_pct': 20,

    'include_difficulty' : False,
}


shared_model_params = {
    #============================ model parameters ========================================================================
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
    'use_PE' : True,
    'patch_size':   16, #16, # 8        ----------------------  optimization param,  Chunk
    
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
    'dim_to_layerFFN_dim': 4,

    'radius'  : 1000,   # in kilometers
    'transient_steps' : 6000,

    'optimizer': 'adam',    # 'adamw' takes more momory for optimizer.
    'lr' : 1e-5,    # ---> changed below
    'bottleneck_dropout' : 0.03,
    'skip_validation' : False,
    'epochs': 50,
    'overwrite_saved_lr': False,
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
}

config.update({
    'model_0': shared_model_params,
    'model_1': shared_model_params,
    'model_2': shared_model_params,
    'model_3': shared_model_params,
    'model_4': shared_model_params,
})


from pathlib import Path
import platform
if platform.system() == "Windows":  
    if not config['dual_drive']:
        config['dataset_path'] = "d:/zeus_data"
        config['management_path'] = "d:/zeus_data/management"
    else:
        config['dataset_path'] = "d:/zeud_data"
        config['management_path'] = "d:/zeus_data/management"
    
    # GPU: RTX 4070, Chunk2D
    config['factor10'] = 4  # 2 is the minimum, as 1 causes a non-float parameter error.
    config['dim_to_layerFFN_dim'] = 4

    config['nHeads'] = 128
    config['nLayers'] = 5
    config['heads_per_chunk'] = None
    config['tokens_per_chunk'] = None
    
    config['use_PE'] = True
    
    config['epochs'] = 100
    config['steps_to_save'] = 5
    config['abundant_GPUs'] = False
    
else:
    if not config['dual_drive']:
        config['dataset_path'] = "/mnt/data"
        config['management_path'] = "/home/mike/zeus_management"
    else:
        config['dataset_path'] = "/mnt"
        config['management_path'] = "/home/mike/zeus_management"
    
    # GPU: 
    config['factor10'] = 10  # 2 is the minimum, as 1 causes a non-float parameter error.
    # config['tokens_per_chunk'] = round(config['T1'] * 72 * config['factor10'] * 144 * config['factor10'] /config['patch_size']/config['patch_size']) // 2

    config['epochs'] = 100
    config['steps_to_save'] = 50
    config['abundant_GPUs'] = False

n = config['nFrames']
C = len(config['variables'])
   
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
    'org_height': 720,
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
    'org_height': 720,
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
    'org_height': 720,
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
    'org_height': 720,
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
    'org_height': 720,
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
    'org_height': 720,
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
    'org_height': 720,
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
    'org_height': 721,
    'factor10': 10,
    'lr': 1e-5,
    'max_lr': 1e-5,
    'overwrite_saved_lr': True,
    'batch_size': 1,
    'useLRLearner' : True,
    'accum_steps' : 10,
    'skip_validation' : True,
    'ObsorbHDistortion': False,
    'nCNNBlocks' : 20,  # NO change - paramemters
    'bottleneckRate' : 0.30,    # NO change - parameters
}

projects["ResNet_NoStage_B15_BR30"] = {
    'management': {
    'Zeus_Model_module' : 'Zeus_ResNet',
    'GPUWaitsData' : True,
    'models_to_play': [0,]
    },
    
    'data': {
    'org_height': 721,
    'batch_size': 1,
    },
    
    'model_0': {
    'lr': 1e-5,     # 1e-3 for the first 120,000 steps
    'max_lr': 1e-5,
    'overwrite_saved_lr': False,
    'useLRLearner' : False,
    'accum_steps' : 10,   
    'skip_validation' : False,
    'ObsorbHDistortion': False,
    'nCNNBlocks' : 15,  # NO change - paramemters
    'bottleneckRate' : 0.30,    # NO change - parameters
    'bottleneck_dropout' : 0.00,
    },
        
    'model_1': {},
    'model_2': {},
    'model_3': {},
    'model_4': {},
}

#-------------------------------- select proejct -----------------------
project = "ResNet_NoStage_B15_BR30" # "ResNet50-NewCodec-2" #"ResNet50-9M" # "ResNet50-3753K" #"ResNet50-NewCodec-2"  #'ResNet50-24M-2'
config['bayesProject'] = project
config.update(projects[project]['management'])
config.update(projects[project]['data'])
for id in range(5):
    key = 'model_' + str(id)
    config[key].update(projects[project][key])

print('bayesProject:', config['bayesProject'])
# print(config)
