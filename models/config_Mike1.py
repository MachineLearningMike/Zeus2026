import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta, timezone

# Change

from modules.zeus_dataset_var_names import VARIABLE_NAME_LIST_UNCHANGEABLE, VARIABLE_GROUP_LIST_UNCHANGEABLE, HECTA_PASCAL_UNCHANGEABLE
# The order of meteo_variables must consistent across training and inference !!!!!!!!!!!!!!

config = {
    #============================ project parameters ===============================
    "bayesProject": None,

    # training
    "seed" : 523,
    "steps_to_save" : 10,
    "log_step": 10,    
        
    # Scheduler
    # (initial_lr = max_lr / div_factor, for pct_start of the total_steps) -> max_lr -> (Final LR = initial_lr/final_div_factor, for the remaining steps)
    "GPUWaitsData" : True,
    
    "abundant_GPUs" : False,
    "doubleZeus" : False,

    "dual_drive" : True,
    "metrics"   : "mae", # "rmse"

    # for "doubleZeus" == True
    "freeze_zeus1" : False,
    "skip_net_for_zeus2" : True,
    
    "train_hours_limit" : 1000,    #------------------------------------------------------------------ for tests ----
    
    #============================ data parameters ===========================================================================
    "dataset_name" : "2008_2025",

    "dtype" : 16,
    "meteo_variables": VARIABLE_NAME_LIST_UNCHANGEABLE,   #!!!!!!!!!!!!!!!!!!!!!!!!!!!! The order of meteo_variables must consistent across training and inference !!!!!!
    "variable_groups" : VARIABLE_GROUP_LIST_UNCHANGEABLE,
    "hecta_pascals" : HECTA_PASCAL_UNCHANGEABLE,
    "targetVariables": [ "t2m", "tp", "u100m","v100m", "d2m", "sp",],
    "pole_angle": 30.0,      #
    "hourly_offset": 0,     # exactly 0, 6, 12, 18
    "modulo": 6,
    "nSubstepsPerStep": 1,

    "org_height": 721,
    "org_width": 1440,
    "factor10" : 10,    # 10 taking the full image size.
    "data_years" : 100,
    "batch_size" : 10, #(10)
    "data_limit" : 10**18,

    "FG": 1, # not 24, for a better granularity control   # frames in a frame group
    "T1":           24, # 24 for Chunk, 3 for 256-head CNNs,
    "T2":           24,
    "nFrames": 24,
    "mFrames" : 24,

    "train_pct": 80,
    "vali_pct": 20,

    "include_difficulty" : False,
    
    "train_start_dt":   datetime(2008, 1, 1, 0) - timedelta(hours=1),
    "vali_start_dt" :   datetime(2025, 1, 1, 0) - timedelta(hours=1),
    "test_start_dt" :   datetime(2026, 1, 1, 0) - timedelta(hours=1),
    "test_end_dt_inclusive": datetime(2026, 1, 1, 0) - timedelta(hours=1),
    
    
    "snapshot_pair_on_cuda": False,     # False for light version
    "pair_not_single_snapshot": False,
    "group_normalize": False,           # False for light version
    
    "dense_variables": ["orograph", "sin_lat", "cos_lat", "sin_lon", "cos_lon"],
    "cat_variables": ["land_sea_mask", "soil_type"],
    "tempo_variables": ["sin_orbit", "cos_orbit", "sin_zenith", "cos_zenith"], # ["sin_orbit", "cos_orbit", "sin_solor", "cos_solar"],
}


shared_model_params = {
    #============================ model parameters ========================================================================
    # Transformer    
    "nHeads":        4, # (128) 8,     # 8    ----------------------    optimization param,  Chunk
    "heads_per_chunk":  2,  # (2)             ---------------------   optimization param,  Chunk
    "tokens_per_chunk":    8,    #             ---------------------   optimization param,  Chunk
    "kv_full_sequence": True,
    "layerFFN" :    "GLU",  # "FFN", "GLU"    ----------------------  optimization param,  Chunk
    "nLayers" :   1, # (10)            ----------------------  optimization param,  Chunk
    "zeusBlocks":   4,    #             ---------------------- optimization param
    "crossLayers":  6,
    "PE": "random",    # "random", "sinusodial"
    "learn_PE": True,
    "use_PE" : True,
    "patch_size":   16, #16, # 8        ----------------------  optimization param,  Chunk
    
    # CNN
    "add_spatial_conv": True,
    "blocksPerState": [3, 4, 6],
    "ObsorbHDistortion" : False,
    "Zeus_Model_module" : "Zeus_ResNet",
    "EncLayer" : "Flash2DEncLayer_shallow", # "Flash2DEncLayer_deep"
    "nCNNBlocks" : 20,
    "bottleneckRate" : 0.3,
    
    "dropout": 0.0,
    "dropPath": 0.0,
    "dim_to_layerFFN_dim": 4,

    "radius"  : 1000,   # in kilometers
    "transient_steps" : 6000,

    #================================ optimizer ==============================
    
    "optimizer": "adamw",    # "adam", "adamw" takes more memory for optimizer.

    #----------- adam & adamw ------------
    'lr' : 1e-5,
    'betas' : (0.9, 0.999),
    'eps' : 1e-8,
    'weight_decay' : 0.01,
    'amsgrad' : False, # False typically - uses max of past v_t instead of exponential average   
       
    #---------------------- scheduler -------------------------
    "max_lr" : 1e-5,    # ----> changed below    
    "pct_start" : 1.0,      #    
    "total_steps" : 60000,
    "anneal_strategy" : "linear",   # "cos"
    "div_factor" : 1.0,    # Initial LR = max_lr / div_factor
    "final_div_factor" : 1.0, # Final LR = initial_lr / final_div_factor

    #------------------------------------------------------------
    "bottleneck_dropout" : 0.03,
    "skip_validation" : False,
    "epochs": 50,
    "overwrite_saved_lr": False,

    "accum_steps" : 10,  # the smaller batch_size is, the larger accum_steps is.
    "use_autocast" : True,
    "epoch_patience" : 10,       # Mike!!!
    "useLRLearner" : False,

    # # SWIN
    # "nHeads": 2,
    # "patch_kernel": 2,
    # "patch_stride": 2,
    # "swin_dim"  : 512,       # Check it.
    # "sWin" : (2, 4, 4),      # blocks 1-10: 2 x 2, blocks 11-20: 4 x 4, blocks 21-30: 4 x 4 or 6 x 6
    # "num_swin_block": 30,
    # "pair_bias_dim": 5,     # Careful, this changes code.
    # "nLevels": 6,
    # "increasing": True,

}


from pathlib import Path
import platform

if platform.system() == "Windows":  
    
    if not config["dual_drive"]:
        config["dataset_path"] = "d:/zeus_data"
        config["management_path"] = "d:/zeus_data/management"
    else:
        config["dataset_path"] = "d:/zeus_data"
        config["management_path"] = "d:/zeus_data/management"
    
    config["data_limit"] = 12
    config["dim_to_layerFFN_dim"] = 4
    
    config["factor10"] = 5  # -------------------------- used for light projects

    # "factor10" : 5,  # 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB? Yes, barely.
    
    # shared_model_params['nHeads'] = 4
    # shared_model_params['patch_kernel'] = 4
    # shared_model_params['patch_stride'] = 4
    # shared_model_params['swin_dim'] = 40
    # shared_model_params['sWin'] = (2, 2, 2)
    # shared_model_params['num_swin_block'] = 30
    # shared_model_params['nLevels'] = 6
    # shared_model_params['increasing'] = True
    # shared_model_params['type_pos_bias'] = 'lonBand' # 'shpere', 'lonBand', 'None'
        
    config["use_PE"] = True
    
    config['nSubstepsPerStep'] = 2
    config['dollarsPerHour'] = 0.1      # 3500$ computer for 8 years with daily use of 12 hours.
    
    config["epochs"] = 100
    config["steps_to_save"] = 200       # 
    config["abundant_GPUs"] = False
    
    config["train_start_dt"] = datetime(2008, 1, 1, 0)
    config["vali_start_dt"] = datetime(2025, 1, 1, 0)
    config["test_start_dt"] = datetime(2026, 1, 1, 0)
    config["test_end_dt_inclusive"] = datetime(2026, 1, 1, 0)
    
else:
    if not config["dual_drive"]:
        config["dataset_path"] = "/mnt/data"
        config["management_path"] = "/home/mike/zeus_management"
    else:
        config["dataset_path"] = "/mnt"
        config["management_path"] = "/home/mike/zeus_management"

    config["factor10"] = 10  # 
    config['nSubstepsPerStep'] = 4        # 2 on A100, 4 on B200
    
    from modules.utils import get_gpu_names
    gpu_names = get_gpu_names()
    if 'A100' in gpu_names:
        config['nSubstepsPerStep'] = 1
        config['dollarsPerHour'] = 0.58
        print(f"A100 found!!! nSubstepsPerStep {config['nSubstepsPerStep']}")
    elif 'B200' in gpu_names:
        config['nSubstepsPerStep'] = 4
        config['dollarsPerHour'] = 0.9475
        print(f"B200 found!!! nSubstepsPerStep {config['nSubstepsPerStep']}")
    else:
        config['nSubstepsPerStep'] = 2    # comeback 1
        config['dollarsPerHour'] = 0.1

    
    # shared_model_params['nHeads'] = 2
    # shared_model_params['patch_kernel'] = 4
    # shared_model_params['patch_stride'] = 4
    # shared_model_params['swin_dim'] = 200
    # shared_model_params['sWin'] = (2, 2, 2)
    # shared_model_params['num_swin_block'] = 30
    # shared_model_params['nLevels'] = 6
    # shared_model_params['increasing'] = True
    # shared_model_params['type_pos_bias'] = 'lonBand' # 'shpere', 'lonBand', 'None'

    config["epochs"] = 100
    config["steps_to_save"] = 50
    config["abundant_GPUs"] = False

config.update({
    "model_0": shared_model_params,
    "model_1": shared_model_params,
    "model_2": shared_model_params,
    "model_3": shared_model_params,
    "model_4": shared_model_params,
})


n = config["nFrames"]
C = len(config["meteo_variables"])

# Land-sea-maks from ERA5 is too noisy to accept, so I created this with the help of GPT.
config["land_sea_mask_npy_path"] = "data/land_sea_mask_721x1440.npy_geo"
# int8 version of ERA5 soil type, from GCP
config["soil_type_npy_path"] = "data/soil_type_721x1440.npy_geo"
# ERA5 geopotential_at_surface, from GCP    
config["orograph_npy_path"] = "data/orograph_721x1440.npy_geo"
   
#-------------------------------------------------- project list ---------------
projects = {}
projects["ResNet50-24M"] = {           #
    "Zeus_Model_module" : "Zeus_ResNet",
    "factor10": 1,
    "lr": 1e-3,
    "max_lr": 1e-3,
    "batch_size": 2, # (?)
    "ObsorbHDistortion": True,
}
projects["ResNet50-24M-2"] = {           #
    "Zeus_Model_module" : "Zeus_ResNet",
    "factor10": 2,
    "lr": 1e-3,
    "max_lr": 1e-3,
    "batch_size": 1,
    "ObsorbHDistortion": True,
}
projects["ResNet50-NewCodec-2"] = {      # longest. 365K params CNN.
    "Zeus_Model_module" : "Zeus_ResNet",
    "factor10": 2,
    "lr": 1e-3,
    "max_lr": 1e-3,
    "batch_size": 5,
    "ObsorbHDistortion": True,
}
projects["ResNet50-3753K"] = {
    "Zeus_Model_module" : "Zeus_ResNet",
    "factor10": 2,
    "lr": 1e-3,
    "max_lr": 1e-4,
    "batch_size": 2,
    "ObsorbHDistortion": True,
}
projects["ResNet50-9M"] = {
    "Zeus_Model_module" : "Zeus_ResNet",
    "factor10": 2,
    "lr": 1e-3,
    "max_lr": 1e-4,
    "batch_size": 2,
    "blocksPerState": [3, 4, 6],
    "ObsorbHDistortion": True,
}
projects["ResNet50-9M-2"] = {
    "Zeus_Model_module" : "Zeus_ResNet",
    "factor10": 2,
    "lr": 1e-3,
    "max_lr": 1e-4,
    "batch_size": 2,
    "blocksPerState": [3, 4, 6],
    "ObsorbHDistortion": False,
}
projects["ResNet50-6M-2"] = {
    "Zeus_Model_module" : "Zeus_ResNet",
    "factor10": 2,
    "lr": 1e-3,
    "max_lr": 1e-4,
    "batch_size": 2,
    "blocksPerState": [3, 20],    #------ stc [3, 4, 6]
    "ObsorbHDistortion": False,
}

projects["Flash2DBlocked-1"] = {         # to check 
    "Zeus_Model_module" : "Zeus_Flash2D_Blocked",
    "org_height": 720,
    "factor10": 2,
    "lr": 5e-5,
    "max_lr": 5e-5,
    "batch_size": 1,
    "dim_to_layerFFN_dim": 4,
    "nHeads" : 128,
    "nLayers" : 5,
    "zeusBlocks":   4,
    "heads_per_chunk" : None,
    "tokens_per_chunk" : None,
    "PE": "random",    # "random", "sinusodial"
    "learn_PE": True,
    "add_spatial_conv": True,
}

projects["Flash2DBlocked-2"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    "Zeus_Model_module" : "Zeus_Flash2D_Blocked",
    "org_height": 720,
    "factor10": 2,      # simulates 10
    "patch_size": 12,   # simulates 16
    "batch_size": 1,
    "nHeads" : 72,      # should be 16 for 1024 dim.
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    "zeusBlocks":  4,
    "nLayers" : 5,      # This is a major upgrade, from 4.
    "heads_per_chunk" : None,
    "tokens_per_chunk" : None,
    "use_PE" : False,
    "PE": "sinusodial",    # "random", "sinusodial"
    "dim_to_layerFFN_dim": 4,
    "learn_PE": True,
    "add_spatial_conv": True,   # spatial_conv has a major improvement.
    "useLRLearner" : True,
    "accum_steps" : 10,
    "use_autocast" : False,     # autocast, to bfloat16, makes it 10x (?) slower to reduce loss, though it takes 14.5/11.2 speedier.
    "lr": 1e-4,
    "max_lr": 1e-4,
    "dropout": 0.0,
}

projects["Flash2DBlocked-3"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    "Zeus_Model_module" : "Zeus_Flash2D_TBlocks",
    "org_height": 720,
    "factor10": 2,      # simulates 10
    "patch_size": 12,   # 12 -> 576 dim,  16 -> 1024 dim
    "batch_size": 1,
    "nHeads" : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    "zeusBlocks":  4,
    "nLayers" : 5,      # This is a major upgrade, from 4.
    "heads_per_chunk" : None,
    "tokens_per_chunk" : None,
    "use_PE" : False,
    "PE": "sinusodial",    # "random", "sinusodial"
    "dim_to_layerFFN_dim": 4,
    "learn_PE": True,
    "add_spatial_conv": False,   # spatial_conv has a major improvement.
    "useLRLearner" : True,
    "accum_steps" : 10,
    "use_autocast" : False, # should be True as flash_attn_func requires bf16.
    "lr": 1e-6,
    "max_lr": 1e-6,
    "dropout": 0.0,
}

projects["Flash2DBlocked-4"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    "Zeus_Model_module" : "Zeus_Flash2D_TBlocks",
    "org_height": 720,
    "factor10": 2,      # simulates 10
    "patch_size": 12,   # 12 -> 576 dim,  16 -> 1024 dim
    "batch_size": 1,
    "nHeads" : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    "zeusBlocks":  4,
    "nLayers" : 5,      # This is a major upgrade, from 4.
    "heads_per_chunk" : None,
    "tokens_per_chunk" : None,
    "use_PE" : True,
    "PE": "sinusodial",    # "random", "sinusodial"
    "dim_to_layerFFN_dim": 4,
    "learn_PE": True,
    "add_spatial_conv": False,   # spatial_conv has a major improvement.
    "useLRLearner" : True,
    "accum_steps" : 10,
    "use_autocast" : False, # should be True as flash_attn_func requires bf16.
    "lr": 1e-6,
    "max_lr": 1e-6,
    "dropout": 0.0,
}

projects["Flash2DBlocked-5"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    "Zeus_Model_module" : "Zeus_Flash2D_TBlocks",
    "org_height": 720,
    "factor10": 2,      # simulates 10
    "patch_size": 12,   # 12 -> 576 dim,  16 -> 1024 dim
    "batch_size": 1,
    "nHeads" : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    "zeusBlocks":  5,
    "nLayers" : 5,      # This is a major upgrade, from 4.
    "heads_per_chunk" : None,
    "tokens_per_chunk" : None,
    "use_PE" : True,
    "PE": "sinusodial",    # "random", "sinusodial"
    "dim_to_layerFFN_dim": 4,
    "learn_PE": True,
    "add_spatial_conv": False,   # spatial_conv has a major improvement.
    "useLRLearner" : True,
    "accum_steps" : 10,
    "use_autocast" : False, # should be True as flash_attn_func requires bf16.
    "lr": 1e-6,
    "max_lr": 1e-6,
    "dropout": 0.0,
}

projects["Flash2DBlocked-6"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    "Zeus_Model_module" : "Zeus_Flash2D_TBlocks",
    "org_height": 720,
    "factor10": 2,      # simulates 10
    "patch_size": 12,   # 12 -> 576 dim,  16 -> 1024 dim
    "batch_size": 1,
    "nHeads" : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    "zeusBlocks":  5,
    "nLayers" : 5,      # This is a major upgrade, from 4.
    "heads_per_chunk" : None,
    "tokens_per_chunk" : None,
    "use_PE" : True,
    "PE": "sinusodial",    # "random", "sinusodial"
    "dim_to_layerFFN_dim": 4,
    "learn_PE": True,
    "add_spatial_conv": False,   # spatial_conv has a major improvement.
    "useLRLearner" : True,
    "accum_steps" : 10,
    "use_autocast" : True, # should be True as flash_attn_func requires bf16.
    "lr": 1e-6,
    "max_lr": 1e-6,
    "dropout": 0.0,
}

projects["Flash2DBlocked-7"] = projects["Flash2DBlocked-6"]
projects["Flash2DBlocked-7"].update({
    "zeusBlocks":  3,
    "nLayers" : 3,      # This is a major upgrade, from 4.
    "PE": "random",    # "random", "sinusodial"
})

projects["Flash2DBlocked-8"] = projects["Flash2DBlocked-7"]
projects["Flash2DBlocked-8"].update({
    "useLRLearner" : False,
})

projects["Flash2DBlocked-9"] = projects["Flash2DBlocked-8"]
projects["Flash2DBlocked-8"].update({
    "useLRLearner" : True,
    "zeusBlocks":  5,
    "nLayers" : 5,      # This is a major upgrade, from 4.
})


projects["Flash2DBlocked-10"] = {
    # Step cycle is reduced from 20 sec to 18 sec.
    "Zeus_Model_module" : "Zeus_Flash2D_TBlocks",
    "org_height": 720,
    "factor10": 10,      # simulates 10
    "patch_size": 12,   # 12 -> 576 dim,  16 -> 1024 dim
    "batch_size": 1,
    "nHeads" : 72,      #
    # (zeusBlocks, nLayers) = (4, 5) dropped trainig loss earlier tha (5, 4) for a single-point data.
    "zeusBlocks":  5,
    "nLayers" : 5,      # This is a major upgrade, from 4.
    "heads_per_chunk" : None,
    "tokens_per_chunk" : None,
    "use_PE" : True,
    "PE": "random",    # "random", "sinusodial"
    "dim_to_layerFFN_dim": 4,
    "learn_PE": True,
    "add_spatial_conv": True,   # spatial_conv has a major improvement.
    "useLRLearner" : True,
    "accum_steps" : 10,
    "use_autocast" : True, # should be True as flash_attn_func requires bf16.
    "lr": 1e-6,
    "max_lr": 1e-6,
    "dropout": 0.0,
}

projects["ResNet_NoStage_B20_BR30"] = {
    "Zeus_Model_module" : "Zeus_ResNet",
    "org_height": 721,
    "factor10": 10,
    "lr": 1e-5,
    "max_lr": 1e-5,
    "overwrite_saved_lr": True,
    "batch_size": 1,
    "useLRLearner" : True,
    "accum_steps" : 10,
    "skip_validation" : True,
    "ObsorbHDistortion": False,
    "nCNNBlocks" : 20,  # NO change - paramemters
    "bottleneckRate" : 0.30,    # NO change - parameters
}

projects["ResNet_NoStage_B15_BR30"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_ResNet",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    "ObsorbHDistortion": False,
    "nCNNBlocks" : 15,  # NO change - paramemters
    "bottleneckRate" : 0.30,    # NO change - parameters
    "bottleneck_dropout" : 0.00,
    },
        
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec03"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    },
        
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec12"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "mae", # "mae", "rmse"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    },
        
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec18_1"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    #----------------------------------- SWIN ------------------------
    'nHeads' : 2,
    'patch_kernel' : 4,
    'patch_stride' : 4,
    'swin_dim' : 200,           # 200
    'sWin' : (2, 2, 2),
    'num_swin_block' : 30,
    'nLevels' : 6,
    'increasing' : True,
    'type_pos_bias' : 'lonBand', # 'shpere', 'lonBand', 'None'
    'pair_bias_dim' : 5,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec20_P4W2"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    #----------------------------------- SWIN ------------------------
    # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
    
    'nHeads' : 4,
    'patch_kernel' : 4,
    'patch_stride' : 4,
    'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
    'sWin' : (2, 2, 2),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
    'num_swin_block' : 30,
    'nLevels' : 6,
    'increasing' : True,
    'type_pos_bias' : 'lonBand', # 'shpere', 'lonBand', 'None'
    'pair_bias_dim' : 5,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec20_P6W2"] = {      #---------------- 192 dim B200: 107 GB 1200 ms; 256 dim B200: 140 GB 1500 ms
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    #----------------------------------- SWIN ------------------------
    # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
    
    'nHeads' : 4,
    'patch_kernel' : 6,
    'patch_stride' : 6,
    'swin_dim' : 256, #192: 107GB, 256: 140GB,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
    'sWin' : (2, 2, 2),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
    'num_swin_block' : 30,
    'nLevels' : 8,
    'increasing' : True,
    'type_pos_bias' : 'lonBand', # 'shpere', 'lonBand', 'None'
    'pair_bias_dim' : 5,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec20_P8W2"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    #----------------------------------- SWIN ------------------------
    # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
    
    'nHeads' : 4,
    'patch_kernel' : 8,
    'patch_stride' : 8,
    'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
    'sWin' : (2, 2, 2),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
    'num_swin_block' : 30,
    'nLevels' : 6,
    'increasing' : True,
    'type_pos_bias' : 'lonBand', # 'sphere', 'None'
    'pair_bias_dim' : 5,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}


projects["Zeus_Swin_25Dec22"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
    "lr": 1e-5,     # 1e-3 for the first 120,000 steps
    "max_lr": 1e-5,
    "overwrite_saved_lr": False,
    "useLRLearner" : False,
    "accum_steps" : 10,   
    "skip_validation" : True,
    #----------------------------------- SWIN ------------------------
    # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
    
    'nHeads' : 4,
    'patch_kernel' : 8,
    'patch_stride' : 8,
    'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
    'sWin' : (2, 6, 12),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
    'num_swin_block' : 30,
    'nLevels' : 8,
    'increasing' : True,
    'type_pos_bias' : 'lonBand', # 'sphere', 'None'
    'pair_bias_dim' : 5,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec24"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-5,     # 1e-3 for the first 120,000 steps
        "max_lr": 1e-5,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],
        'head_list' : [4, 8, 16],
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec25_2"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec25_test"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 12),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec26"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 4, 8),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec26_6x6"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_Swin_25Dec26_6x6_no_codec_skip"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_25Dec26_6x6_no_skip_learn"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_25Dec26_6x6_skippable_learnable"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
        'skippable' : True,
        'skip_learnable' : True,
        'qkv_bias' : True,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_25Dec26_6x6_learn_skip_2"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
        'skippable' : True,
        'skip_learnable' : True,
        'qkv_bias' : True,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_25Dec26_6x6_learn_skip_3"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
        'skippable' : True,
        'skip_learnable' : True,
        'qkv_bias' : True,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_25Dec27_6x6_learn_skip_1"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
        'skippable' : True,
        'skip_learnable' : True,
        'qkv_bias' : True,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_25Dec27_6x6_learn_skip_2"] = {
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    'max_cascade_depth' : 1,
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "useLRLearner" : False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [4, 8, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
        'skippable' : True,
        'skip_learnable' : True,
        'qkv_bias' : True,
        
        'choose_best_loss_model' : True,
    },
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_25Dec28_1"] = {
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB?
        
        'nHeads' : 4,
        'patch_kernel' : 4,
        'patch_stride' : 4,
        'swin_dim' : 192,           # 480 is the right value, if we have enough GPU memory. Reduce this to fit onto GPU memory.
        'sWin' : (2, 6, 6),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'num_swin_block' : 30,
        'nLevels' : 8,
        'increasing' : True,
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [6, 12, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
        'skippable' : True,
        'skip_learnable' : True,
        'qkv_bias' : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs
    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_26Jan18"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Mike1",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 1e-4,
        "max_lr": 1e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        #----------------------------------- SWIN ------------------------
        # 'factor10' : 2 on Windows else 10.   3 works on 8GB, 10 will work on 80GB? Yes, barely.
        
        'patch_kernel' : 6,
        'patch_stride' : 6,
        'swin_dim' : 192,    # ??? Reduce this to fit onto GPU memory.
        'sWin' : (2, 4, 4),  # Adjacency, locality of physical interactions require 2, preventing short-cut learning.
        'type_pos_bias' : 'lonBand', # 'sphere', 'None'
        'pair_bias_dim' : 5,
        'depth_list' : [6, 12, 4],   # [6, 8, 4] leads to OOP on A100
        'head_list' : [4, 8, 16],
        'skippable' : True,
        'skip_learnable' : True,
        'qkv_bias' : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs
    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_26Jan30"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 20,     #
        "axial_loops" : 1,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs
    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_26Feb09"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 20,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs
    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_26Feb09_2"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 20,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : False,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs
    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_26Feb11"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 10,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs
    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_26Feb11_2"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 30,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs
    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

from modules.uSide import Reshape
from modules.Transformer_Modes import AttnConn, PEConn, PEType, Flash, PBias, TransForm

projects["Zeus_uNet_Feb12"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light_uNet",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 30,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs

        'block_dicts' : [
            # block and side
            {
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (2, 2)
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.Nothing,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 4,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
        ]

    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}


projects["Zeus_uNet_Feb14"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light_uNet",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 30,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs

        'block_dicts' : [
            # block and side
            {   #---------------------------------- block 1
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (2, 2)
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 2
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 3
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_D_1X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 4
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_D_1X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 5
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_D_1X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 6
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.Nothing,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
        ]

    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_uNet_Feb15"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light_uNet",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 30,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs

        'block_dicts' : [
            # block and side
            {   #---------------------------------- block 1
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (2, 2)
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 2
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 3
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 4
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 5
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 6
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.Nothing,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
        ]

    }, 
    
    "model_1": {},
    "model_2": {},
    "model_3": {},
    "model_4": {},
}

projects["Zeus_uNet_Feb16"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light_uNet",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 30,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs

        'block_dicts' : [
            # block and side
            {   #---------------------------------- block 1
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (2, 2)
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 2
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 3
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 4
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.No),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 5
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.No),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 6
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.Nothing,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.No),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            # {   #---------------------------------- block 7
            #     # managed: 'top_res', 'top_dim'
            #     'top_mersplit': Reshape.Nothing,
            #     'down_mersplit': Reshape.Nothing,
            #     'square' : {
            #         # managed: 'res', 'dim'
            #         'num_heads' : 8,
            #         'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
            #         'nLayers' : 3,
            #         'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
            #     },
            # },
        ]

    }, 
}


projects["Zeus_uNet_Feb17"] = {    #---------------------------------------------------------- Micro-model for a single 8GB GPU -------
    # ------------------------------- This overrides the above rules ----------------------
    "management": {
    "Zeus_Model_module" : "Zeus_Light_uNet",
    "GPUWaitsData" : True,
    "models_to_play": [0,],
    "metrics" : "rmse", # "mae", "rmse"
    # Let it be chosen by GPU rule 'nSubstepsPerStep' : 2,       # Can be changed upto the type of GPU.
    "nSubstepsPerStep": 2,
    'max_cascade_depth' : 1,     # max = nSubstepsPerStep. 1 = no actual cascades, raw prediction only
    },
    
    "data": {
    "org_height": 721,
    "batch_size": 1,
    'lossXtargets' : 5.0,
    },
    
    "model_0": {
        "lr": 5e-4,
        "max_lr": 5e-4,
        "overwrite_saved_lr": False,
        "accum_steps" : 10,   
        "skip_validation" : True,
        
        'patch_kernel' : 4,
        'patch_stride' : 4,

        "batch_size": 1,
        
        "attn_dim" : 40,
        "nHeads" : 5,       #
        "nLayers" : 30,     #
        "axial_loops" : 1,
        "axial_not_total_pe" : True,
        
        'choose_best_loss_model' : False,
        'useLRLearner' : False,
        'use_checkpoint' : False,
        
        # The current model goes OOM on A100, with layers_to_enhance_in_enc_dec = [0] and nEnhancementPairs = [1]
        'layers_to_enhance_in_enc_dec' : [],
        'nEnhancementPairs': [],  # pairs

        'block_dicts' : [
            # block and side
            {   #---------------------------------- block 1
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (2, 2)
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 2
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
                },
            },
            {   #---------------------------------- block 3
                # Flash.Yes is required when AttnConn.Global, as Flash.No slows down 10x.
                # AttnConn.Shift is impossible, as it requiers attn_mask, which is not compatible with Flash.Yes.
             
                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 4
                # Shape: arount 23/45
                # Flash.No doens't degradate speed.

                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.No),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 5
                # Shape: arount 12/24
                # Flash.No doens't degradate speed.

                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.No),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 6
                # Shape: arount 6/12
                # Flash.No doens't degradate speed.

                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.R_Any_MHA, 'down_nWin': [1, 1], 'down_pWin': 1,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.No),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
            {   #---------------------------------- block 7
                # Shape: arount 3/6
                # Flash.No doens't degradate speed.

                # managed: 'top_res', 'top_dim'
                'top_mersplit': Reshape.Nothing,
                'down_mersplit': Reshape.Nothing,
                'square' : {
                    # managed: 'res', 'dim'
                    'num_heads' : 8,
                    'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                    'nLayers' : 3,
                    'transForm': TransForm(AttnConn.Global, PEConn.Global, PEType.Learnable, Flash.No),     # Flash.No: 2.72 s, Flash.No 1.99 s
                },
            },
        ]

    }, 
}

#-------------------------------- select proejct -----------------------

project = "Zeus_uNet_Feb16" # "Zeus_26Feb09" # "Zeus_uNet_Feb12"   # "Zeus_Swin_25Dec25_2"
config["bayesProject"] = project
# updata 'management'
config.update(projects[project]["management"])
# updata 'data
config.update(projects[project]["data"])
# update 'model_x'
for id in range(1):
    key = "model_" + str(id)
    config[key].update(projects[project][key])

print("bayesProject:", config["bayesProject"])
