import torch
from types import SimpleNamespace

class Model: 
    def __init__(self): pass   

class Params:
    def __init__(self, config):
        
        #============================ project (data, models) ===============================
        self.dtype = torch.float16 if config['dtype'] == 16 else torch.float32 if config['dtype'] == 32 else None

        self.bayesProject = config['bayesProject']

        self.seed = config['seed']
        self.steps_to_save = config['steps_to_save']
        self.log_step = config['log_step']

        self.GPUWaitsData = config['GPUWaitsData']
        
        self.abundant_GPUs = config['abundant_GPUs']
        self.doubleZeus = config['doubleZeus']

        self.dual_drive = config['dual_drive']
        self.metrics = config['metrics']

        self.freeze_zeus1 = config['freeze_zeus1']
        self.skip_net_for_zeus2 = config['skip_net_for_zeus2']       

        self.dataset_path = config['dataset_path']
        self.management_path = config['management_path']
        
        self.models_to_play = config['models_to_play']

        #============================ data, shared ===========================================================================
        self.variables = config['variables']
        self.targetVariables = config['targetVariables']
        self.hourly_offset = config['hourly_offset']
        self.modulo = config['modulo']
        self.nCascades = config['nCascades']
        
        self.height = int(config['org_height'] * config['factor10'] / 10)       # int not round , config['org_height'] = 721 (transformer) or 720 (CNN)
        self.width = int(config['org_width'] * config['factor10'] / 10)         # int not round , config['org_width'] = 1440
        self.pole_h_lines = int(self.height/180*config['pole_angle'])
        self.factor10 = config['factor10']
        self.data_years = config['data_years']
        self.batch_size = config['batch_size']
        
        self.data_limit = config['data_limit']
        
        self.FG = config['FG']
        self.T1 = config['T1']

        self.nVariables = len(config['variables'])
        self.nFramesIn = config['T1'] * config['FG']                # 2 * 24 = 48           # temporal seq_len

        self.nFramesOut = config['T2'] * config['FG']                          # 24
        self.nFrames = config['nFrames']
        self.mFrames = config['mFrames']

        self.train_pct = config['train_pct']
        self.vali_pct = config['vali_pct']
        
        self.include_difficulty = config['include_difficulty']
        
        self.train_start_dt = config['train_start_dt']
        self.vali_start_dt = config['vali_start_dt']
        self.test_start_dt = config['test_start_dt']
        self.test_end_dt_inclusive = config['test_end_dt_inclusive']
            
        #============================ model, individual ========================================================================

        self.models = [] 

        for id in range(5):
            key = 'model_' + str(id)
            model_config_dict = config[key]

            obj = SimpleNamespace(**config[key])
            self.models.append(obj)

            # model_params_objectt = Model()
            
            
            
            
            # self.models.append(model_config_dict)
                            
            # # Transformer            
            # model_params_objectt.nHeads = model_config_dict['nHeads']
            # model_params_objectt.heads_per_chunk = model_config_dict['heads_per_chunk']
            # model_params_objectt.tokens_per_chunk = model_config_dict['tokens_per_chunk']
            # model_params_objectt.kv_full_sequence = model_config_dict['kv_full_sequence'] 
            # model_params_objectt.layerFFN = model_config_dict['layerFFN']
            # model_params_objectt.nLayers = model_config_dict['nLayers']
            # model_params_objectt.zeusBlocks = model_config_dict['zeusBlocks']        
            # model_params_objectt.crossLayers = model_config_dict['crossLayers']
            # model_params_objectt.PE = model_config_dict['PE']
            # model_params_objectt.learn_PE = model_config_dict['learn_PE']
            # model_params_objectt.use_PE = model_config_dict['use_PE']
            # model_params_objectt.patch_size = model_config_dict['patch_size']                      # 16
            
            # model_params_objectt.nPatches_h = self.height // model_params_objectt.patch_size      # 720 / 16 = 45
            # model_params_objectt.nPatches_w = self.width // model_params_objectt.patch_size       # 1,440 / 16 = 90
            # model_params_objectt.nPatches = model_params_objectt.nPatches_h * model_params_objectt.nPatches_w           # 45 * 90 = 4,050       # spatial seq_len
            # model_params_objectt.patch_dim = self.nVariables * model_params_objectt.patch_size ** 2      # 16 * 16 * 5 = 1,280
            # model_params_objectt.d_model = model_params_objectt.patch_dim
            # model_params_objectt.dim_head = int(model_params_objectt.d_model / model_config_dict['nHeads'])

            # # CNN
            # model_params_objectt.add_spatial_conv = model_config_dict['add_spatial_conv']        
            # model_params_objectt.blocksPerState = model_config_dict['blocksPerState']
            # model_params_objectt.ObsorbHDistortion = model_config_dict['ObsorbHDistortion']
            # model_params_objectt.Zeus_Model_module = model_config_dict['Zeus_Model_module']
            # model_params_objectt.EncLayer = model_config_dict['EncLayer']        
            # model_params_objectt.nCNNBlocks = model_config_dict['nCNNBlocks']        
            # model_params_objectt.bottleneckRate = model_config_dict['bottleneckRate']           
            
            # model_params_objectt.dropout = model_config_dict['dropout']
            # model_params_objectt.dropPath = model_config_dict['dropPath']
            # model_params_objectt.dim_layerFFN = model_params_objectt.d_model * model_config_dict['dim_to_layerFFN_dim']
            
            # model_params_objectt.radius = model_config_dict['radius']
            # model_params_objectt.transient_steps = model_config_dict['transient_steps']

            # model_params_objectt.optimizer = model_config_dict['optimizer']
            # model_params_objectt.lr = model_config_dict['lr']
            # model_params_objectt.bottleneck_dropout = model_config_dict['bottleneck_dropout']
            # model_params_objectt.skip_validation = model_config_dict['skip_validation']        
            # model_params_objectt.epochs = model_config_dict['epochs']
            # model_params_objectt.overwrite_saved_lr = model_config_dict['overwrite_saved_lr']
            # model_params_objectt.max_lr = model_config_dict['max_lr']
            # model_params_objectt.total_steps = model_config_dict['total_steps']
            # model_params_objectt.weight_decay = model_config_dict['weight_decay']
            # model_params_objectt.pct_start = model_config_dict['pct_start']
            # model_params_objectt.anneal_strategy = model_config_dict['anneal_strategy']
            # model_params_objectt.div_factor = model_config_dict['div_factor']
            # model_params_objectt.final_div_factor = model_config_dict['final_div_factor']
            # model_params_objectt.accum_steps = model_config_dict['accum_steps']
            # model_params_objectt.use_autocast = model_config_dict['use_autocast']
            # model_params_objectt.epoch_patience = model_config_dict['epoch_patience']
            # model_params_objectt.useLRLearner = model_config_dict['useLRLearner']
            
            # # SWIN
            # model_params_objectt.patch_kernel = model_config_dict['patch_kernel']
            # model_params_objectt.patch_stride = model_config_dict['patch_stride']
            # model_params_objectt.swin_dim = model_config_dict['swin_dim']
            # model_params_objectt.window_size = model_config_dict['window_size']
            # model_params_objectt.num_swin_block = model_config_dict['num_swin_block']
            
def flatten_pm_single(pm, i):
    # Extract all top-level attributes except 'models'
    base_attrs = {k: v for k, v in pm.__dict__.items() if k != 'models'}
    
    # Merge with the chosen model's attributes
    merged_attrs = {**base_attrs, **pm.models[i].__dict__}
    
    # Return as a SimpleNamespace for attribute access
    return SimpleNamespace(**merged_attrs)
        
if __name__ == "__main__":

    import os, sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from models.config_Mike1 import config
    
    params = Params(config)
    # print(params.models[0])
    for k, v in params.models[0].__dict__.items():
        print(f"{k}: {v}")
