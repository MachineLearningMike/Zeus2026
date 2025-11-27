import torch
from types import SimpleNamespace

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
        
        #============================ model, individual ========================================================================

        self.models = []
        class Model: 
            def __init__(self): pass    

        for id in range(5):
            key = 'model_' + str(id)
            params = config[key]

            model = Model()
            self.models.append(model)
                            
            # Transformer            
            model.nHeads = params['nHeads']
            model.heads_per_chunk = params['heads_per_chunk']
            model.tokens_per_chunk = params['tokens_per_chunk']
            model.kv_full_sequence = params['kv_full_sequence'] 
            model.layerFFN = params['layerFFN']
            model.nLayers = params['nLayers']
            model.zeusBlocks = params['zeusBlocks']        
            model.crossLayers = params['crossLayers']
            model.PE = params['PE']
            model.learn_PE = params['learn_PE']
            model.use_PE = params['use_PE']
            model.patch_size = params['patch_size']                      # 16
            
            model.nPatches_h = self.height // model.patch_size      # 720 / 16 = 45
            model.nPatches_w = self.width // model.patch_size       # 1,440 / 16 = 90
            model.nPatches = model.nPatches_h * model.nPatches_w           # 45 * 90 = 4,050       # spatial seq_len
            model.patch_dim = self.nVariables * model.patch_size ** 2      # 16 * 16 * 5 = 1,280
            model.d_model = model.patch_dim
            model.dim_head = int(model.d_model / params['nHeads'])

            # CNN
            model.add_spatial_conv = params['add_spatial_conv']        
            model.blocksPerState = params['blocksPerState']
            model.ObsorbHDistortion = params['ObsorbHDistortion']
            model.Zeus_Model_module = params['Zeus_Model_module']
            model.EncLayer = params['EncLayer']        
            model.nCNNBlocks = params['nCNNBlocks']        
            model.bottleneckRate = params['bottleneckRate']           
            
            model.dropout = params['dropout']
            model.dropPath = params['dropPath']
            model.dim_layerFFN = model.d_model * params['dim_to_layerFFN_dim']
            
            model.radius = params['radius']
            model.transient_steps = params['transient_steps']

            model.optimizer = params['optimizer']
            model.lr = params['lr']
            model.bottleneck_dropout = params['bottleneck_dropout']
            model.skip_validation = params['skip_validation']        
            model.epochs = params['epochs']
            model.overwrite_saved_lr = params['overwrite_saved_lr']
            model.max_lr = params['max_lr']
            model.total_steps = params['total_steps']
            model.weight_decay = params['weight_decay']
            model.pct_start = params['pct_start']
            model.anneal_strategy = params['anneal_strategy']
            model.div_factor = params['div_factor']
            model.final_div_factor = params['final_div_factor']
            model.accum_steps = params['accum_steps']
            model.use_autocast = params['use_autocast']
            model.epoch_patience = params['epoch_patience']
            model.useLRLearner = params['useLRLearner']
            
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
    from models.config_14nov25 import config
    
    params = Params(config)
    # print(params.models[0])
    for k, v in params.models[0].__dict__.items():
        print(f"{k}: {v}")
