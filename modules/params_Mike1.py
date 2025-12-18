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
        
        self.train_hours_limit = config['train_hours_limit']

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
        
        self.snapshot_pair_on_cuda = config['snapshot_pair_on_cuda']
        self.pair_not_single_snapshot = config['pair_not_single_snapshot']
            
        #============================ model, individual ========================================================================

        self.models = [] 

        for id in range(5):
            key = 'model_' + str(id)
            model_config_dict = config[key]

            obj = SimpleNamespace(**config[key])
            self.models.append(obj)
            
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
