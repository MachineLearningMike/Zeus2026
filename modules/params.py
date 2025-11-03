import torch
import math

def nearest_multiple_of_3_ge(p, h):
    x0 = math.ceil(p / h) * h      # ceiling multiple of h ≥ p
    remainder = x0 % 3
    if remainder == 0:
        x = x0
    else:
        x = x0 + (3 - remainder)   # round up to next multiple of 3
    return x

class Params:
    def __init__(self, config):
        self.dtype = torch.float16 if config['dtype'] == 16 else torch.float32 if config['dtype'] == 32 else None
        
        self.height = int(config['org_height'] * config['factor10'] / 10)       # int not round , config['org_height'] = 721 (transformer) or 720 (CNN)
        self.width = int(config['org_width'] * config['factor10'] / 10)         # int not round , config['org_width'] = 1440
        self.variables = config['variables']
        self.nVariables = len(config['variables'])
        self.T1 = config['T1']
        self.nFramesIn = config['T1'] * config['FG']                # 2 * 24 = 48           # temporal seq_len
        self.FG = config['FG']
        self.nFramesOut = config['T2'] * config['FG']                          # 24
        
        self.nFrames = config['nFrames']
        self.mFrames = config['mFrames']
        self.radius = config['radius']
        self.factor10 = config['factor10']
        self.transient_steps = config['transient_steps']

        self.train_pct = config['train_pct']
        self.vali_pct = config['vali_pct']
        
        self.add_spatial_conv = config['add_spatial_conv']

        self.patch_size = config['patch_size']                      # 16
        self.nPatches_h = self.height // self.patch_size      # 720 / 16 = 45
        self.nPatches_w = self.width // self.patch_size       # 1,440 / 16 = 90
        self.nPatches = self.nPatches_h * self.nPatches_w           # 45 * 90 = 4,050       # spatial seq_len

        self.skip_validation = config['skip_validation']
        self.patch_dim = self.nVariables * self.patch_size ** 2      # 16 * 16 * 5 = 1,280

        self.d_model = self.patch_dim
        self.dim_head = int(self.d_model / config['nHeads'])
        
        self.nHeads = config['nHeads']
        self.heads_per_chunk = config['heads_per_chunk']
        self.tokens_per_chunk = config['tokens_per_chunk']
        self.kv_full_sequence = config['kv_full_sequence']        
        self.dropout = config['dropout']
        self.dropPath = config['dropPath']
        
        self.nLayers = config['nLayers']
        self.zeusBlocks = config['zeusBlocks']
        self.ObsorbHDistortion = config['ObsorbHDistortion']
        self.Zeus_Model_module = config['Zeus_Model_module']
        self.EncLayer = config['EncLayer']
        
        self.PE = config['PE']
        self.learn_PE = config['learn_PE']
        self.ObsorbHDistortion = config['ObsorbHDistortion']

        self.nCNNBlocks = config['nCNNBlocks']
        self.bottleneckRate = config['bottleneckRate']        
        
        self.crossLayers = config['crossLayers']
        self.layerFFN = config['layerFFN']
        self.dim_layerFFN = self.d_model * config['dim_to_layerFFN_dim']
        
        self.cnn_hidden_channels = config['cnn_hidden_channels']
        self.blocksPerState = config['blocksPerState']

        #----------------- Training ---------------
        self.include_difficulty = config['include_difficulty']
        self.epochs = config['epochs']
        self.steps_to_save = config['steps_to_save']
        self.batch_size = config['batch_size']
        self.log_step = config['log_step']
        self.seed = config['seed']
        self.log_step = config['log_step']
        self.optimizer = config['optimizer']
        self.lr = config['lr']
        self.overwrite_saved_lr = config['overwrite_saved_lr']
        self.max_lr = config['max_lr']
        self.total_steps = config['total_steps']
        self.weight_decay = config['weight_decay']
        self.pct_start = config['pct_start']
        self.anneal_strategy = config['anneal_strategy']
        self.div_factor = config['div_factor']
        self.final_div_factor = config['final_div_factor']
        self.epoch_patience = config['epoch_patience']
        self.useLRLearner = config['useLRLearner']
        self.accum_steps = config['accum_steps']
        self.use_autocast = config['use_autocast']
        
        self.data_years = config['data_years']
        
        self.seed = config['seed']
        self.bayesProject = config['bayesProject']
        
        self.use_PE = config['use_PE']
        
        self.abundant_GPUs = config['abundant_GPUs']
        self.dual_drive = config['dual_drive']
        self.root_path = config['root_path']
        
        self.doubleZeus = config['doubleZeus']
        self.freeze_zeus1 = config['freeze_zeus1']
        self.skip_net_for_zeus2 = config['skip_net_for_zeus2']
        
        # if self.height % self.patch_size != 0:
        #     raise Exception("Image height must be divisible by the patch size: {}, {}".format(self.height, self.patch_size))
        # if self.width % self.patch_size != 0:
        #     raise Exception("Image height must be divisible by the patch size: {}, {}".format(self.width, self.patch_size))