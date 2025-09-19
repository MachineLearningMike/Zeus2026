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
        
        self.image_height = config['height']                        # 720
        self.image_width = config['width']                          # 1,440
        self.variables = config['variables']
        self.T1 = config['T1']
        self.nFramesIn = config['T1'] * config['FG']                # 2 * 24 = 48           # temporal seq_len
        self.FG = config['FG']
        self.nFramesOut = 1 * config['FG']                          # 24
        self.maxT2 = config['maxT2']

        self.train_pct = config['train_pct']
        self.vali_pct = config['vali_pct']

        self.patch_size = config['patch_size']                      # 16
        self.nPatches_h = self.image_height // self.patch_size      # 720 / 16 = 45
        self.nPatches_w = self.image_width // self.patch_size       # 1,440 / 16 = 90
        self.nPatches = self.nPatches_h * self.nPatches_w           # 45 * 90 = 4,050       # spatial seq_len
        self.nSequences = self.nFramesIn * self.nPatches            # 48 * 4,050 = 194,400

        self.nChannels = len(config['variables'])                         # 5
        self.patch_dim = self.nChannels * self.patch_size ** 2      # 16 * 16 * 5 = 1,280

        self.dim = self.patch_dim
        self.dim_head = int(self.dim / config['heads'])
        
        self.nClasses = self.nChannels
        self.nHeads = config['heads']
        self.dropout = config['dropout']
        self.attn_dropout = config['attn_dropout']
        self.drop_path = config['drop_path']
        
        self.attLayers = config['attLayers']
        self.zeusLayers = config['zeusLayers']
        self.mlp_dim = self.dim * config['dim_to_mlp_dim']

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
        self.max_lr = config['max_lr']
        self.weight_decay = config['weight_decay']
        self.pct_start = config['pct_start']
        self.anneal_strategy = config['anneal_strategy']
        self.div_factor = config['div_factor']
        self.final_div_factor = config['final_div_factor']
        self.epoch_patience = config['epoch_patience']

        self.hypers = config['hypers']
        self.seed = config['seed']
        self.bayesProject = config['bayesProject']
        
        self.abundant_GPUs = config['abundant_GPUs']
        
        self.doubleZeus = config['doubleZeus']
        self.freeze_zeus1 = config['freeze_zeus1']
        self.skip_net_for_zeus2 = config['skip_net_for_zeus2']
        
        if self.image_height % self.patch_size != 0:
            raise Exception("Image height must be divisible by the patch size: {}, {}".format(self.image_height, self.patch_size))
        if self.image_width % self.patch_size != 0:
            raise Exception("Image height must be divisible by the patch size: {}, {}".format(self.image_width, self.patch_size))