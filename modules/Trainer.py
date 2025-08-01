
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table

import env
import recorder
import metrics
from Zeus import Zeus

class Trainer():

    def __init__(self, config):
        self.image_height = config['height']                        # 720
        self.image_width = config['width']                          # 1,440
        self.nFramesIn = config['T1'] * config['FG']                # 2 * 24 = 48           # temporal seq_len
        self.nFramesOut = 1 * config['FG']                          # 24

        self.patch_size = config['patch_size']                      # 16
        self.nPatches_h = self.image_height // self.patch_size      # 720 / 16 = 45
        self.nPatches_w = self.image_width // self.patch_size       # 1,440 / 16 = 90
        self.nPatches = self.nPatches_h * self.nPatches_w           # 45 * 90 = 4,050       # spatial seq_len
        self.nSequences = self.nFramesIn * self.nPatches            # 48 * 4,050 = 194,400

        self.nChannels = config['channels']                         # 5
        self.patch_dim = self.nChannels * self.patch_size ** 2      # 16 * 16 * 5 = 1,280
        self.dim = self.patch_dim
        
        self.nClasses = self.nChannels
        self.nHeads = config['heads']
        self.dim_head = config['dim_head']
        self.dropout = config['dropout']
        self.attn_dropout = config['attn_dropout']
        self.drop_path = config['drop_path']
        self.scale_dim = config['scale_dim']

        self.model = Zeus(config)


    def _run_epoch(self, data_loader, train=False):
        T1 = self.config('T1'); FG = self.config('FG')

        #================================== epoch main ===============================
        epoch_loss = []
        pbar = tqdm(env.train_loader)

        new_T2 = env.get_next_steps()
        data_loader.prefetch(new_T2)        # Fetch one batch asynchronously

        for batch_x, batch_y in pbar:         # Wait for the asynchronous .prefetch
            T2 = (batch_y.shape)[1] // FG
            new_T2 = env.get_next_steps()
            data_loader.prefetch(new_T2)    # Fetch one batch asynchronously

            torch.cuda.synchronize()
            batch_x, batch_y = batch_x.to(env.device), batch_y.to(env.device)
            
            B, T1FG, C, H, W = batch_x.shape
            pred_y = torch.zeros(size=(B,0,C,H,W))
            step_loss = []

            for substep in range(T2):
                if train: env.optimizer.zero_grad()                       # train

                base_x = torch.concat((batch_x[:, substep*FG:, :,:,:], pred_y[:, max(substep-T1,0)*FG:, :,:,:]), dim=1)  #(B,T1FG,C,H,W)
                pred = self.model(base_x)     # (B,FG,C,H,W)
                pred_y = torch.concat((pred_y, pred), dim=1)

                target = batch_y[:, substep*FG:(substep+1)*FG, :,:]
                substep_loss = env.difficulty_rmse(pred, target)
                step_loss.append(substep_loss)
                if train: substep_loss.backward()                         # train
                if train: env.optimizer.step()                            # train
          
            # pred_y: (B,T2FG,C,H,W)
            step_loss = sum(step_loss)/len(step_loss)   # tensor
            epoch_loss.append(step_loss)
            pbar.set_description('train loss: {:.4f}'.format(step_loss.item()))

            if train: env.scheduler.step()                                # train
            if train and env.step % env.update_ema_every == 0: env.step_ema()     # train
            if train: env.step += 1                                               # train
        
        epoch_loss = sum(epoch_loss)/len(epoch_loss)    # tensor
        #--------------------------------------------------------------------------------

    def _validate(self, tape_recorder, epoch, train_loss, args):

        self.model.eval()
        with torch.no_grad():
            vali_loss = self._run_epoch(self, env.vali_loader, train=False)

        if epoch % (args.log_step * 100) == 0: env._save(name=str(epoch))
            
        env.print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
            epoch + 1, train_loss.item(), vali_loss.item()))
        tape_recorder(vali_loss.item(), self.model, env.path)
        go_next_epoch = True
        
        return go_next_epoch


    def train(self, data_loader, args):
        config = args.__dict__
        tape_recorder = recorder.Recorder(verbose=True)
        
        T1 = self.config('T1'); FG = self.config('FG')

        for epoch in range(config['epochs']):
            self.model.train()

            epoch_loss = self.run_epoch(self, data_loader, train=True)

            go_next_epoch = self._validate(tape_recorder, epoch, epoch_loss, args)
            if not go_next_epoch: break

        best_model_path = env.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return # - latest ma loss, bayesian optimization.


    def test(self, data_loader):
        self.model.eval()
        T1 = self.config('T1'); FG = self.config('FG')

        epoch_loss = self._run_epoch(self, env.test_loader, train=False)

        env.print_log('vali rmse:{:.4f}'.format(epoch_loss.item()))
        
        return epoch_loss


# def inference(model, lat, lon, start_hour, hours):
#     # sketch only
#     x, t_date = get_latest_truth_data()     # x: (T1,C,H,W)
#     steps = get_steps(t_date, start_hour, hours)

#     batch_x = x.unsqueeze(0)    # (B=1,T1,C,H,W)

#     pred_list = []
#     for step in range(steps):
#         pred_y = model(batch_x.to(env.device), pred_list)    # (B,SG,C,H,W). t is implicit.
#         pred_list.append(pred_y)
    
#     response = get_response(pred_list, t_date, lat, lon, start_hour, hours)
#     response = format_zeuse(response)

#     return response

if __name__ == "__main__":
    config = {
        # image h w c
        'height': 720,
        'width': 1440,
        'channels': 5,
        # video length in and out
        'T1': 2,    # 5 later.
        'MaxT2': 5,
        'FG': 24,   # frames in a frame group
        # patch size
        'patch_size': 16,
        # transformer
        'heads': 8,
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        # layers
        'attLayers' : 1,
        'zeusLayers': 6,
    }

    model = Zeus(config)