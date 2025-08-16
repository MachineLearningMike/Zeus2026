
import numpy as np
import torch
from torch import nn

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pathlib import Path
import logging

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.Losses import Losses
from modules.environment import TrainEnvironment
from modules.zeus_tqdm import zeus_tqdm
import modules.metrics
from modules.zeus_dataset import Zeus_Dataset
from modules.params import Params
from Zeus import Zeus


class Trainer():

    def __init__(self, config, root_path, bayes=None):
        self.bayes = bayes
        self.config = config
        self.root_path = root_path

        self.dataset = Zeus_Dataset(config, root_path)
        # self._initialize(config, self.root_path)
        return

    def _initialize_train(self, config):
        self.pm = Params(config)
        print("-------- patch_dim: {}, dim: {}, dim_head: {}, nHeads: {}".format(self.pm.patch_dim, self.pm.dim, self.pm.dim_head, self.pm.nHeads))

        self.model = Zeus(config).float()
        train_set, _, _ = self.dataset.generate_x_sets(seed=self.pm.seed)  
        self.env = TrainEnvironment(config, self.root_path, self.model, len(train_set))
        self.losses = Losses(self.env, self.pm.variables)

        self.train_loss_ema = 0.0
        self.vali_loss_ema = 0.0
        self.ema_alpha = 2 / (10 + 1)   # We want EMA similar to 1 10-day simple moving average (SME).
        return
    
    def _run_epoch(self, epoch, x_set_label, train=False):  
        self.model.to(self.env.device)

        mean = rearrange(torch.tensor(self.dataset.mean), 'c -> c 1 1').to(self.env.device)
        std = rearrange(torch.tensor(self.dataset.std), 'c -> c 1 1').to(self.env.device)

        self.dataset.select_x_set(x_set_label, seed=self.pm.seed)
        pbar = zeus_tqdm(self.dataset)
        pbar.start_prefetch_thread()
        
        FG = self.pm.FG; T1 = self.pm.nFramesIn // FG
        epoch_loss = []; epoch_rmses = []

        if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Fetch one batch asynchronously
        for batch in pbar:              # Wait for the asynchronous .prefetch
            # print('got data!')
            if batch is None: break     # just in case
            batch_x, batch_y, batch_diff, era5_data_vars = batch    # (B, T1FG, C, H, W), (B, T2FG, C, H, W), (B, C, H, W), (C,)
            
            no_more_data = pbar.no_more_data()
            if not no_more_data:  pbar.prefetch(x_set_label)    # Fetch one batch asynchronously

            # Mike!!! Remove this block in production version.
            batch_x = batch_x[:, :, :, :self.pm.image_height, :self.pm.image_width]
            batch_y = batch_y[:, :, :, :self.pm.image_height, :self.pm.image_width]
            if self.pm.include_difficulty:
                batch_diff = batch_diff[:, :, :self.pm.image_height, :self.pm.image_width]

            T2 = (batch_y.shape)[1] // FG
            batch_x, batch_y = batch_x.float().to(self.env.device), batch_y.float().to(self.env.device)

            (B, T1FG, C, H, W) = batch_x.shape
            pred_y = torch.zeros(size=(B,0,C,H,W)).to(self.env.device)
            step_loss = []; step_rmses = []

            for t2 in range(T2):
                if train: self.env.optimizer.zero_grad()                       # train
                base_x = torch.concat([batch_x, pred_y], dim=1)
                base_x = base_x[:, -T1*FG:, :, :, :]     #(B, T1FG, C, H, W)
                pred = self.model(base_x)     # (B,FG,C,H,W)
                # pred_y = torch.concat((pred_y, pred), dim=1).to(self.env.device)
                pred_y = torch.concat((pred_y, pred.detach()), dim=1)
                target = batch_y[:, t2*FG:(t2+1)*FG, :,:].to(self.env.device)   # (B,FG,C,H,W)
                # print('1', pred.shape, std.shape, mean.shape)

                pred = pred * std + mean
                target = target * std + mean
                # loss = torch.sqrt(torch.mean((pred - target) ** 2))
                # loss, rmse = self.losses.loss(target, pred, batch_diff.to(self.env.device), era5_data_vars.to(self.env.device))
                loss, rmse = self.losses.loss(target, pred, batch_diff, era5_data_vars.to(self.env.device)) # (,), (C,)

                step_loss.append(loss)  # (,) - list
                step_rmses.append(rmse)      # (C,) - list
                if train: loss.backward()                         # train
                if train: self.env.optimizer.step()               # train
          
            # Conclude a step
            if train: self.env.scheduler.step()
            step_loss = torch.tensor(step_loss).mean()   # (,)
            step_rmses = torch.stack(step_rmses, dim=0).mean(dim=0)   # (C,)
            strC = ", ".join(f"{x:.2f}" for x in step_rmses.ravel().tolist())
            pbar.set_description('Loss: {:.3f}, rmses: {}'.format(step_loss.item(), strC))

            epoch_loss.append(step_loss)    # (,) - list
            epoch_rmses.append(step_rmses)   # (C,) - list
            if no_more_data: break
            # print('waiting for data...')

        epoch_loss = torch.stack(epoch_loss, dim=0).mean(dim=0)  # (,)
        epoch_rmses = torch.stack(epoch_rmses, dim=0).mean(dim=0)       # (C,)
        strC = ", ".join(f"{x:.2f}" for x in epoch_rmses.ravel().tolist())
        pbar.set_description('Loss: {:.3f}, rmses: {}'.format(epoch_loss.item(), strC))
        pbar.shutdown()

        return epoch_loss, epoch_rmses  # (,), (C,)
        #--------------------------------------------------------------------------------

    def _validate(self, epoch, train_loss, train_rmses):    # (,), (C,)
        self.model.eval()
        with torch.no_grad():
            vali_loss, vali_rmses = self._run_epoch(epoch, "vali_set", train=False)     # (,), (C,)

        # if epoch % (self.pm.log_step * 100) == 0: self.env._save(name=str(epoch))


        train_loss = train_loss.item()
        vali_loss = vali_loss.item()

        self.env.save_best(train_rmses, vali_rmses) # (C,), (C,)

        train_rmses = ", ".join(f"{x:.2f}" for x in train_rmses.ravel().tolist())
        vali_rmses = ", ".join(f"{x:.2f}" for x in vali_rmses.ravel().tolist())

        logging.info("Epoch: {} | Train Loss: {:.4f} Vali Loss: {:.4f} train RMSE: {} vali RMSE: {}\n".format(
            epoch, train_loss, vali_loss, train_rmses, vali_rmses))
        
        self.train_loss_ema = self.ema_alpha * train_loss + (1-self.ema_alpha) * self.train_loss_ema
        self.vali_loss_ema = self.ema_alpha * vali_loss + (1-self.ema_alpha) * self.vali_loss_ema

        go_next_epoch = True
        
        return go_next_epoch


    def train(self, start_epoch=0):
        broken = False
        for epoch in range(start_epoch, self.pm.epochs):
            print("---------- epoch {} ------------".format(epoch))
            self.model.train()
            train_loss, train_rmses = self._run_epoch(epoch, "train_set", train=True)   # (,), (C,)

            go_next_epoch = True
            if epoch >= 1:              # Mike!!!
                go_next_epoch = self._validate(epoch, train_loss, train_rmses)
            if not go_next_epoch:
                broken = True; break

            self.env._save_train_state(epoch=epoch)
            if self.bayes is not None:  self.bayes.on_epoch_end()

        if broken:
            if self.bayes is not None:  self.bayes.on_epoch_end()

        return # - latest ma loss, bayesian optimization.

    def test(self):
        self.model.eval()
        T1 = self.config('T1'); FG = self.config('FG')

        epoch = 0
        epoch_loss = self._run_epoch(epoch, "test_set", train=False)

        self.env.print_log('vali rmse:{:.4f}'.format(epoch_loss.item()))
        
        return epoch_loss
    
    # def bayesian_target(self, attLayers, zeusLayers):
    #     import time
    #     time.sleep(3)
    #     print('1')
    #     time.sleep(3)
    #     print('2')
    #     self.bayes.on_epoch_end()
    #     time.sleep(3)
    #     print('3')
    #     return attLayers + zeusLayers

    def bayesian_target(self, patch_size, heads, attLayers, zeusLayers):
        # Update with new hyper-parameters
        self.config['patch_size'] = patch_size
        self.config['heads'] = heads
        self.config['attLayers'] = attLayers
        self.config['zeusLayers'] = zeusLayers
    
        # Initialize: self.pm, self.model, self.env, self.loss, and state variables
        self._initialize_train(self.config)

        # Load: epoch, self.model state, env.optimizer, env.scheduler
        # If epoch == -1, they've not been loaded/updated.
        epoch = self.env._load_train_state()

        profit = self.train(start_epoch=epoch+1)

        return profit

        


# def inference(model, lat, lon, start_hour, hours):
#     # sketch only
#     x, t_date = get_latest_truth_data()     # x: (T1,C,H,W)
#     steps = get_steps(t_date, start_hour, hours)

#     batch_x = x.unsqueeze(0)    # (B=1,T1,C,H,W)

#     pred_list = []
#     for step in range(steps):
#         pred_y = model(batch_x.to(self.device), pred_list)    # (B,SG,C,H,W). t is implicit.
#         pred_list.append(pred_y)
    
#     response = get_response(pred_list, t_date, lat, lon, start_hour, hours)
#     response = format_zeuse(response)

#     return response

if __name__ == "__main__":

    from models.config import config

    import platform
    if platform.system() == "Windows":  root_path = Path("D:/Zeus_Data")
    else:   root_path = Path("/mnt/Data/Zeus_Data")

    trainer = Trainer(config, root_path, bayes=None)

    # x = torch.rand(1, config['T1'] * config['FG'], config['channels'], config['height'], config['width'])
    # print(x.shape)
    # pred = trainer.model(x)
    # print(pred.shape)

    heads = config['heads'], attLayers, zeusLayers = config['attLayers'], config['zeusLayers']
    profit = trainer.bayesian_target(attLayers, zeusLayers)