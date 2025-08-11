
import numpy as np
import torch
from torch import nn

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pathlib import Path
import logging

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.environment import Environment
from modules.zeus_tqdm import zeus_tqdm
import modules.metrics
from modules.zeus_dataset import Zeus_Dataset
# from modules_zeus.era5_cds import Era5CDSLoader
from modules.params import Params
from Zeus import Zeus


class Trainer():

    def __init__(self, config, root_dir, param_path, split_dates_path, variables):
        self.pm = Params(config)

        self.model = Zeus(config).float()
        self.dataset = Zeus_Dataset(config, root_dir, param_path, split_dates_path, variables)
        train_steps = self.dataset.initialize()
        self.env = Environment(config, self.model, train_steps)

        self.train_loss_ema = 0.0
        self.vali_loss_ema = 0.0
        self.ema_alpha = 2 / (10 + 1)   # We want EMA similar to 1 10-day simple moving average (SME).


    def _run_epoch(self, x_set_label, train=False):  
        self.model.to(self.env.device)           
        self.dataset.initialize_x_set(x_set_label, seed=523)
        pbar = zeus_tqdm(self.dataset)
        FG = self.pm.FG; T1 = self.pm.nFramesIn // FG
        epoch_loss = []

        if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Fetch one batch asynchronously
        # batch_x, batch_y, batch_diff, self.era5_data_vars     # (B, T1FG, C, H, W), (B, T2FG, C, H, W), (B, 1, C, H, W), (C,)
        for batch_x, batch_y, batch_diff, era5_data_vars in pbar:         # Wait for the asynchronous .prefetch
            if not pbar.no_more_data():  pbar.prefetch(x_set_label)    # Fetch one batch asynchronously

            # THIS BLOCK is for scale-down test only, where self.pm.image_height, width are smaller than 720 and 1440.
            batch_x = batch_x[:, :, :, :self.pm.image_height, :self.pm.image_width]
            batch_y = batch_y[:, :, :, :self.pm.image_height, :self.pm.image_width]
            batch_diff = batch_diff[:, :, :, :self.pm.image_height, :self.pm.image_width]

            T2 = (batch_y.shape)[1] // FG
            batch_x, batch_y = batch_x.float().to(self.env.device), batch_y.float().to(self.env.device)

            (B, T1FG, C, H, W) = batch_x.shape
            pred_y = torch.zeros(size=(B,0,C,H,W)).to(self.env.device)
            step_loss = []

            for t2 in range(T2):
                if train: self.env.optimizer.zero_grad()                       # train
                base_x = torch.concat([batch_x, pred_y], dim=1)
                base_x = base_x[:, -T1*FG:, :, :, :]     #(B, T1FG, C, H, W)
                pred = self.model(base_x)     # (B,FG,C,H,W)
                # pred_y = torch.concat((pred_y, pred), dim=1).to(self.env.device)
                pred_y = torch.concat((pred_y, pred.detach()), dim=1)
                target = batch_y[:, t2*FG:(t2+1)*FG, :,:].to(self.env.device)
                loss = torch.sqrt(torch.mean((pred - target) ** 2))
                step_loss.append(loss)
                if train: loss.backward()                         # train
                if train: self.env.optimizer.step()                            # train
          
            # pred_y: (B,T2FG,C,H,W)
            step_loss = sum(step_loss)/len(step_loss)   # tensor
            epoch_loss.append(step_loss)
            pbar.set_description('Loss: {:.4f}'.format(step_loss.item()))

            if train: self.env.scheduler.step()                                # train
        
        epoch_loss = sum(epoch_loss)/len(epoch_loss)    # tensor
        pbar.shutdown()
        print('epoch finished')
        return epoch_loss
        #--------------------------------------------------------------------------------

    def _validate(self, epoch, train_loss):
        self.model.eval()
        with torch.no_grad():
            vali_loss = self._run_epoch("vali_set", train=False)

        # if epoch % (self.pm.log_step * 100) == 0: self.env._save(name=str(epoch))
        
        train_loss = train_loss.item()
        vali_loss = vali_loss.item()

        logging.info("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
            epoch + 1, train_loss, vali_loss))
        self.env.save_best(vali_loss)

        self.train_loss_ema = self.ema_alpha * train_loss + (1-self.ema_alpha) * self.train_loss_ema
        self.vali_loss_ema = self.ema_alpha * vali_loss + (1-self.ema_alpha) * self.vali_loss_ema

        go_next_epoch = True
        
        return go_next_epoch


    def train(self):
        for epoch in range(self.pm.epochs):
            self.model.train()

            print('------------- training epoch -----------')
            epoch_loss = self._run_epoch("train_set", train=True)

            print('-------------- validating----------')
            go_next_epoch = self._validate(epoch, epoch_loss)
            if not go_next_epoch: break

        # best_model_path = os.path.join(self.env.checkpoints_path, 'checkpoint.pth')
        # self.model.load_state_dict(torch.load(best_model_path))
        return # - latest ma loss, bayesian optimization.


    def test(self):
        self.model.eval()
        T1 = self.config('T1'); FG = self.config('FG')

        epoch_loss = self._run_epoch("test_set", train=False)

        self.env.print_log('vali rmse:{:.4f}'.format(epoch_loss.item()))
        
        return epoch_loss


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

    root_dir = Path("D:\Download\CDS")
    param_path = Path("D:\Download\dataset_params.json")
    split_dates_path = Path("D:\Download\split_dates.json")
    variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind']   # order matters.

    config = {
        # image h w c
        'height': 72,
        'width': 144,
        'channels': 4,
        # video length in and out
        'T1': 3,    # 5 later.
        'MaxT2': 5,
        'FG': 24,   # frames in a frame group
        # patch size
        'patch_size': 8, # 8
        # transformer
        'heads': 8,     # 8
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'patch_dim_to_dim' : 1,
        'dim_to_mlp_dim': 2,
        # layers
        'attLayers' : 1,
        'zeusLayers': 6,
        # training
        'seed' : 523,
        'epochs': 20,
        'log_step': 10,
        'optimizer': 'adam',
        'lr' : 1e-5,
        'max_lr' : 1e-3,
        'weight_decay': 0.01,
        'pct_start' : 0.3,
        'anneal_strategy' : 'linear',   # 'cos'
        'div_factor' : 25.0,
        'final_div_factor' : 1e4,
        'ex_folder' : 'D:\Download\experiments',
        'ex_name' : '1st',
    }


    trainer = Trainer(config, root_dir, param_path, split_dates_path, variables)

    x = torch.rand(1, config['T1'] * config['FG'], config['channels'], config['height'], config['width'])
    print(x.shape)
    pred = trainer.model(x)
    print(pred.shape)

    trainer.train()