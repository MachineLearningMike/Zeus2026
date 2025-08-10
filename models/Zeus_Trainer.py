
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pathlib import Path

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.env import env
from modules.recorder import Recorder
import modules.metrics
from modules.zeus_dataset import Zeus_Dataset
# from modules_zeus.era5_cds import Era5CDSLoader
from Zeus import Params, Zeus


class Trainer():

    def __init__(self, config, root_dir, param_path, split_dates_path, variables):
        self.pm = Params(config)

        self.model = Zeus(config)
        self.dataset = Zeus_Dataset(config, root_dir, param_path, split_dates_path, variables)

        self.dataset.load_dataset_params()
        self.dataset.split_data_dates(train_pct=0.5, vali_pct=0.3, seed=523)
        self.dataset.generate_x_sets(seed=523)       
        # batch = self.dataset.take_batch('vali_set', batch_size=3)

        self.train_loss_ema = 0.0
        self.vali_loss_ema = 0.0
        self.ema_alpha = 2 / (10 + 1)   # We want EMA similar to 1 10-day simple moving average (SME).


    def _run_epoch(self, data_loader, train=False):
        T1 = self.config('T1'); FG = self.config('FG')

        #================================== epoch main ===============================
        epoch_loss = []
        pbar = tqdm(env.train_loader)

        new_T2 = env.get_next_steps()
        data_loader.prefetch(new_T2)        # Fetch one batch asynchronously

        for batch_x, batch_y in pbar:         # Wait for the asynchronous .prefetch
            batch_x = batch_x[:, :, :, :-1, :]      # 721 -> 720
            batch_y = batch_y[:, :, :, :-1, :]      # 721 -> 720

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
        
        train_loss = train_loss.item()
        vali_loss = vali_loss.item()

        env.print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
            epoch + 1, train_loss.item(), vali_loss.item()))
        tape_recorder(vali_loss.item(), self.model, env.path)

        self.train_loss_ema = self.ema_alpha * train_loss + (1-self.ema_alpha) * self.train_loss_ema
        self.vali_loss_ema = self.ema_alpha * vali_loss + (1-self.ema_alpha) * self.vali_loss_ema

        go_next_epoch = True
        
        return go_next_epoch


    def train(self, data_loader, args):
        config = args.__dict__
        tape_recorder = Recorder(verbose=True)
        
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
        'T1': 5,    # 5 later.
        'MaxT2': 5,
        'FG': 24,   # frames in a frame group
        # patch size
        'patch_size': 8,
        # transformer
        'heads': 8,
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'patch_dim_to_dim' : 1,
        'dim_to_mlp_dim': 2,
        # layers
        'attLayers' : 1,
        'zeusLayers': 6,
    }

    trainer = Trainer(config, root_dir, param_path, split_dates_path, variables)

    x = torch.rand(1, config['T1'] * config['FG'], config['channels'], config['height'], config['width'])
    print(x.shape)
    pred = trainer.model(x)
    print(pred.shape)