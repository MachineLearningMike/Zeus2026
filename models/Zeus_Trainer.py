
import numpy as np
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pathlib import Path
import logging

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.utils import check_dir
from models.Losses import Losses
from modules.environment import TrainEnvironment
from modules.zeus_tqdm import zeus_tqdm
import modules.metrics
from modules.zeus_dataset import Zeus_Dataset
from modules.params import Params
from Zeus import Zeus


class Trainer():

    def __init__(self, config, root_path, bayes_path, bayes=None):
        self.config = config
        self.bayes = bayes

        self.bayes_path = bayes_path
        self.setup_logging(self.bayes_path)

        self.dataset = Zeus_Dataset(config, root_path)
        logging.info("dataset is created.")
        
        self.scaler = GradScaler(device='cuda')
        return
    
    def setup_logging(self, log_dir):
        """
        logging.info("This is an info message")
        logging.warning("Watch out! This is a warning")
        logging.error("An error happened")
        logging.debug("Debug details here (won't show unless level is DEBUG)")
        """
        # Remove all handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_path = log_dir / 'log.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler (append mode)
        check_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)


    def _initialize_train(self, config):
        self.pm = Params(config)
        logging.info("patch_dim: {}, dim: {}, dim_head: {}, nHeads: {}".format(self.pm.patch_dim, self.pm.dim, self.pm.dim_head, self.pm.nHeads))

        self.model = Zeus(config).float()
        logging.info("Zeus model is created.")
        train_set, _, _ = self.dataset.generate_x_sets(seed=self.pm.seed)
        logging.info("x_sets are generated.")
        self.env = TrainEnvironment(config, self.bayes_path, self.model, len(train_set))
        self.losses = Losses(self.env, self.pm.variables)
        return
    
    def _print_tensor_state(self, n, t):
        print(n, t.dtype, t.device, t.requires_grad, t.shape)
    
    def _run_epoch(self, epoch, x_set_label, train=False):  
        self.model.to(self.env.device)
        
        self.dataset.refresh_x_set(x_set_label, seed=self.pm.seed)
        logging.info("{} is refreshed.".format(x_set_label))
        pbar = zeus_tqdm(self.dataset)
        pbar.start_prefetch_thread()
        
        FG = self.pm.FG; T1 = self.pm.nFramesIn // FG
        epoch_loss = []; epoch_rmses = []

        if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Fetch one batch asynchronously
        for batch in pbar:              # Wait for the asynchronous .prefetch
            # print('got data!')
            if train: self.env.optimizer.zero_grad()                     # train
            
            if batch is None: break     # just in case
            batch_x, batch_y, batch_diff, era5_data_vars = batch    # (B, T1FG, C, H, W), (B, T2FG, C, H, W), None or (B, C, H, W), (C,)
            # self._print_tensor_state(1, batch_x)    # 32, cpu, False
            # self._print_tensor_state(2, batch_y)    # 32, cpu, False
            # self._print_tensor_state(3, era5_data_vars) # 32, cpu, False
            if self.pm.include_difficulty: self._print_tensor_state(4, batch_diff)
            no_more_data = pbar.no_more_data()
            if not no_more_data:  pbar.prefetch(x_set_label)    # Fetch one batch asynchronously

            batch_x = batch_x[:, :, :, :self.pm.image_height, :self.pm.image_width]     # torch.float32, cpu
            batch_y = batch_y[:, :, :, :self.pm.image_height, :self.pm.image_width]     # torch.float32, cpu
            era5_data_vars = era5_data_vars # torch.float32, cpu
            self._print_tensor_state(5, batch_x)        # 32, cpu, False
            self._print_tensor_state(6, batch_y)        # 32, cpu, False
            self._print_tensor_state(7, era5_data_vars) # 32, cpu, False

            if self.pm.include_difficulty:
                batch_diff = batch_diff[:, :, :self.pm.image_height, :self.pm.image_width]
                self._print_tensor_state(8, batch_diff)

            T2 = (batch_y.shape)[1] // FG
            batch_x = batch_x.to(self.env.device)               # torch.float32, cuda
            batch_y = batch_y.float().to(self.env.device)       # torch.float32, cuda
            era5_data_vars = era5_data_vars.to(self.env.device) # torch.float32, cuda
            self._print_tensor_state(9, batch_x)            # 32, cuda, False
            self._print_tensor_state(10, batch_y)           # 32, cuda, False
            self._print_tensor_state(11, era5_data_vars)    # 32, cuda, False

            (B, T1FG, C, H, W) = batch_x.shape
            pred_y = torch.zeros(size=(B,0,C,H,W)).to(self.env.device)  # torch.float32, cuda
            self._print_tensor_state(12, pred_y)            # 32, cuda, False
            step_loss = []; step_rmses = []

            for t2 in range(T2):
                with autocast(device_type='cuda'):
                    self._print_tensor_state(13, batch_x)   # 32, cuda, False
                    self._print_tensor_state(14, pred_y)    # 32, cuda, False
                    base_x = torch.cat([batch_x, pred_y], dim=1)
                    base_x = base_x[:, -T1*FG:, :, :, :]     #(B, T1FG, C, H, W), torch.float32, cuda, requires_grad False
                    self._print_tensor_state(15, base_x)    # 32, cuda, False
                    pred = self.model(base_x)     # (B,FG,C,H,W), torch.float16 !!!, cuda, requires_grad True !!!
                    self._print_tensor_state(16, pred)      # 16 !!!, cuda, True !!!
                    pred_y = torch.cat([pred_y, pred.detach()], dim=1) # torch.float32, cuda, requires_grad False
                    self._print_tensor_state(17, pred_y)    # 32, cuda, False
                    target = batch_y[:, t2*FG:(t2+1)*FG, :,:].to(self.env.device)   # (B,FG,C,H,W)  # torch.float32, cuda, requries_grad False
                    self._print_tensor_state(18, target)    # 32, cuda, False
                    pred = self.dataset.unnormalize(pred)   # torch.float32 !!!, cuda, requires_grad True !!!
                    self._print_tensor_state(19, pred)      # 32, cuda, True !!!
                    target = self.dataset.unnormalize(target)   # torch.float32, cuda, requires_grad False
                    self._print_tensor_state(20, target)    # 32, cuda, False
                    loss, rmse = self.losses.loss(target, pred, batch_diff, era5_data_vars.to(self.env.device)) # (,), (C,), torch.float32, cuda, requries_grad True !!!
                    self._print_tensor_state(21, loss)      # 32, cuda, True !!!
                    self._print_tensor_state(22, rmse)      # 32, cuda, True !!!
                    step_loss.append(loss.cpu().detach())  # (,) - list.        Even if no cpu(), their stack results in cpu !!!
                    step_rmses.append(rmse.cpu().detach())      # (C,) - list.  If no cpu(), their stack results in cuda

                if train:
                    loss = loss / T2    # Normalize loss, so gradients match what we'd get if we did a one big forwards pass.
                    self.scaler.scale(loss).backward()          # accumulate gradients T2 times by backpropagating (scaled) loss

            if train: 
                self.scaler.step(self.env.optimizer)            # (un-scalsed) step for accumulated gradients.
                self.scaler.update()                            # This is the only place to adjust the scale factor.
          
            # Conclude a step
            self.env.scheduler.step()   # Our scheduler is OneCycleLR, which is a per-batch scheduler
            step_loss = torch.tensor(step_loss).mean()   # (,)
            self._print_tensor_state(23, step_loss)         # 32, cpu False
            step_rmses = torch.stack(step_rmses, dim=0).mean(dim=0)   # (C,)
            self._print_tensor_state(24, step_rmses)        # 32, cpt False
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

        train_loss = train_loss.item()
        vali_loss = vali_loss.item()

        keep_go_next_epoch, best_rmse_yet = self.env.save_best_epoch(train_rmses, vali_rmses) # (C,), (C,)

        train_rmses = ", ".join(f"{x:.2f}" for x in train_rmses.ravel().tolist())
        vali_rmses = ", ".join(f"{x:.2f}" for x in vali_rmses.ravel().tolist())

        logging.info("Epoch: {} | Train Loss: {:.4f} Vali Loss: {:.4f} train RMSE: {} vali RMSE: {}\n".format(
            epoch, train_loss, vali_loss, train_rmses, vali_rmses))
        
        return keep_go_next_epoch, best_rmse_yet


    def train(self, start_epoch=0):
        keep_go_next_epoch = True
        best_rmse_yet = 1e10

        for epoch in range(start_epoch, self.pm.epochs):
            print("---------- epoch {} ------------".format(epoch))
            self.model.train()
            train_loss, train_rmses = self._run_epoch(epoch, "train_set", train=True)   # (,), (C,)
            # self.env.scheduler.step()   # Commented out, because our scheduler is OneCycleLR, which is a per-batch scheduler

            if epoch >= 0:              # Mike!!!
                keep_go_next_epoch, best_rmse_yet = self._validate(epoch, train_loss, train_rmses)
            self.env._save_train_state(epoch=epoch)
            if self.bayes is not None:  self.bayes.on_epoch_end()

            if not keep_go_next_epoch: break

        logging.info("Train finishes or breaks after epoch: {}".format(epoch))
        self.env._save_train_state(epoch=epoch, train_finished=True)

        return - best_rmse_yet

    def test(self):
        self.model.eval()
        T1 = self.config('T1'); FG = self.config('FG')

        epoch = 0
        epoch_loss = self._run_epoch(epoch, "test_set", train=False)

        self.env.print_log('vali rmse:{:.4f}'.format(epoch_loss.item()))
        
        return epoch_loss
    
    # def bayesian_target(self, patch_size, heads, attLayers, zeusLayers):
    #     import time
    #     time.sleep(2)
    #     print('1')
    #     time.sleep(2)
    #     print('2')
    #     self.bayes.on_epoch_end()
    #     time.sleep(2)
    #     print('3')
    #     return attLayers + zeusLayers

    def bayesian_target(self, patch_size, heads, attLayers, zeusLayers):
        # Update with new hyper-parameters
        self.config['patch_size'] = patch_size
        self.config['heads'] = heads
        self.config['attLayers'] = attLayers
        self.config['zeusLayers'] = zeusLayers
        logging.info("Trainer is configured with new hyper-parameters. patch_size {}, heads {}, attLayers {}, zeusLayers {}" \
                    .format(patch_size, heads, attLayers, zeusLayers))

        # Initialize: self.pm, self.model, self.env, self.loss, and state variables
        self._initialize_train(self.config)

        # Load: epoch, self.model state, env.optimizer, env.scheduler
        # If epoch == -1, they've not been loaded/updated.
        # If the previous train has finished, epoch == -1
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

    from models.config import config, root_path

    bases_path = root_path / 'bayes' / 'test'
    trainer = Trainer(config, root_path, bases_path, bayes=None)

    # x = torch.rand(1, config['T1'] * config['FG'], config['channels'], config['height'], config['width'])
    # print(x.shape)
    # pred = trainer.model(x)
    # print(pred.shape)

    heads = config['heads'], attLayers, zeusLayers = config['attLayers'], config['zeusLayers']
    profit = trainer.bayesian_target(attLayers, zeusLayers)