import torch
from datetime import datetime
import json
import pytz, tzlocal
from einops import rearrange
from pathlib import Path
import logging

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import modules.utils as utils
from modules.params import Params
from modules_zeus.constants import ERA5_DATA_VARS
from modules_zeus.constants import (
    REWARD_DIFFICULTY_SCALER,  
    REWARD_IMPROVEMENT_WEIGHT,
    REWARD_IMPROVEMENT_MIN_DELTA,
)

class TrainEnvironment:
    def __init__(self, config, bayes_path, model, train_steps):
        super(TrainEnvironment, self).__init__()

        self.pm = Params(config)
        self.model = model
        self.train_steps = train_steps

        self.bayes_path = bayes_path
        # utils.check_dir(self.bayes_path)  not yet

        configPath =  self.bayes_path / 'config.json'
        utils.check_dir(configPath.parent)
        with open(configPath, 'w') as f:
            json.dump(config, f, indent=4)

        utils.set_seed(self.pm.seed)
        self.device = self._acquire_device()
        self.optimizer, self.scheduler = self._select_optimizer()

        self.era5_data_vars = self.load_era5_data_vars()
        self.rmses_old = torch.ones_like(self.era5_data_vars) * 1e9
        self.nWasNotBestEpoch = 0

        #---------- Loss --------------
        # improve this line regarding missing variables.
        min_sota_delta = torch.tensor([value for (key, value) in REWARD_IMPROVEMENT_MIN_DELTA.items() if key in self.pm.variables])  # (C,)
        self.min_sota_delta = rearrange(min_sota_delta, 'c -> c 1 1')

        return
    
    def get_minute_stamp(self):
        local_tz = tzlocal.get_localzone()
        now = datetime.now(local_tz)
        stamp = now.strftime("%Y%m%d_%H%M")
        return stamp

    def load_era5_data_vars(self):
        vars = [ERA5_DATA_VARS[k] for k in self.pm.variables if k in ERA5_DATA_VARS.keys()]
        self.era5_data_vars = torch.tensor(vars)
        return self.era5_data_vars

    def _acquire_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Default to CPU
        # device = torch.device("cpu")
        
        # if self.args.use_gpu and torch.cuda.is_available():
        #     # Clamp GPU index to available range
        #     gpu_index = min(self.args.gpu, torch.cuda.device_count() - 1)
            
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        #     device = torch.device(f"cuda:{gpu_index}")
            
        #     utils.print_log(f"Using GPU: {gpu_index} ({torch.cuda.get_device_name(gpu_index)})")
        # else:
        #     if self.args.use_gpu:
        #         utils.print_log("GPU requested but not available — falling back to CPU.")
        #     else:
        #         utils.print_log("Using CPU.")
        
        return device

    
    def _select_optimizer(self):
        # Choose optimizer type, default to Adam
        optimizer_name = getattr(self.pm, 'optimizer', 'adam').lower()
        lr = getattr(self.pm, 'lr', 1e-3)
        weight_decay = getattr(self.pm, 'weight_decay', 0.01)

        if optimizer_name == 'adamw':
            OptimizerClass = torch.optim.AdamW
        elif optimizer_name == 'sgd':
            OptimizerClass = torch.optim.SGD
        else:
            OptimizerClass = torch.optim.Adam

        # At the time of calling this, self.model.parameters() may be on CPU.
        # This call let optmizer keep reference to the pararmeters.
        # Later, if the model is moved to GPU, the optimizer's references points to Cuda tensors.
        # If optimizer.step() happens after moving the model, the states will also be on Cuda 
        # because optimizer's state is created lazily when calling optimizer.step() first.
        optimizer = OptimizerClass(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Check train_loader length for scheduler
        steps_per_epoch = self.train_steps

        # Scheduler parameters with defaults, overridable by args
        max_lr = getattr(self.pm, 'max_lr', lr)
        pct_start = getattr(self.pm, 'pct_start', 0.3)
        epochs = getattr(self.pm, 'epochs', 10)
        anneal_strategy =  getattr(self.pm, 'anneal_strategy', 'linear')
        div_factor = getattr(self.pm, 'div_factor', 25.0)
        final_div_factor = getattr(self.pm, 'final_div_factor', 1e4)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(    # Mike!!! This is a per-batch scheduler.
            # A per-batch scheduler's time axis is on mini-batch, not on epochs.
            # It only counts how many mini-batches it traverses, across epochs, and doesn't aware of epochs.
            # steps_per_epoch and epochs are only used to get total_steps.
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            epochs=epochs,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

        logging.info(f"Optimizer selected: {optimizer_name.upper()} with lr={lr}, weight_decay={weight_decay}")
        logging.info(f"Scheduler: OneCycleLR with max_lr={max_lr}, epochs={epochs}, steps_per_epoch={steps_per_epoch}")

        return optimizer, scheduler

    def save_best_epoch(self, train_rmses, vali_rmses):   # (C,), (C,)
        wasBestEpoch = False
        device = train_rmses.device
        assert device == vali_rmses.device
        rmses = 0.3 * train_rmses + 0.7 * vali_rmses
        rmses = rmses * (self.era5_data_vars.to(device))

        if torch.mean(rmses) < torch.mean(self.rmses_old):
            wasBestEpoch = True
            self.rmses_old = rmses
            name = "best-" + "-".join([str(v) for v in self.pm.hypers.values()])
            logging.info("saving best model... avg rmse: {:.3f}, name: {}".format(torch.mean(rmses).item(), name))
            self._save_train_state(name=name)

        self.nWasNotBestEpoch = self.nWasNotBestEpoch + 1 if not wasBestEpoch else 0
        
        keep_go_next_epoch = self.nWasNotBestEpoch < self.pm.epoch_patience
        best_rmse_yet = torch.mean(self.rmses_old).item()

        return keep_go_next_epoch, best_rmse_yet
    
    def _initialize_train_state(self, config):  # trainer's responsibility
        pass

    def _save_train_state(self, epoch=0, name='train_state', train_finished=False):  # called at the end of epoch
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'rmses_old': self.rmses_old,
            'nWasNotBestEpoch': self.nWasNotBestEpoch,
            'train_finished' : train_finished,
        }
        utils.check_dir(self.bayes_path)
        torch.save(checkpoint, self.bayes_path / (name + '.pth'))
        print(f"Checkpoint '{name}' saved, including model, optimizer, scheduler, and rmses_old")
        
        return


    def _load_train_state(self, name='train_state'):   # get ready to run the next epoch
        try:
            checkpoint_path = self.bayes_path / (name + '.pth')
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            assert checkpoint['train_finished'] == False    # It raise an deliberate exception if True.

            # Load model parameters
            epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])

            # Ensure optimizer & scheduler exist before loading states
            if not hasattr(self, 'optimizer') or not hasattr(self, 'scheduler'):
                self.optimizer, self.scheduler = self._select_optimizer()

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.rmses_old = checkpoint['rmses_old']
            self.nWasNotBestEpoch = checkpoint['nWasNotBestEpoch']

            print(f"Checkpoint '{name}' loaded to {self.device}, including model, optimizer, scheduler, and rmses_old")
        except:
            # Everything else has its default value.
            epoch = -1

        return epoch

    #------------------------------ Loss -----------------------------------

    def _print_tensor_state(self, n, t):
        print(n, t.dtype, t.device, t.requires_grad, t.shape)

    def get_curved_scores(self, raw_rmse: torch.Tensor, gamma: float) -> torch.Tensor:
        # gamma: (B, C, H, W)
        device = raw_rmse.device
        min_rmse = rearrange(torch.tensor((0.1, 0.0, 0.4, 0.4)), 'c -> c 1 1').to(device)    # Mike!!! Rough. They come from w&d
        max_rmse = raw_rmse * 1.2   # Mike!!! A rouph assumption

        min_rmse = torch.minimum(min_rmse, raw_rmse)
        eq = (min_rmse == max_rmse)

        amin = torch.minimum(min_rmse, max_rmse)
        amax = torch.maximum(min_rmse, max_rmse)
        den  = (amax - amin)
        norm = (raw_rmse - amin) / torch.clamp(den, min=1e-12)
        norm = torch.where(eq, torch.ones_like(norm), norm)

        return torch.pow(norm, gamma)  # gamma.shape = (B, C, H, W)
    
    # def loss(self, target, pred, batch_diff, era5_data_vars):
    def loss(self, target, pred, batch_diff, era5_data_vars):
        # (B,FG,C,H,W), (B,FG,C,H,W), (B, C, H, W), (C,)
        # 32, cuda, False ; 32, cuda, True !!!, 32, cuda, False, 32, cuda, False
        device = pred.device
        rmse = ((pred - target) ** 2).mean(dim=1).sqrt()    # (B, C, H, W), 32, cuda, True !!!
        # self._print_tensor_state(1, target)                 # 
        # self._print_tensor_state(2, pred)
        # self._print_tensor_state(3, batch_diff)
        # self._print_tensor_state(4, era5_data_vars)
        # self._print_tensor_state(4.5, rmse)

        if batch_diff is not None:
            baseline_rmse = 0
            improvement = baseline_rmse - rmse  - self.min_sota_delta.to(device)   # (B, C, H, W), 32, cuda, True,  the-higher-the-better
            # self._print_tensor_state(5, improvement)
            baseline_improvement = torch.maximum(torch.tensor(0.0).to(device), improvement)    # 0.0, (B, C, H, W), 32, cuda, True !!!,  the-higher-the-better
            # self._print_tensor_state(6, baseline_improvement)
            gamma = torch.pow(REWARD_DIFFICULTY_SCALER, batch_diff * 2 - 1).to(device)  # (B, C, H, W), 32, cuda, False,  the-higher-the-more-difficult
            # self._print_tensor_state(7, gamma)
            quality_score = self.get_curved_scores(rmse, gamma)       # (B, C, H, W), 32, cuda, True !!!, the-higher-the-worse
            # self._print_tensor_state(8, quality_score)
            improvement_score = self.get_curved_scores(-baseline_improvement, gamma) # (B, C, H, W), 32, cuda, True !!!, the-higher-the-worse
            # self._print_tensor_state(10, improvement_score)
            rmse = (1 - REWARD_IMPROVEMENT_WEIGHT) * quality_score + REWARD_IMPROVEMENT_WEIGHT * improvement_score    # (B, C, H, W), 32, cuda, True !!!
            # self._print_tensor_state(11, rmse)
        
        reward = rmse * rearrange(era5_data_vars, 'c -> c 1 1')   # (B, C, H, W), 32, cuda, True !!!, the-higher-the-worse
        # self._print_tensor_state(12, reward)
        reward = reward.mean()  # (,)
        rmse = rmse.mean(dim=(0,2,3))   # (C,)

        return reward, rmse # (,), (C,)     32, cuda, True