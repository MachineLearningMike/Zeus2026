import os, sys
import json

import torch
from datetime import datetime
import pytz, tzlocal
from pathlib import Path
import logging

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import modules.utils as utils
from modules.params import Params
from modules_zeus.constants import ERA5_DATA_VARS

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

        rmses = 0.3 * train_rmses + 0.7 * vali_rmses
        rmses = rmses * self.era5_data_vars.to(self.device)

        if torch.mean(rmses) < torch.mean(self.rmses_old):
            wasBestEpoch = True
            logging.info("saving best model... avg rmse: {:.3f}".format(torch.mean(rmses).item()))
            self.rmses_old = rmses
            self._save_train_state(name='best_state')

        self.nWasNotBestEpoch = self.nWasNotBestEpoch + 1 if not wasBestEpoch else 0
        
        keep_go_next_epoch = self.nWasNotBestEpoch < self.pm.epoch_patience
        best_rmse_yet = torch.mean(self.rmses_old).item()

        return keep_go_next_epoch, best_rmse_yet

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

            print(f"Checkpoint '{name}' loaded to {self.device}")
        except:
            # Everything else has its default value.
            epoch = -1

        return epoch

