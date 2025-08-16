import os, sys
import json
import math
import pickle
import logging

import torch
from datetime import datetime
import pytz, tzlocal
from pathlib import Path

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import modules.utils as utils
from modules.params import Params
from modules_zeus.constants import ERA5_DATA_VARS

class TrainEnvironment:
    def __init__(self, config, root_path, model, train_steps):
        super(TrainEnvironment, self).__init__()

        self.pm = Params(config)
        self.model = model
        self.train_steps = train_steps

        self.bayes_path = root_path / 'bayes' / self.pm.bayesProject
        # utils.check_dir(self.bayes_path)  not yet

        configPath =  self.bayes_path / 'config.json'
        utils.check_dir(configPath.parent)
        with open(configPath, 'w') as f:
            json.dump(config, f, indent=4)

        self.setup_logging(self.bayes_path)

        utils.set_seed(self.pm.seed)
        self.device = self._acquire_device()
        self.optimizer, self.scheduler = self._select_optimizer()

        self.era5_data_vars = self.load_era5_data_vars()
        self.rmses_old = torch.ones_like(self.era5_data_vars) * 1e9

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

        log_path = os.path.join(log_dir, 'log.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler (append mode)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)

        logging.info(f"Logging started. Outputting to console and {log_path}")

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

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
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

    def save_best(self, train_rmses, vali_rmses):   # (C,), (C,)
        rmses = 0.3 * train_rmses + 0.7 * vali_rmses
        if torch.mean(rmses) < torch.mean(self.rmses_old):
            logging.info(f"saving best model... score: {torch.mean(rmses).item()}")
            self.rmses_old = rmses
            self._save_train_state(name='best_state')

    def _save_train_state(self, epoch=0, name='train_state'):  # called at the end of epoch
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'rmses_old': self.rmses_old,
        }
        utils.check_dir(self.bayes_path)
        torch.save(checkpoint, self.bayes_path / (name + '.pth'))
        return


    def _load_train_state(self, name='train_state'):   # get ready to run the next epoch
        try:
            checkpoint_path = self.bayes_path / (name + '.pth')
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model parameters
            epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])

            # Ensure optimizer & scheduler exist before loading states
            if not hasattr(self, 'optimizer') or not hasattr(self, 'scheduler'):
                self._select_optimizer()

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.rmses_old = checkpoint['rmses_old']
            print('rmses restored: ', self.rmses_old)

            print(f"Checkpoint '{name}' loaded to {self.device}")
        except:
            epoch = -1

        return epoch

