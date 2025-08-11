import os, sys
import json
import math
import pickle
import logging

import numpy as np
import torch
from tqdm import tqdm

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import modules.utils as utils
from modules.params import Params

class Environment:
    def __init__(self, config, model, train_steps):
        super(Environment, self).__init__()

        self.pm = Params(config)
        self.model = model
        self.train_steps = train_steps

        self.ex_path = os.path.join(self.pm.ex_folder, self.pm.ex_name)
        utils.check_dir(self.ex_path)

        configPath = os.path.join(self.ex_path, 'config.json')
        with open(configPath, 'w') as f:
            json.dump(config, f, indent=4)

        self.checkpoints_path = os.path.join(self.ex_path, 'checkpoints')
        utils.check_dir(self.checkpoints_path)

        self.setup_logging(self.ex_path)

        utils.set_seed(self.pm.seed)
        self.device = self._acquire_device()
        self.optimizer, self.scheduler = self._select_optimizer()

        self.best_score, self.delta_score = None, 0.001

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

    def save_best(self, vali_loss):
        score = -vali_loss
        if self.best_score is None:
            self.best_score = score
            self._save(name=self.pm.ex_name)
        elif score >= self.best_score + self.delta_score:
            self.best_score = score
            self._save(name=self.pm.ex_name)

    def _save(self, name=''):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.checkpoints_path, name + '.pth'))
        return


    def _load(self, name=''):
        checkpoint_path = os.path.join(self.checkpoints_path, name + '.pth')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model parameters
        self.model.load_state_dict(checkpoint['model'])

        # Ensure optimizer & scheduler exist before loading states
        if not hasattr(self, 'optimizer') or not hasattr(self, 'scheduler'):
            self._select_optimizer()

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        print(f"Checkpoint '{name}' loaded to {device}")
        return

