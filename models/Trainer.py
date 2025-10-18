import torch
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# from models.Train_Chunk2D import Train
from models.Train_CNN import Train
import modules.utils as utils
from modules.params import Params

class Trainer():
    def __init__(self, config, root_path, bayes_path, bayes=None):
        self.config = config
        self.root_path = root_path
        self.bayes_path = bayes_path
        self.bayes = bayes
        self.setup_logging(bayes_path)

        self.pm = None
        self.train = None
    
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
        utils.check_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        
    def run_train(self, start_epoch=0, step_state=None):
        run = False
        keep_go_next_epoch = True
        best_rmse_yet = 1e10
            
        for epoch in range(start_epoch, self.pm.epochs):
            run = True
            print("---------- epoch {} ------------".format(epoch))
                        
            train_loss, train_metrics = self.train.run_epoch(epoch, "train_set", trainable=True, step_state=step_state)   # (,), (C,)
            step_state = None # Valid initially only.
            # self.scheduler.step()   # Commented out, because our scheduler is OneCycleLR, which is a per-batch scheduler
            display = self.train.get_epoch_display(train_loss, train_metrics)
            logging.info("Train epoch: {}, {}".format(epoch, display))
    
            if self.pm.skip_validation:              # Mike!!!
                logging.info("Validation skipped as configured with 'skip_validation'")
            else:
                keep_go_next_epoch, best_rmse_yet = self.validate(epoch, train_loss, train_metrics)

            if self.bayes is not None:  self.bayes.on_epoch_end()

            if not keep_go_next_epoch: break

        if run:
            self.train.save_state(train_finished=True)  # Train_save. Save no step_state. Do not take care of validate()
            logging.info("Train finishes or breaks after epoch: {}".format(epoch))

        return - best_rmse_yet

    def validate(self, epoch, train_loss, train_metrics):
        vali_loss, vali_metrics = self.train.run_epoch(epoch, "vali_set", trainable=False)     # (,), (C,)

        display = self.train.get_epoch_display(vali_loss, vali_metrics)
        logging.info("Vali epoch: {}, {}".format(epoch, display))

        keep_go_next_epoch, best_rmse_yet = self.train.save_best_train(train_metrics, vali_metrics) # (C,), (C,)
        
        return keep_go_next_epoch, best_rmse_yet

    def bayesian_target(self, patch_size, heads, heads_per_chunk, tokens_per_chunk, attLayers): #, layerFFN):
        # Update with new hyper-parameters
        self.config['patch_size'] = patch_size
        self.config['heads'] = heads
        self.config['heads_per_chunk'] = heads_per_chunk
        self.config['tokens_per_chunk'] = tokens_per_chunk
        self.config['attLayers'] = attLayers
        # self.config['layerFFN'] = layerFFN

        logging.info("Trainer configured with new hyper-parameters. patch_size {}, heads {}, chunks {}, attLayers {}, layerFFN {}" \
                    .format(patch_size, heads, heads_per_chunk, tokens_per_chunk, attLayers, self.config['layerFFN']))
        
        self.config['hypers'] = {'patch_size': patch_size, 'heads': heads, 'heads_per_chunk': heads_per_chunk, 'tokens_per_chunk': tokens_per_chunk, 'attLayers': attLayers, 'layerFFN': self.config['layerFFN']}
        self.pm = Params(self.config)

        self.train = Train(self.config, self.root_path, self.bayes_path, trainable=True)
        epoch, step_state = self.train.try_load_state()

        profit = self.run_train(start_epoch=epoch, step_state=step_state)

        return profit

if __name__ == "__main__":
    from models.config import config, root_path

