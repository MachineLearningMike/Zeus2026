import torch
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.Train import Train
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
        
    def run_train(self, start_epoch=0):
        keep_go_next_epoch = True
        best_rmse_yet = 1e10

        for epoch in range(start_epoch, self.pm.epochs):
            print("---------- epoch {} ------------".format(epoch))
            self.train.set_train_mode()
            train_loss, train_rmses = self.train.run_epoch(epoch, "train_set", train=True)   # (,), (C,)
            # self.scheduler.step()   # Commented out, because our scheduler is OneCycleLR, which is a per-batch scheduler

            if epoch >= 0:              # Mike!!!
                keep_go_next_epoch, best_rmse_yet = self.validate(epoch, train_loss, train_rmses)
            self.train.save_state()
            if self.bayes is not None:  self.bayes.on_epoch_end()

            if not keep_go_next_epoch: break

        logging.info("Train finishes or breaks after epoch: {}".format(epoch))
        self.train.save_state(train_finished=True)

        return - best_rmse_yet

    def validate(self, epoch, train_loss, train_rmses):
        self.train.set_eval_mode()
        with torch.no_grad():
            vali_loss, vali_rmses = self.train.run_epoch(epoch, "vali_set", train=False)     # (,), (C,)

        train_loss = train_loss.item()
        vali_loss = vali_loss.item()

        keep_go_next_epoch, best_rmse_yet = self.train.save_best_train(train_rmses, vali_rmses) # (C,), (C,)

        train_rmses = ", ".join(f"{x:.2f}" for x in train_rmses.ravel().tolist())
        vali_rmses = ", ".join(f"{x:.2f}" for x in vali_rmses.ravel().tolist())

        logging.info("Epoch: {} | Train Loss: {:.4f} Vali Loss: {:.4f} train RMSE: {} vali RMSE: {}\n".format(
            epoch, train_loss, vali_loss, train_rmses, vali_rmses))
        
        return keep_go_next_epoch, best_rmse_yet

    def test(self):
        self.train.set_eval_mode()
        T1 = self.config('T1'); FG = self.config('FG')

        epoch = 0
        test_loss, test_rmses = self.train.run_epoch(epoch, "test_set", train=False)

        logging.info("Test finished: vali rmse:{:.4f}".format(test_rmses.numpy()))
        
        return test_loss, test_rmses

    def bayesian_target(self, patch_size, heads, attLayers, zeusLayers):
        # Update with new hyper-parameters
        self.config['patch_size'] = patch_size
        self.config['heads'] = heads
        self.config['attLayers'] = attLayers
        self.config['zeusLayers'] = zeusLayers
        logging.info("Trainer is configured with new hyper-parameters. patch_size {}, heads {}, attLayers {}, zeusLayers {}" \
                    .format(patch_size, heads, attLayers, zeusLayers))
        
        self.config['hypers'] = {'patch_size': patch_size, 'heads': heads, 'attLayers': attLayers, 'zeusLayers': zeusLayers}

        self.pm = Params(self.config)
        self.train = Train(self.config, self.root_path, self.bayes_path)

        # Load: epoch, self.model state, env.optimizer, env.scheduler
        # If epoch == -1, they've not been loaded/updated.
        # If the previous train has finished, epoch == -1
        epoch = self.train.try_load_state()

        profit = self.run_train(start_epoch=epoch+1)

        return profit

if __name__ == "__main__":
    from models.config import config, root_path

    bases_path = root_path / 'bayes' / 'test'
    trainer = Trainer(config, root_path, bases_path, bayes=None)

    # x = torch.rand(1, config['T1'] * config['FG'], config['channels'], config['height'], config['width'])
    # print(x.shape)
    # pred = trainer.model(x)
    # print(pred.shape)
    patch_size = config['patch_size']
    heads = config['heads']
    attLayers = config['attLayers']
    zeusLayers = config['zeusLayers']
    profit = trainer.bayesian_target(patch_size, heads, attLayers, zeusLayers) 
