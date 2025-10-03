import json
import csv
import logging
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

from pathlib import Path

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.Trainer import Trainer
from modules.params import Params
from modules.utils import check_dir

# Enable live updating of plots
# plt.ion()

class MLBayesianOptimizer:
    def __init__(self, config, root_path, pbounds, random_state=42):
        """
        trainer    : Your ML trainer instance (must have train(hp_dict) -> score)
        pbounds    : Dict of hyperparameter bounds
        save_path  : Path to save optimizer state (pickle)
        log_path   : Path to CSV log file
        """
        self.pm = Params(config)
        self.bayes_path = root_path / 'bayes' / self.pm.bayesProject
        self.trainer = Trainer(config, root_path, self.bayes_path, bayes=self)
        self.pbounds = pbounds
        self.save_path = self.bayes_path / 'bayes_res.json'
        self.log_path = self.bayes_path / 'bayes_log.csv'
        self.random_state = random_state

        self.finished = False

        # self.bayes_optimizer = self._load()     # deferred

        # Prepare CSV log file if needed
        if self.log_path and not os.path.exists(self.log_path):
            check_dir(self.log_path.parent)
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = list(self.pbounds.keys()) + ["score"]
                writer.writerow(header)

        # Initialize plot
        # self.fig, self.ax = plt.subplots(figsize=(6, 4))
        print("ptach*patch*nChannels, heads, dim, dim_head", self.pm.patch_dim, self.pm.nHeads, self.pm.dim, self.pm.dim_head)

    def _objective(self, **params):     # a part of self.bayes_optimizer
        # print('_objective', params)
        score = self.trainer.bayesian_target(*params.values())  # Higher is better
        self._log(params, score)
        # self._save()  # self.trainer calls this at every end of epoch
        # self._plot_progress()
        return score

    def maximize(self, init_points=2, n_iter=25, acq='ei'):
        self.bayes_optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=self.pbounds,
            random_state=self.random_state
        )
        logging.info("A new Bayesian optimizer is created.")

        createNew = False
        try:
            if self.save_path:
                with open(self.save_path, "rb") as f:
                    res = json.load(f)
                    for r in res:
                        self.bayes_optimizer.register(params=r["params"], target=r["target"])                    
                    # Note _save is called at the end of each training epoch.
                    logging.info("The Bayesian optimizer is updated with a saved one.")
        except:
            createNew = True

        if createNew:
            self._save_bayes_optimizer()

        self.bayes_optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
            # acq=acq
        )
        return self.bayes_optimizer.max

    def on_epoch_end(self):     # Only take care of yourself
        self._save_bayes_optimizer()

    def _save_bayes_optimizer(self):
        if self.save_path:
            with open(self.save_path, "w") as f:
                # .res is the list of test reports.
                # A test report is like {'target': 0.5,   'params': {'x': 0.1, 'y': 0.8}}
                json.dump(self.bayes_optimizer.res, f, indent=4)

    def _plot_progress(self):
        if not self.log_path:
            return
        df = pd.read_csv(self.log_path)

        self.ax.clear()
        self.ax.plot(df["score"], marker="o", label="Score")
        self.ax.set_xlabel("Trial")
        self.ax.set_ylabel("Score")
        self.ax.set_title("Bayesian Optimization Progress")
        self.ax.grid(True)
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _log(self, params, score):
        if self.log_path:
            with open(self.log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                row = [params[k] for k in self.pbounds.keys()] + [score]
                writer.writerow(row)

    @property
    def best_params(self):
        return self.bayes_optimizer.max["params"]

    @property
    def best_score(self):
        return self.bayes_optimizer.max["target"]
    

if __name__ == "__main__":

    from models.config import config, root_path

    from pathlib import Path
    import platform
    if platform.system() == "Windows":
        nFrames = config['nFrames']
        mFrames = config['mFrames']
        radius = config['radius']
        factor10 = config['factor10']
        cnn_hidden_channels = config['cnn_hidden_channels']
        
        pbounds = {     # Mike!!!
            "nFrames": (nFrames, nFrames, int),
            "mFrames": (mFrames, mFrames, int),
            "radius": (radius, radius, int),
            "factor10": (factor10, factor10, int),
            "cnn_hidden_channels": (len(cnn_hidden_channels), len(cnn_hidden_channels), int),
        }
    else:
        nFrames = config['nFrames']
        mFrames = config['mFrames']
        radius = config['radius']
        factor10 = config['factor10']
        cnn_hidden_channels = config['cnn_hidden_channels']
        
        pbounds = {     # Mike!!!
            "nFrames": (nFrames, nFrames, int),
            "mFrames": (mFrames, mFrames, int),
            "radius": (radius, radius, int),
            "factor10": (factor10, factor10, int),
            "cnn_hidden_channels": (len(cnn_hidden_channels), len(cnn_hidden_channels), int),
        }
        
    bayes = MLBayesianOptimizer(config, root_path, pbounds, random_state=523)

    best = bayes.maximize(init_points=3, n_iter=10)     # Mike!!!

    print("Best parameters:", bayes.best_params)
    print("Best score:", bayes.best_score)