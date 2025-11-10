import json
import csv
import logging
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

from pathlib import Path

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.Train import Train
# from models.Trainer import Trainer
from modules.params import Params
from modules.utils import check_dir

# Enable live updating of plots
# plt.ion()

class MLBayesianOptimizer:
    def __init__(self, config, pbounds, random_state=42):
        """
        trainer    : Your ML trainer instance (must have train(hp_dict) -> score)
        pbounds    : Dict of hyperparameter bounds
        save_path  : Path to save optimizer state (pickle)
        log_path   : Path to CSV log file
        """
        
        self.pm = Params(config)        
        dataset_path = Path(self.pm.dataset_path)
        management_path = Path(self.pm.management_path)
        check_dir(management_path)
        self.project_path = management_path / "projects" / self.pm.bayesProject
        check_dir(self.project_path)
        self.save_path = self.project_path / 'bayes_res.json'
        self.log_path = self.project_path / 'bayes_log.csv'

        self.project_path.mkdir(parents=True, exist_ok=True)

        #---------- Save config ---------
        configPath =  self.project_path / ('params') # + '-' + self.get_train_id_string() +'.json')
        check_dir(configPath.parent)
        with open(configPath, 'w') as f:
            json.dump(config, f, indent=4)

        # self.trainer = Trainer(self.pm, dataset_path, management_path, self.project_path, bayes=self)
        self.train = Train(self.pm, dataset_path, management_path, self.project_path, bayes=self, trainable=True)
        self.pbounds = pbounds
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
        print("ptach*patch*nChannels, nHeads, dim, dim_head", self.pm.patch_dim, self.pm.nHeads, self.pm.d_model, self.pm.dim_head)

    def _objective(self, **params):     # a part of self.bayes_optimizer
        score = self.train.train()
        self._log(params, score)
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

    from models.config import config

    from pathlib import Path
    import platform
    if platform.system() == "Windows":
        
        patch_size = config['patch_size']
        nHeads = config['nHeads']
        heads_per_chunk = config['heads_per_chunk']
        tokens_per_chunk = config['tokens_per_chunk']
        nLayers = config['nLayers']
        layerFFN = config['layerFFN']
                        
        pbounds = {     # Mike!!!
            # "patch_size": (patch_size, patch_size, int),
            # "nHeads": (nHeads, nHeads, int),
            # "nLayers": (nLayers, nLayers, int),
            # # "layerFFN" : (layerFFN, layerFFN, str),
        }
    else:
        patch_size = config['patch_size']
        nHeads = config['nHeads']
        heads_per_chunk = config['heads_per_chunk']
        tokens_per_chunk = config['tokens_per_chunk']
        nLayers = config['nLayers']
        layerFFN = config['layerFFN']
                        
        pbounds = {     # Mike!!!
            # "patch_size": (patch_size, patch_size, int),
            # "nHeads": (nHeads, nHeads, int),
            # "nLayers": (nLayers, nLayers, int),
            # # "layerFFN" : (layerFFN, layerFFN, str),
        }
        
    bayes = MLBayesianOptimizer(config, pbounds, random_state=523)

    best = bayes.maximize(init_points=3, n_iter=10)     # Mike!!!

    print("Best parameters:", bayes.best_params)
    print("Best score:", bayes.best_score)