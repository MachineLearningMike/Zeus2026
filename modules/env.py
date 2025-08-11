import os
import json
import math
import pickle
import logging

import numpy as np
import torch
from tqdm import tqdm

import utils

class env:
    def __init__(self, args):
        super(env, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        utils.print_log(utils.output_namespace(self.args))

        # self._preparation() =======================
        # seed
        utils.set_seed(self.args.seed)
        # log and checkpoint
        self.path = os.path.join(self.args.res_dir, self.args.ex_name)
        utils.check_dir(self.path)

        self.checkpoints_path = os.path.join(self.path, 'checkpoints')
        utils.check_dir(self.checkpoints_path)

        sv_param = os.path.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=os.path.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')

        # self._get_data() ======================
        config = self.args.__dict__

        # self.model = self._build_model() =====================
        args = self.args

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

        self.ema = utils.EMA(0.995)
        self.update_ema_every = 10
        self.step_start_ema = 2000
        self.step = 0

        self.t_sample = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def _acquire_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if self.args.use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        #     device = torch.device('cuda:{}'.format(0))
        #     utils.print_log('Use GPU: {}'.format(self.args.gpu))
        # else:
        #     device = torch.device('cpu')
        #     utils.print_log('Use CPU')
        return device

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), pct_start=0.0, epochs=self.args.epochs)
        return self.optimizer

    def difficulty_rmse(self, x, y):
        # improve to employ difficulty.
        return torch.sqrt(torch.nn.MSELoss(x, y))

    def _select_criterion(self):
        self.criterion = self.difficulty_rmse

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        torch.save(self.ema_model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def print_log(message):
        print(message)
        logging.info(message)

    def get_next_steps(self):
        steps = 10  # sample from the probability distribution of steps
        steps = math.ceil(steps)

        return steps
