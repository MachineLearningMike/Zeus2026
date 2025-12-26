import torch
import numpy as np
import json, tzlocal
from torch.amp import autocast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import sys, gc, copy, time, math
from importlib import import_module

# import matplotlib
# matplotlib.use("Agg")  # non-GUI backend, safe in threads
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.utils import check_dir, check_gradient
from modules.zeus_tqdm import zeus_tqdm
from modules.params_Mike1 import flatten_pm_single
from models.config_Mike1 import config

module = import_module(config['Zeus_Model_module'])
Zeus = getattr(module, 'Zeus')

# from models.Zeus_ResNet import Zeus
# # from Zeus_Flash2D_Blocked import Zeus

from models.Model_Dom_Mike1 import Model_Dom
from modules.zeus_dataset_mike1 import Zeus_Dataset
# from models.Zeus_Mike1 import Zeus
from modules.zeus_dataloader import Zeus_DataLoader
import modules.utils as utils
from modules.params_Mike1 import Params
# from modules.read_cursor_keys_linux_windows_2 import KeyReader    This is crazy.
from modules.read_cursor_keys_linux_windows import KeyReader

kr = KeyReader()
kr.init_key_listener()

from AdaptiveLRLearner import AdaptiveLRLearner

def print_data_attrs(obj):
    for name, value in vars(obj).items():
        print(f"{name}: {value}")

class Model_Trainer():
    def __init__(self, pm, dataset_path, management_path, project_path, bayes=None, trainable=True):
        self.pm = pm
        self.bayes = bayes
        # self.management_path = management_path
        self.project_path = project_path
        self.setup_logging(project_path)

        self.dataset = Zeus_Dataset(self.pm, dataset_path, management_path)
        # self.dataset = Zeus_DataLoader(Zeus_Dataset(self.pm, dataset_path, management_path), batch_size=None, shuffle=None)     # A bit slower?
        
        self.dataset.load_data()
        logging.info("Zeus_Dataset created and loaded.")
        self.device_type, self.nDevices, self.device = self._acquire_device()
        self.devices = [ self.device_type + ':' + str(id) for id in range(self.nDevices)]
        
        # self.model = Zeus(self.pm, self.devices[0])
        self.model_doms = []
        for md_id in range(len(pm.models)):
            if md_id not in pm.models_to_play: continue
            flatten_pm = flatten_pm_single(pm, md_id)
            md = Model_Dom(md_id, self.devices[0], flatten_pm, self.dataset, management_path, project_path, bayes=bayes, trainable=trainable)
            self.model_doms.append(md)
            
        self.focus_md_id = 0
        self.transient_step = 0
        self.train_bombs = 1
        
        self.train_start_time = time.time()
        
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

        log_path = log_dir / 'log.txt'
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
        
    def _acquire_device(self):
        if torch.cuda.is_available():
            device_type = 'cuda'
            default_device = torch.device('cuda:0')
            nDevices = torch.cuda.device_count()
        else:
            device_type = 'cpu'
            default_device = torch.device('cpu:0')
            nDevices = torch.cpu.device_count()
                   
        return device_type, nDevices, default_device
    
    def save_state(self, name=None):
        if name is None: name = "train_state_trainer"  # + '-' + self.get_train_id_string()
        checkpoint = {
            'dataset': self.dataset.pack_state(),
            'transient_step': self.transient_step,
        }
        utils.check_dir(self.project_path)
        torch.save(checkpoint, self.project_path / (name + '.pth'))
        torch.save(checkpoint, self.project_path / (name + '.pth.bak'))   # Create a backup, as creating takes a long time and may fail by the cloud Spot service.
        return
    
    def try_load_state(self, name=None):    # Pairs with save_state
        if name is None: name = name = "train_state_trainer" #+ self.get_train_id_string()
        
        file_found = True
        
        try:
            checkpoint_path = self.project_path / (name + '.pth')
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logging.info("Trainer checkpoint loaded from the .pth file")
        except Exception as e:
            logging.info(str(e))
            try:
                checkpoint_path = self.project_path / (name + '.pth.bak')     # Maybe the previous save.
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                logging.info("Trainer checkpoint loaded from the .pth.bak file")
            except Exception as e:
                logging.info(str(e))
                file_found = False
                
        if file_found:        
            if checkpoint.get('dataset', None) is not None:
                self.dataset.unpack_state(checkpoint['dataset'])
                logging.info("dataset state loaded with saved state")

            if checkpoint.get('transient_step', None) is not None:
                self.transient_step = checkpoint['transient_step']
                logging.info("transient_step loaded: {}".format(self.transient_step))
        else:
            pass
        
    def train(self):
        [md.zero_losses_metrics() for md in self.model_doms]        
        [md.try_load_state() for md in self.model_doms]
        self.try_load_state()        
        
        ran_epochs = False
        best_metric_yet = 1e10
        
        mds_to_keep_going = self.model_doms
        
        for epoch in range(0, 100): #self.pm.epochs):
            mds_to_train = mds_to_keep_going
            self.model_doms = mds_to_keep_going
        
            if len(mds_to_train) <= 0: break            
            ran_epochs = True
             
            for md in mds_to_train: md.epoch += 1
            
            [logging.info("--------- Starting train epoch {} @ model {} ----------".format(md.epoch, md.md_id)) for md in mds_to_train]
            
            # md_focus_list = [md for md in mds_to_train if md.md_id == self.focus_md_id]
            # md_focus = md_focus_list[0] if len(md_focus_list) == 1 else mds_to_train[0]
            
            train_losses, train_metrics = self.run_epoch(mds_to_train, "train_set", trainable=True)   # (,), (C,)

            [md.get_epoch_display_batch(loss_list, metric_list) for (md, loss_list, metric_list) in zip(mds_to_train, train_losses, train_metrics)]
            [logging.info("Finished train epoch {}, {}".format(md.epoch, (md.display))) for md in mds_to_train]
    
            mds_to_validate = [md for md in mds_to_train if md.pm.skip_validation is False]
            if len(mds_to_validate) <= 0:
                logging.info("All validation(s) skipped as configured with 'skip_validation'")
                continue
            else:
                [logging.info("--------- Starting valid epoch {} @ model {} ----------".format(md.epoch, md.md_id)) for md in mds_to_validate]
                            
                # md_focus_list = [md for md in mds_to_validate if md.md_id == self.focus_md_id]
                # md_focus = md_focus_list[0] if len(md_focus_list) == 1 else mds_to_validate[0]
                
                vali_losses, vali_metrics = self.run_epoch(mds_to_validate, "vali_set", trainable=False)     # (,), (C,)   #------------------------

                [md.get_epoch_display_batch(loss, metric) for (md, loss, metric) in zip(mds_to_validate, vali_losses, vali_metrics)]
                [logging.info("Finished vali epoch {}, {}".format(md.epoch, (md.display))) for md in mds_to_validate]

                keep_goings, best_metrics = map(list, zip(*[md.save_best_train(vm) for md, vm in zip(mds_to_validate, vali_metrics)]))
                mds_to_keep_going = [md for (md, keep_going) in zip(mds_to_validate, keep_goings) if keep_going is True]
                
                [md.save_state(train_finished=True) for md in mds_to_validate  if md not in mds_to_keep_going]
                [logging.info("!!! Train finished: model {}".format(md.md_id)) for md in mds_to_validate  if md not in mds_to_keep_going]
                self.save_state()
                
                if len(mds_to_validate) == 1:
                    best_metric_yet = best_metrics[0]
                    if self.bayes is not None:  self.bayes.on_epoch_end()
            
            if len(mds_to_keep_going) == 0: break

        return - best_metric_yet
    
    def show_data(self, batch_x):
        B, T, C, H, W = batch_x.shape
        # Pick random b, t, c indices
        b = torch.randint(0, B, (1,)).item()
        t = torch.randint(0, T, (1,)).item()
        c = torch.randint(0, C, (1,)).item()
        print(f"Selected slice: b={b}, t={t}, c={c}")

        # Extract 2D slice
        slice_2d = batch_x[b, t, c].cpu().numpy()  # shape (H, W)
        mean = np.mean(slice_2d)
        std = np.std(slice_2d)

        # Create coordinate grid
        x = np.arange(W)
        y = np.arange(H)
        Xg, Yg = np.meshgrid(x, y)

        # Plot as 3D surface
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(Xg, Yg, slice_2d, cmap='viridis', edgecolor='none')

        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Value')
        ax.set_title(f"3D Surface of X[b={b}, t={t}, c={c}, mean={mean}, std={std}]")

        plt.tight_layout()
        plt.show()

    # def loss(self, target, pred, batch_diff, era5_data_vars):
    def loss(self, target, pred, batch_diff, era5_data_vars, loss_mask=None):
        # (B,FG,C,H,W), (B,FG,C,H,W), (B, C, H, W), (C,)
        # 32, cuda, False ; 32, cuda, True !!!, 32, cuda, False, 32, cuda, False
        device = pred.device
        target = target.to(device)
        rmse = ((pred - target) ** 2).mean(dim=(1,2)).sqrt()    # (C,), 32, cuda, True !!!
        # self._print_tensor_state(1, target)                 # 
        # self._print_tensor_state(2, pred)
        # self._print_tensor_state(3, batch_diff)
        # self._print_tensor_state(4, era5_data_vars)
        # self._print_tensor_state(4.5, rmse)

        # if batch_diff is not None:
        #     baseline_rmse = 0
        #     improvement = baseline_rmse - rmse  - self.min_sota_delta.to(device)   # (B, C, H, W), 32, cuda, True,  the-higher-the-better
        #     # self._print_tensor_state(5, improvement)
        #     baseline_improvement = torch.maximum(torch.tensor(0.0).to(device), improvement)    # 0.0, (B, C, H, W), 32, cuda, True !!!,  the-higher-the-better
        #     # self._print_tensor_state(6, baseline_improvement)
        #     gamma = torch.pow(REWARD_DIFFICULTY_SCALER, batch_diff * 2 - 1).to(device)  # (B, C, H, W), 32, cuda, False,  the-higher-the-more-difficult
        #     # self._print_tensor_state(7, gamma)
        #     quality_score = self.get_curved_scores(rmse, gamma)       # (B, C, H, W), 32, cuda, True !!!, the-higher-the-worse
        #     # self._print_tensor_state(8, quality_score)
        #     improvement_score = self.get_curved_scores(-baseline_improvement, gamma) # (B, C, H, W), 32, cuda, True !!!, the-higher-the-worse
        #     # self._print_tensor_state(10, improvement_score)
        #     rmse = (1 - REWARD_IMPROVEMENT_WEIGHT) * quality_score + REWARD_IMPROVEMENT_WEIGHT * improvement_score    # (B, C, H, W), 32, cuda, True !!!
        #     # self._print_tensor_state(11, rmse)
        
        # era5_data_vars = era5_data_vars.to(device)
        # reward = rmse * era5_data_vars   # (C,), 32, cuda, True !!!, the-higher-the-worse
        # self._print_tensor_state(12, reward)
        # reward = reward.mean()  # (,)

        # return rmse.mean(), rmse # (,), (C,)     32, cuda, True

        # era5_data_vars = era5_data_vars.to(device)
        
        if loss_mask is not None: rmse = rmse * loss_mask
        rmse = rmse * self.pm.loss_weights_cuda
        return (rmse.mean()) ** 2, rmse # (,), (C,)     32, cuda, True
    
    def clean_gpu_every(self, n_iter, step):
        if step % n_iter == 0:
            gc.collect()
            torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()  # only for inter-process communicatoin
    
    def respond_to_cursor(self):
        direction, count = kr.read_key()
        factor = None; step = math.pow(10.0, 1/10)

        if direction is None:
            return
        elif direction == "up":
            factor = math.pow(step, count)         
        elif direction == "down":
            factor = math.pow(1 / step, count)
        elif direction == "left":
            self.train_bombs = self.train_bombs - count if self.train_bombs - count >= 1 else 1
            return
        elif direction == "right":
            self.train_bombs = self.train_bombs + count
            return
        elif direction == "0":
            self.focus_md_id = 0
            return
        elif direction == "1":
            self.focus_md_id = 1
            return
        elif direction == "2":
            self.focus_md_id = 2
            return
        elif direction == "3":
            self.focus_md_id = 3
            return
        elif direction == "4":
            self.focus_md_id = 4
            return
        else: return

        focus_md = self.get_md_focus(self.model_doms)

        if factor:
            lr = focus_md.get_learning_rate()
            assert lr is not None
            lr *= factor
            focus_md.set_learning_rate(lr)

        return
    
    def get_md_focus(self, mds):
        md_focus_list = [md for md in mds if md.md_id == self.focus_md_id]
        md_focus = md_focus_list[0] if len(md_focus_list) == 1 else mds[0]
        return md_focus
    
    def run_epoch(self, mds, x_set_label, trainable=True):
               
        # data_slots = [0] + [md.md_id + 1 for md in mds]
        # self.dataset.set_data_slots(data_slots)
        renewed = self.dataset.refresh_x_set(x_set_label, seed=self.pm.seed, renew_if_new_epoch=True)
        logging.info("{} {}".format(x_set_label, "renewed" if renewed else "left over from previous run"))

        for md in mds:
            if md.packed_losses_metrics is not None:
                logging.info("Continuing saved epoch... epoch/model {}/{}".format(md.epoch, md.md_id))
                        
        pbar = zeus_tqdm(self.dataset)
        pbar.start_prefetch_thread()

        if self.nDevices > 1: gpu_executor = ThreadPoolExecutor(max_workers=len(self.devices))
        else: gpu_executor = None
        
        if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Fetch one batch asynchronously

        step = 0
        start = time.time()
        start1 = time.time()
        for batch in pbar:              # Wait for the asynchronous .prefetch
            end = time.time()
            
            if (end-self.train_start_time) > self.pm.train_hours_limit * 3600: break

            if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Prefetch ensures getting new bacthes.
            if batch is None: break     # just in case
            batch_of_ss_lists, batch_diff, era5_data_vars = batch   # (slots, B, T1FG, C, H, W), (B, T2FG, C, H, W), None or (B, C, H, W), (C,)

            if False:
                self.show_data(batch_x)
                continue

            [md.optimizer.zero_grad(set_to_none=True) for md in mds]            
            # self.transient_step += slots.shape[1]     # increase by batch size
            
            if self.nDevices <= 1:
                # !!! This context leads lots of parameters that don't receive gradient. Move into single_GPU_forward_loss_backward.
                # context = autocast(device_type=self.device_type, dtype=torch.bfloat16) if self.pm.use_autocast else nullcontext()
                # with context:
                # with torch.set_grad_enabled(trainable):   # grad is managed by single_GPU_forward_loss_backward

                device = self.devices[0]
                # for _ in  range(self.train_bombs):    # train_bomb cannot work because we destroy batch_of_ss_lists.
                    
                Xs = batch_of_ss_lists[0]   # batch_of_ss_lists undergoes changes. This code doesn't know the content of X
                assert len(Xs) == self.pm.nStepsPerBatch + (2 if self.pm.pair_not_single_snapshot else 1)
                    
                for md in mds:            
                    with autocast(device_type='cuda', dtype=torch.bfloat16):    # time: 97 -> 82, memory: little
                        #------------------------ this algorithm is studied in zeus_train_6_hourly.py -----------
                        
                        # assert len(Xs) >= 2, "length of Xs is not enough 1"
                        
                        X0 = Xs[0]; Xs = Xs[1:]
                        if self.pm.pair_not_single_snapshot:
                            X1 = Xs[0]; Xs = Xs[1:]
                        else:
                            X1 = X0
                                            
                        oldQ = [(X0, X1)]

                        # assert len(Xs) > 0, "length of Xs is not enough 2"
                        
                        #--------------------------------------------- starting a set of steps for md...

                        step_loss = []; step_metrics = []; cnt = 0
                        while len(Xs) > 0:  # loops at least once
                            # State: 
                            # len(Xs) > 0, len(oldQ) > 0, [x0.device == device and x1.device == device for (x0, x1) in oldQ]
                            # 2 * len(oldQ) count of X is stored in oldQ in device, together with model parameters.
                            # oldQ stores raw X's
                            
                            #----------------------------------------- starting a step for md...

                            Y = Xs[0]; Xs = Xs[1:]
                            
                            if self.pm.pair_not_single_snapshot:
                                newQ = [(oldQ[0][1], Y)]  # assert len(newQ) > 0        # newQ[0] stores raw X's
                            else:
                                newQ = [(Y, Y)]

                            while len(oldQ) > 0:     # loops at least once
                                
                                #-------------------------------------- starting a substep for md...
                                
                                X0, X1 = oldQ[0]; oldQ = oldQ[1:]    # assert X.device == device
                                cascade_step = len(newQ)
                                
                                X0.to(device)
                                if self.pm.pair_not_single_snapshot:
                                    X1.to(device)
                                Y.to(device)
                                                            
                                substep_result, _substep_loss, _substep_metrics = md.forward_backward_no_step(
                                        X0, X1, Y, cascade_step,
                                        self.loss, self.dataset.normalize_meteo, self.dataset.unnormalize_meteo, 
                                        batch_diff, era5_data_vars, trainable=trainable,
                                        type_metrics = self.pm.metrics, predicted_value_required = (len(Xs) > 0 or len(oldQ) > 0),
                                        pressure_indices = self.dataset.pressure_indices,
                                        pressure_values = self.dataset.pressure_values,
                                    )
                                cnt += 1

                                if len(newQ) < self.pm.max_cascade_step: 
                                    if self.pm.pair_not_single_snapshot:
                                        newQ.append((X1, substep_result))                      # +1 X
                                    else:
                                        newQ.append((substep_result, substep_result))
                                    
                                del X0
                                if not self.pm.pair_not_single_snapshot:
                                    del X1      # The same object, but different reference.
                                                            
                                #--------------------------------------- The substep finished.
                                
                                step_loss.append(_substep_loss)          # [substep_loss...]
                                step_metrics.append(_substep_metrics)    # [substep_metrics...]
                                
                            #------------------------------------------- The step finished.
                            # state: len(oldQ) == 0, len(oldQ) > 0
                            oldQ = newQ
                            
                        # assert cnt == int((self.pm.nStepsPerBatch + 1) * self.pm.nStepsPerBatch / 2)
                        # print('++++++++++++++++++++++++++', cnt)
                            
                        #------------------------------------------------ The set of steps finished.
                        md.store_step_loss_and_metrics(step_loss, step_metrics) # [substep_loss...], [substep_metrics...]
                        
                        if trainable:
                            check_gradient(md.model)
                            if md.pm.useLRLearner:
                                md.lr_controller.stepOptimizer(_substep_loss.item())
                            else:
                                md.scaler.step(md.optimizer)
                                md.scaler.update()
            else:
                pass
                    
            md_focus = self.get_md_focus(mds)
            end1 = time.time()
            pbar.set_description("{}, wait/cycle {:.0f}/{:.0f} u {:.0f} lr {:1.1e} tb {}"
                .format(md_focus.pbar_description, (end-start) * 1000, (end1-start1) * 1000, ((end1-start1) * 1000 - (end-start) * 1000) / self.train_bombs, md_focus.get_learning_rate(), self.train_bombs))
            start1 = time.time()
            
            step += 1
            if trainable is True and step % self.pm.steps_to_save == 0:
                [(md.pack_losses_metrics(), md.save_state()) for md in mds]
                self.save_state()
            self.clean_gpu_every(1, step)
            self.respond_to_cursor()
            start = time.time()
        
        [md.get_pbar_description_epoch() for md in mds]
        md_focus = self.get_md_focus(mds)
        pbar.set_description(md_focus.pbar_description)

        pbar.stop_prefetch_thread()
        
        if trainable is True: 
            [(md.pack_losses_metrics(), md.save_state()) for md in mds]
            self.save_state()
        if gpu_executor is not None: gpu_executor.shutdown()

        epoch_losses = [md.epoch_losses for md in mds]
        epoch_metrics = [md.epoch_metrics for md in mds]

        return epoch_losses, epoch_metrics  # (,), (C,)

    
if __name__ == "__main__":
    
    from models.config_Mike1 import config
    from modules.params_Mike1 import Params
    
    pm = Params(config)
    dataset_path = Path(pm.dataset_path)
    management_path = Path(pm.management_path)
    check_dir(management_path)
    project_path = management_path / "projects" / pm.bayesProject
    check_dir(project_path)
    
    model_sequence_trainer = Model_Trainer(pm, dataset_path, management_path, project_path, bayes=None, trainable=True)
    model_sequence_trainer.train()
