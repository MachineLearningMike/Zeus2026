import torch
import json, tzlocal
from torch.amp import autocast, GradScaler
from einops import rearrange
from pathlib import Path
import logging
from datetime import datetime
import sys, gc

sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.utils import check_dir
from modules.zeus_tqdm import zeus_tqdm
from models.Zeus import Zeus
from modules.zeus_dataset import Zeus_Dataset
import modules.utils as utils
from modules.params import Params
from modules_zeus.constants import ERA5_DATA_VARS
from modules_zeus.constants import (
    REWARD_DIFFICULTY_SCALER,  
    REWARD_IMPROVEMENT_WEIGHT,
    REWARD_IMPROVEMENT_MIN_DELTA,
)

class Train():
    def __init__(self, config, root_path, bayes_path):
               
        #======================== Fixed train state =======================
        # utils.set_seed(self.pm.seed)
        self.pm = Params(config)
        self.bayes_path = bayes_path
        
        self.dataset = Zeus_Dataset(config, root_path)
        logging.info("Zeus_Dataset is created.")
        self.nDevices, self.device = self._acquire_device()

        #---------- Loss ----------------
        min_sota_delta = torch.tensor([value for (key, value) in REWARD_IMPROVEMENT_MIN_DELTA.items() if key in self.pm.variables])  # (C,)
        self.min_sota_delta = rearrange(min_sota_delta, 'c -> c 1 1')
        self.era5_data_vars = self.load_era5_data_vars()
        #---------- Save config ---------
        configPath =  self.bayes_path / ('config-' + self.get_train_id_string() +'.json')
        utils.check_dir(configPath.parent)
        with open(configPath, 'w') as f:
            json.dump(config, f, indent=4)

        #=========================== Varing train state =========
        # Identified by bayesProject.
        # Updated during trainig, 
        # saved at the end of every epoch, 
        # loaded at the start of training, if saved earlier.
        # x_set is refreshed at every epoch, but not included.
        self.model = Zeus(config, self.nDevices).float()   # The only place where the model is created.
        logging.info("Zeus model is created.")

        self.optimizer = self._select_optimizer()
        logging.info("Optmizer is created")

        self.scheduler = self._select_scheduler(self.optimizer)
        logging.info("Scheduler is created.")
        
        self.scaler = GradScaler(device=self.device)

        self.epoch = 0
        self.epochs_since_improvement = 0
        self.rmses_old = torch.ones_like(self.era5_data_vars) * 1e9


    def _acquire_device(self):
        assert torch.cuda.is_available()
        nDevices = torch.cuda.device_count()
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # default device. device 0 if default is not changed.

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
        
        return nDevices, default_device
    
    def get_train_id_string(self):
        string = "-".join([str(v) for v in self.pm.hypers.values()])
        return string
    
    def get_minute_stamp(self):
        local_tz = tzlocal.get_localzone()
        now = datetime.now(local_tz)
        stamp = now.strftime("%Y%m%d_%H%M")
        return stamp

    def load_era5_data_vars(self):
        vars = [ERA5_DATA_VARS[k] for k in self.pm.variables if k in ERA5_DATA_VARS.keys()]
        self.era5_data_vars = torch.tensor(vars)
        return self.era5_data_vars
    
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

        logging.info(f"Optimizer selected: {optimizer_name.upper()} with lr={lr}, weight_decay={weight_decay}")

        return optimizer
    

    def _select_scheduler(self, optimizer):
        # Check train_loader length for scheduler
        steps_per_epoch = self.dataset.get_steps_per_epoch()

        # Scheduler parameters with defaults, overridable by args
        max_lr = getattr(self.pm, 'max_lr', optimizer.param_groups[0]['lr'] * 2)
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

        logging.info(f"Scheduler: OneCycleLR with max_lr={max_lr}, epochs={epochs}, steps_per_epoch={steps_per_epoch}")

        return scheduler
    
    def save_best_train(self, train_rmses, vali_rmses):   # (C,), (C,)
        wasBestEpoch = False
        device = train_rmses.device
        assert vali_rmses.device == device
        # rmses = 0.3 * train_rmses + 0.7 * vali_rmses    # Hard-coded for brevity.
        rmses = vali_rmses
        rmses = rmses * (self.era5_data_vars.to(device))

        if torch.mean(rmses) < torch.mean(self.rmses_old):
            wasBestEpoch = True
            self.rmses_old = rmses
            name = 'best-' + self.get_train_id_string()
            logging.info("saving best train_state... avg rmse: {:.3f}, name: {}".format(torch.mean(rmses).item(), name))
            self.save_state(name=name)

        self.epochs_since_improvement = self.epochs_since_improvement + 1 if not wasBestEpoch else 0
        
        keep_go_next_epoch = self.epochs_since_improvement < self.pm.epoch_patience
        best_rmse_yet = torch.mean(self.rmses_old).item()

        return keep_go_next_epoch, best_rmse_yet
    
    def save_state(self, name=None, train_finished=False):
        if name is None: name = 'train_state-' + self.get_train_id_string()
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler':   self.scaler.state_dict(),
            'epoch': self.epoch,
            'rmses_old': self.rmses_old,
            'epochs_since_improvement': self.epochs_since_improvement,
            'train_finished' : train_finished,
        }
        utils.check_dir(self.bayes_path)
        torch.save(checkpoint, self.bayes_path / (name + '.pth'))
        print(f"Checkpoint '{name}' saved, including model, optimizer, scheduler, scaler, and rmses_old")
    
    def try_load_state(self, name=None):    # Couples with save_state
        if name is None: name = 'train_state-' + self.get_train_id_string()
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
            self.scaler.load_state_dict(checkpoint['scaler'])

            self.rmses_old = checkpoint['rmses_old']
            self.epochs_since_improvement = checkpoint['epochs_since_improvement']

            print(f"Checkpoint '{name}' loaded to {self.device}, including model, optimizer, scheduler, scaler, and rmses_old")
        except:
            # Everything else has its default value.
            epoch = -1

        return epoch
    
    def set_train_mode(self):
        if self.model is not None: self.model.train()
    
    def set_eval_mode(self):
        if self.model is not None: self.model.eval()
    
    def run_epoch(self, epoch, x_set_label, train=False):
        self.epoch = epoch

        gc.collect()
        torch.cuda.empty_cache()

        if not self.model.deployed_to_gpus: self.model.to(self.device)
        
        self.dataset.refresh_x_set(x_set_label, seed=self.pm.seed)
        logging.info("{} is refreshed.".format(x_set_label))
        pbar = zeus_tqdm(self.dataset)
        pbar.start_prefetch_thread()
        
        FG = self.pm.FG; T1 = self.pm.nFramesIn // FG
        epoch_loss = []; epoch_rmses = []
        
        if train: self.optimizer.zero_grad()                     # train
        if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Fetch one batch asynchronously
        for batch in pbar:              # Wait for the asynchronous .prefetch
            # print('got data!')
            
            if batch is None: break     # just in case
            batch_x, batch_y, batch_diff, era5_data_vars = batch    # (B, T1FG, C, H, W), (B, T2FG, C, H, W), None or (B, C, H, W), (C,)
            # self._print_tensor_state(1, batch_x)    # 32, cpu, False
            # self._print_tensor_state(2, batch_y)    # 32, cpu, False
            # self._print_tensor_state(3, era5_data_vars) # 32, cpu, False
            if self.pm.include_difficulty:
                pass 
                # self._print_tensor_state(4, batch_diff)   # 32, cpu, False
            no_more_data = pbar.no_more_data()
            if not no_more_data:  pbar.prefetch(x_set_label)    # Fetch one batch asynchronously

            batch_x = batch_x[:, :, :, :self.pm.image_height, :self.pm.image_width].pin_memory()     # torch.float32, cpu
            batch_y = batch_y[:, :, :, :self.pm.image_height, :self.pm.image_width].pin_memory()     # torch.float32, cpu
            era5_data_vars = era5_data_vars.pin_memory() # torch.float32, cpu
            # self._print_tensor_state(5, batch_x)        # 32, cpu, False
            # self._print_tensor_state(6, batch_y)        # 32, cpu, False
            # self._print_tensor_state(7, era5_data_vars) # 32, cpu, False

            if self.pm.include_difficulty:
                batch_diff = batch_diff[:, :, :self.pm.image_height, :self.pm.image_width]
                # self._print_tensor_state(8, batch_diff)

            T2 = (batch_y.shape)[1] // FG
            batch_x = batch_x.to(self.device, non_blocking=True)               # torch.float32, cuda
            batch_y = batch_y.float().to('cuda:' + str(self.nDevices-1), non_blocking=True)       # torch.float32, cuda
            era5_data_vars = era5_data_vars.to('cuda:' + str(self.nDevices-1), non_blocking=True) # torch.float32, cuda
            # self._print_tensor_state(9, batch_x)            # 32, cuda, False
            # self._print_tensor_state(10, batch_y)           # 32, cuda, False
            # self._print_tensor_state(11, era5_data_vars)    # 32, cuda, False

            (B, T1FG, C, H, W) = batch_x.shape
            pred_y = torch.zeros(size=(B,0,C,H,W)).to(self.device)  # torch.float32, cuda
            # self._print_tensor_state(12, pred_y)            # 32, cuda, False
            step_loss = []; step_rmses = []

            for t2 in range(T2):
                with autocast(device_type='cuda'):
                    self._print_tensor_state(13, batch_x)   # 32, cuda, False
                    pred_y = pred_y.to(self.device, non_blocking=True)  #------------
                    self._print_tensor_state(14, pred_y)    # 32, cuda, False
                    base_x = torch.cat([batch_x, pred_y], dim=1)    #---------------
                    base_x = base_x[:, -T1*FG:, :, :, :]     #(B, T1FG, C, H, W), torch.float32, cuda, False
                    self._print_tensor_state(15, base_x)    # 32, cuda, False
                    pred = self.model(base_x)     # (B,FG,C,H,W), torch.float16 !!!, cuda, True !!!
                    self._print_tensor_state(16, pred)      # 16 !!!, cuda7, True !!!
                    pred_y = pred_y.to('cuda:' + str(self.nDevices-1), non_blocking=True)   #-------------
                    pred_y = torch.cat([pred_y, pred.detach()], dim=1) # torch.float32, cuda, False
                    self._print_tensor_state(17, pred_y)    # 32, cuda7, False
                    target = batch_y[:, t2*FG:(t2+1)*FG, :,:] #.to(self.device)   # (B,FG,C,H,W)  # torch.float32, cuda, False
                    self._print_tensor_state(18, target)    # 32, cuda, False
                    pred = self.dataset.unnormalize(pred)   # torch.float32 !!!, cuda, True !!!
                    self._print_tensor_state(19, pred)      # 32, cuda, True !!!
                    target = self.dataset.unnormalize(target)   # torch.float32, cuda, False
                    self._print_tensor_state(20, target)    # 32, cuda, False
                    loss, rmse = self.loss(target, pred, batch_diff, era5_data_vars)  #.to(self.device)) # (,), (C,), torch.float32, cuda, True !!!
                    self._print_tensor_state(21, loss)      # 32, cuda, True !!!
                    self._print_tensor_state(22, rmse)      # 32, cuda, True !!!
                    step_loss.append(loss.cpu().detach())  # (,) - list.        Even if no cpu(), their stack results in cpu !!!
                    step_rmses.append(rmse.cpu().detach())      # (C,) - list.  If no cpu(), their stack results in cuda

                if train:
                    loss = loss / T2    # Normalize loss, so gradients match what we'd get if we did a one big forwards pass.
                    self.scaler.scale(loss).backward()          # accumulate gradients T2 times by backpropagating (scaled) loss

            if train: 
                self.scaler.step(self.optimizer)            # (un-scalsed) step for accumulated gradients.
                self.optimizer.zero_grad()                     # train
                self.scaler.update()                            # This is the only place to adjust the scale factor.
          
            # Conclude a step
            self.scheduler.step()   # Our scheduler is OneCycleLR, which is a per-batch scheduler
            step_loss = torch.tensor(step_loss).mean()   # (,)
            self._print_tensor_state(23, step_loss)         # 32, cpu False
            step_rmses = torch.stack(step_rmses, dim=0).mean(dim=0)   # (C,)
            self._print_tensor_state(24, step_rmses)        # 32, cpt False
            strC = ", ".join(f"{x:.2f}" for x in step_rmses.ravel().tolist())
            pbar.set_description('Loss: {:.3f}, rmses: {}'.format(step_loss.item(), strC))

            epoch_loss.append(step_loss)    # (,) - list
            epoch_rmses.append(step_rmses)   # (C,) - list
            if no_more_data: break
            # print('waiting for data...')

        epoch_loss = torch.stack(epoch_loss, dim=0).mean(dim=0)  # (,)
        epoch_rmses = torch.stack(epoch_rmses, dim=0).mean(dim=0)       # (C,)
        strC = ", ".join(f"{x:.2f}" for x in epoch_rmses.ravel().tolist())
        pbar.set_description('Loss: {:.3f}, rmses: {}'.format(epoch_loss.item(), strC))
        pbar.stop_prefetch_thread()

        return epoch_loss, epoch_rmses  # (,), (C,)
    
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
    
if __name__ == "__main__":
    
    from models.config import config, root_path

    bayes_path = root_path / 'bayes' / 'test'
    pbounds = {     # Mike!!!
        "patch_size": (8, 8, int),
        "heads": (8, 8, int),
        "attLayers": (8, 8, int),   # !!! assert attLayers + zeusLayers <= 3
        "zeusLayers": (8, 8, int),  # 
    }
    
    train = Train(config, root_path, bayes_path)
    
