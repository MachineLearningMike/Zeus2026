import torch
import json, tzlocal
from torch.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor
from einops import rearrange
from pathlib import Path
import logging
from datetime import datetime
import sys, gc, copy

sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.utils import check_dir
from modules.zeus_tqdm import zeus_tqdm
from models.Zeus_Cross import Zeus, DoubleZeus
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
    def __init__(self, config, root_path, bayes_path, trainable=True):
        self.config = config
        #======================== Fixed train state =======================
        # utils.set_seed(self.pm.seed)
        self.pm = Params(config)
        self.bayes_path = bayes_path
        self.trainable = trainable
        
        self.dataset = Zeus_Dataset(config, root_path)
        logging.info("Zeus_Dataset created.")
        self.device_type, self.nDevices, self.device = self._acquire_device()
        self.devices = [ self.device_type + ':' + str(id) for id in range(self.nDevices)]
        # if self.trainable: assert self.nDevices == self.pm.zeusLayers
        
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

        try:
            if not self.pm.doubleZeus:
                self.model = Zeus(config, self.nDevices, device_type=self.device_type, abundant_GPUs=self.pm.abundant_GPUs).float()                
                logging.info("Zeus model created.")
                if self.nDevices > 1:
                    self.slave_models = self.create_slave_models(self.devices)
            else:
                self.model = DoubleZeus(config, self.nDevices, device_type=self.device_type, abundant_GPUs=self.pm.abundant_GPUs, freeze_zeus1=self.pm.freeze_zeus1)
                logging.info("DoubleZeus model created. nDevices: {}, trainable: {}".format(self.nDevices, trainable))

        except Exception as e:
            logging.info(str(e))
            sys.exit(0)
        
        if self.trainable:
            self.optimizer = self._select_optimizer()
            self.scheduler = self._select_scheduler(self.optimizer)
            self.scaler = GradScaler(device=self.device)

            self.epoch = 0
            self.epochs_since_improvement = 0
            self.rmses_old = torch.ones_like(self.era5_data_vars) * 1e9

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
        lr = getattr(self.pm, 'lr', 1e-6)
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
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        logging.info(f"Optimizer created: {optimizer_name.upper()} with lr={lr}, weight_decay={weight_decay}")

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

        logging.info(f"Scheduler created: OneCycleLR with max_lr={max_lr}, epochs={epochs}, steps_per_epoch={steps_per_epoch}")

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
            # train==True means the next possible loading should train not validate.
            self.save_state(name=name)  # Extra epoch_save after as a conlusion of validate()
            # print('----------- best_saved')

        self.epochs_since_improvement = self.epochs_since_improvement + 1 if not wasBestEpoch else 0
        
        keep_go_next_epoch = self.epochs_since_improvement < self.pm.epoch_patience
        best_rmse_yet = torch.mean(self.rmses_old).item()

        return keep_go_next_epoch, best_rmse_yet
       
    def save_state(self, name=None, train_finished=False, step_state=None):
    #-------------- My version
        assert self.trainable, "Non-trainable train object cannot be saved."
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
            'dataset': self.dataset.pack_state(),
            'step_state': step_state,
        }
        utils.check_dir(self.bayes_path)
        torch.save(checkpoint, self.bayes_path / (name + '.pth'))
        torch.save(checkpoint, self.bayes_path / (name + '.pth.bak'))   # Create a backup, as creating takes a long time and may fail by the cloud Spot service.
        # logging.info(f"Checkpoint '{name}' saved, including model, optimizer, scheduler, scaler, rmses_old, and dataset state")
    
    
    def try_load_state(self, name=None):    # Pairs with save_state
        if name is None: name = 'train_state-' + self.get_train_id_string()
        
        file_not_found = False; train_finished = False; file_loaded = False

        try:
            checkpoint_path = self.bayes_path / (name + '.pth')
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logging.info("Checkpoint loaded from the .pth file")
        except Exception as e:
            logging.info(str(e))
            try:
                checkpoint_path = self.bayes_path / (name + '.pth.bak')     # Maybe the previous save.
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                logging.info("Checkpoint loaded from the .pth.bak file")
            except Exception as e:
                logging.info(str(e))
                file_not_found = True

        if not file_not_found:
            if checkpoint.get('train_finished', None) is not None:
                train_finished = checkpoint['train_finished']
            else:
                logging.info("Train_finished not found")
                sys.exit()
        
        if file_not_found and not self.trainable:
            logging.info("Train checkpoint not found")
            sys.exit()
        
        if not file_not_found and (not train_finished or not self.trainable):
            try:
                # Load model parameters
                epoch = checkpoint['epoch']
                
                state_dict = checkpoint['model']
                strict = True
                
                if self.pm.doubleZeus:
                       
                    zeus1_state = {key : value for key, value in state_dict.items() if key.startswith('zeus1')}
                    zeus2_state = {key : value for key, value in state_dict.items() if key.startswith('zeus2')}
                    # zeus2_state = {(key.replace("mlp_head.", "", 1) if key.count("mlp_head.") > 1 else key) : value for key, value in zeus2_state.items()}
                                
                    missing_keys, unexpected_keys = self.model.load_state_dict(zeus1_state, strict=False)
                    logging.info("Model loaded with saved zeus1 state: \nmissing_keys: {}, \nunexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))

                    missing_keys, unexpected_keys = self.model.load_state_dict(zeus2_state, strict=False)
                    logging.info("Model loaded with saved zeus2 state: \nmissing_keys: {}, \nunexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))

                else:
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    logging.info("Model loaded with saved state: \nmissing_keys: {}, \nunexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))

                step_state = None
                if self.trainable:
                    
                    try:                       
                        # Restore optimizer
                        if self.optimizer is not None and "optimizer" in checkpoint:
                            self.optimizer.load_state_dict(checkpoint["optimizer"])

                            # Move optimizer state tensors to the same device as their parameters
                            for group in self.optimizer.param_groups:
                                for p in group["params"]:
                                    if p in self.optimizer.state:
                                        for k, v in self.optimizer.state[p].items():
                                            if torch.is_tensor(v):
                                                self.optimizer.state[p][k] = v.to(p.device, non_blocking=True)                       
                        logging.info("Optimizer loaded with saved state")

                        self.scheduler.load_state_dict(checkpoint['scheduler'])
                        logging.info("Scheduler loaded with saved state")
                        self.scaler.load_state_dict(checkpoint['scaler'])
                        logging.info("Scaler loaded with saved state")

                        self.rmses_old = checkpoint['rmses_old']
                        self.epochs_since_improvement = checkpoint['epochs_since_improvement']
                        logging.info("rmses_old and epochs_since_improvement loaded with saved values")
                    
                        if checkpoint.get('dataset', None) is not None:
                            self.dataset.unpack_state(checkpoint['dataset'])
                            logging.info("dataset state loaded with saved state")

                        if checkpoint.get('step_state', None) is not None:
                            step_state = checkpoint['step_state']

                    except Exception as e:
                        logging.info(str(e))

                # print('----------- step_state loaded is ', step_state)
                if step_state is not None:  # The previous session was terminated after a new step was finished and before an epoch is finished.
                    epoch = epoch
                else:   # The previous session was terminated after an epoch was finished and before a new step was finished. 
                    epoch = epoch + 1

                logging.info(f"Checkpoint '{name}' loaded to {self.device}, including model, optimizer, scheduler, scaler, rmses_old, and datase state")
                file_loaded = True
            except Exception as e:
                logging.info(str(e))
                file_loaded = False
        
        else:
            pass
        
        if file_not_found:               
            # Start a new training, creating the file.
            epoch = 0; step_state = None    # This is the moment where an existing train will be overriden !!!    
        elif train_finished and self.trainable:
            # Keep the finished train intact.
            logging.info("You tried to start/continue a finished training. Aborting...")
            sys.exit()
        elif not file_loaded:
            # Start a new training, creating the file.
            epoch = 0; step_state = None    # This is the moment where an existing train will be overriden !!!    
        else: # file found, train not finished, file loaded
            # Continue the loaded train.
            epoch = epoch; step_state = step_state

        return epoch, step_state
    
    def set_train_mode(self):
        if self.model is not None: self.model.train()
    
    def set_eval_mode(self):
        if self.model is not None: self.model.eval()
        
    def pack_step_state(self, epoch_loss, epoch_rmses):
        assert epoch_loss is not None and epoch_rmses is not None
        return {
            "epoch_loss": [t.item() for t in epoch_loss],          # each scalar -> float
            "epoch_rmses": [t.tolist() for t in epoch_rmses]         # each (4,) tensor -> list    
        }

    def unpack_step_state(self, data):
        epoch_loss, epoch_rmses = [], []
        if data is not None:
            epoch_loss = [torch.tensor(v) for v in data["epoch_loss"]]
            epoch_rmses = [torch.tensor(v) for v in data["epoch_rmses"]]
        return epoch_loss, epoch_rmses
    
    def create_slave_models(self, devices):
        slave_models = {device : copy.deepcopy(self.model) for device in devices}
        # if self.pm.abundant_GPUs:
        #     slave_models = {device : copy.deepcopy(self.model).to(device) for device in devices}        
        # else:
        #     slave_models = {device : copy.deepcopy(self.model).to('cpu') for device in devices}
        return slave_models
    
    def average_gradients_to_master(self, master_model, grads_list):
        with torch.no_grad():
            for idx, param in enumerate(master_model.parameters()):
                stacked = torch.stack([g[idx] for g in grads_list])
                param.grad = stacked.mean(dim=0)
                
    
    def run_epoch(self, epoch, x_set_label, trainable=False, step_state=None):
        self.epoch = epoch

        gc.collect()
        torch.cuda.empty_cache()
        self.model.train(trainable)
       
        if step_state is not None:
            logging.info("Continuing saved epoch...")
        else:
            self.dataset.refresh_x_set(x_set_label, seed=self.pm.seed)
            logging.info("{} refreshed.".format(x_set_label))
        
        FG = self.pm.FG; T1 = self.pm.nFramesIn // FG
        if step_state is not None:
            _epoch_loss, _epoch_rmses = self.unpack_step_state(step_state)
            logging.info("Step state restored with saved one")
        else:
            _epoch_loss, _epoch_rmses = [], []
        
        if trainable: self.optimizer.zero_grad()                     # trainable
        pbar = zeus_tqdm(self.dataset)
        pbar.start_prefetch_thread()

        if self.nDevices > 1: executor = ThreadPoolExecutor(max_workers=len(self.devices))
        
        if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Fetch one batch asynchronously
        step = 0
        for batch in pbar:              # Wait for the asynchronous .prefetch
            if not pbar.no_more_data():  pbar.prefetch(x_set_label)        # Prefetch ensures getting new bacthes.
            if batch is None: break     # just in case
            batch_x, batch_y, batch_diff, era5_data_vars = batch    # (B, T1FG, C, H, W), (B, T2FG, C, H, W), None or (B, C, H, W), (C,)
            if self.pm.include_difficulty:
                pass
            batch_x = batch_x[:, :, :, :self.pm.image_height, :self.pm.image_width].pin_memory()     # torch.float32, cpu
            batch_y = batch_y[:, :, :, :self.pm.image_height, :self.pm.image_width].pin_memory()     # torch.float32, cpu
            era5_data_vars = era5_data_vars #.pin_memory() # torch.float32, cpu

            if self.pm.include_difficulty:
                batch_diff = batch_diff[:, :, :self.pm.image_height, :self.pm.image_width]

            # batch_x = batch_x.to(self.device, non_blocking=True)               # torch.float32, cuda
            # batch_y = batch_y.float().to(self.device_type + ':' + str(self.nDevices-1), non_blocking=True)       # torch.float32, cuda
            # era5_data_vars = era5_data_vars.to(self.device_type + ':' + str(self.nDevices-1), non_blocking=True) # torch.float32, cuda

            step_loss = []; step_rmses = []

            if trainable: self.optimizer.zero_grad() 
            
            if self.pm.abundant_GPUs or (not self.pm.abundant_GPUs and not trainable):
                with autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    with torch.set_grad_enabled(trainable):
                        pred = self.model(batch_x)
                    pred = self.dataset.unnormalize(pred)   # torch.float32 !!!, cuda, True !!!
                    target = self.dataset.unnormalize(batch_y)   # torch.float32, cuda, False
                    loss, rmse = self.loss(target, pred, batch_diff, era5_data_vars)  # (,), (C,), torch.float32, cuda, True !!!
                    if trainable:
                        loss.backward()
                step_loss.append(loss.detach().cpu())  # (,) - list.        Even if no cpu(), their stack results in cpu !!!
                step_rmses.append(rmse.detach().cpu())      # (C,) - list.  If no cpu(), their stack results in cuda

            elif not self.pm.abundant_GPUs and trainable:
                if self.nDevices <= 1:
                    loss, rmse = self.model.sigle_GPU_forward_loss_backward(batch_x, batch_y, self.loss, self.dataset.unnormalize, batch_diff, era5_data_vars)
                    step_loss.append(loss.detach().cpu())  # (,) - list.        Even if no cpu(), their stack results in cpu !!!
                    step_rmses.append(rmse.detach().cpu())      # (C,) - list.  If no cpu(), their stack results in cuda
                else:
                    chunks_x = batch_x.chunk(self.nDevices)
                    chunks_y = batch_y.chunk(self.nDevices)
                    
                    for device in self.devices:
                        self.slave_models[device].load_state_dict(self.model.state_dict())
                    
                    futures = [
                        executor.submit(
                            self.slave_models[self.devices[i]].sigle_GPU_forward_loss_backward, mb_x, mb_y, self.loss, self.dataset.unnormalize, 
                            batch_diff, era5_data_vars, self.devices[i]
                        )
                        for i, (mb_x, mb_y) in enumerate(zip(chunks_x, chunks_y))                        
                    ]   # returns immediately.
                    grads_list = []
                    for fut in futures:
                        grads, (loss, rmse) = fut.result()  # blocks until the single future is done.
                        grads_list.append(grads)
                        step_loss.append(loss.detach().to('cpu'))
                        step_rmses.append(rmse.detach().to('cpu'))
                    
                    self.average_gradients_to_master(self.model, grads_list)
                    
            if trainable:
                self.optimizer.step()                
                self.scheduler.step()   # Our scheduler is OneCycleLR, which is a per-batch scheduler

            step_loss = torch.tensor(step_loss).mean()   # (,)
            step_rmses = torch.stack(step_rmses, dim=0).mean(dim=0)   # (C,)
            strC = ", ".join(f"{x:.3f}" for x in step_rmses.ravel().tolist())
            pbar.set_description('Loss: {:.3f}, rmses: {}'.format(step_loss.item(), strC))

            _epoch_loss.append(step_loss)    # (,) - list
            _epoch_rmses.append(step_rmses)   # (C,) - list
            
            step += 1
            if trainable is True and step % self.pm.steps_to_save == 0:
                step_state = self.pack_step_state(_epoch_loss, _epoch_rmses)
                self.save_state(step_state=step_state)  # Step_save
                
        _epoch_loss = torch.stack(_epoch_loss, dim=0).mean(dim=0)  # (,)
        _epoch_rmses = torch.stack(_epoch_rmses, dim=0).mean(dim=0)       # (C,)
        strC = ", ".join(f"{x:.2f}" for x in _epoch_rmses.ravel().tolist())
        pbar.set_description('Loss: {:.3f}, rmses: {}'.format(_epoch_loss.item(), strC))
        pbar.stop_prefetch_thread()
        
        if trainable is True: self.save_state()   # Epoch_save. No step_state saved.
        if self.nDevices > 1: executor.shutdown()

        return _epoch_loss, _epoch_rmses  # (,), (C,)
    
    
    def compare_save_load(self):
            # Save the model's state
        torch.save(self.model.state_dict(), 'saved_model.pth')

        # Load the model's state back
        loaded_model = Zeus(self.config, self.config['zeusLayers'], device_type=self.device_type)
        loaded_model.load_state_dict(torch.load('saved_model.pth'))

        # Check if weights are the same
        for param1, param2 in zip(self.model.parameters(), loaded_model.parameters()):
            print(torch.allclose(param1, param2))

    
    def get_epoch_display(self, _epoch_loss, _epoch_rmses):
        loss = "Loss: {:.4f}".format(_epoch_loss.item())
        rmses = "RMSE: " + ", ".join(f"{x:.2f}" for x in _epoch_rmses.ravel().tolist())
        display = loss + ', ' + rmses
        return display
    
    def _print_tensor_state(self, n, t):
        print('-----------', n, t.dtype, t.device, t.requires_grad, t.shape)
    
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
        target = target.to(device)
        rmse = ((pred - target) ** 2).mean(dim=(0,1,3,4)).sqrt()    # (C,), 32, cuda, True !!!
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
        
        era5_data_vars = era5_data_vars.to(device)
        reward = rmse * era5_data_vars   # (C,), 32, cuda, True !!!, the-higher-the-worse
        # self._print_tensor_state(12, reward)
        reward = reward.mean()  # (,)

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
    
