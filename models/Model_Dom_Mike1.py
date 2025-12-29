import torch
import numpy as np
import json, tzlocal
from torch.amp import GradScaler
from einops import rearrange
from pathlib import Path
import logging
import sys
import shutil
from importlib import import_module


sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.utils import check_dir, check_gradient

from models.config_Mike1 import config
module = import_module(config['Zeus_Model_module'])
Zeus = getattr(module, 'Zeus')

# from models.Zeus_ResNet import Zeus
# # from Zeus_Flash2D_Blocked import Zeus
# from Zeus_Mike1 import Zeus

import modules.utils as utils
from modules_zeus.constants import ERA5_DATA_VARS
# from modules_zeus.constants import (
#     # REWARD_DIFFICULTY_SCALER,  
#     # REWARD_IMPROVEMENT_WEIGHT,
#     # REWARD_IMPROVEMENT_MIN_DELTA,
# )

from AdaptiveLRLearner import AdaptiveLRLearner

def print_data_attrs(obj):
    for name, value in vars(obj).items():
        print(f"{name}: {value}")


class Model_Dom():
    def __init__(self, md_id, device, pm, dataset, management_path, project_path, bayes=None, trainable=True):
        self.md_id = md_id
        self.device = device
        self.pm = pm
        self.dataset = dataset
        # print_data_attrs(self.pm)
                
        self.project_path = project_path
        check_dir(self.project_path)

        self.bayes = bayes
        self.trainable = trainable
        
        self.scaler = GradScaler()
        
        # try:
        if not self.pm.doubleZeus:
            self.model = Zeus(self.pm, device=self.device).to(self.device)
            num_params = sum(p.numel() for p in self.model.parameters())
            logging.info("Zeus model {} created with a total of ========= {:,} ========= parameters".format(self.md_id, num_params))
            # if self.nDevices > 1:
            #     self.slave_models = self.create_slave_models(self.devices)
        else:
            pass

        # except Exception as e:
        #     logging.info("Creating model...." + str(e))
        #     sys.exit(0)

        if self.trainable:
            self.optimizer = self._select_optimizer()
            self.scheduler = self._select_scheduler(self.optimizer)
            self.scaler = GradScaler(device=self.device)

            self.epoch = 0
            self.packed_losses_metrics = None
            self.epochs_since_improvement = 0
            self.weighted_metric_old = torch.ones_like(self.load_era5_data_vars()) * 1e9
            self.learning_rate = None
            
        if self.trainable:
            self.lr_controller = AdaptiveLRLearner(self.optimizer, accum_steps=self.pm.accum_steps, init_lr=self.get_learning_rate())

        # work variables            
        self.total_losses = []
        self.epoch_losses, self.epoch_metrics = [], []
        self.pbar_description = None
        self.step_loss = None
        self.best_loss = 1e9
        self.accum_iterations = 0
    
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
        total_steps = self.pm.total_steps
        anneal_strategy =  getattr(self.pm, 'anneal_strategy', 'linear')
        div_factor = getattr(self.pm, 'div_factor', 25.0)
        final_div_factor = getattr(self.pm, 'final_div_factor', 1e4)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(    # Mike!!! This is a per-batch scheduler.
            # A per-batch scheduler's time axis is on mini-batch, not on epochs.
            # It only counts how many mini-batches it traverses, across epochs, and doesn't aware of epochs.
            # steps_per_epoch and epochs are only used to get total_steps.
            
            # (initial_lr = max_lr / div_factor, for pct_start of the total_steps) -> max_lr -> (Final LR = initial_lr/final_div_factor, for the remaining steps)
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            # suppress_warnings=True,
        )

        logging.info(f"Scheduler created: OneCycleLR with max_lr={max_lr}, pct_start={pct_start}, total_steps={total_steps}, div_factor={div_factor}, final_div_factor={final_div_factor}")

        return scheduler
    
    def load_era5_data_vars(self):
        vars = [ERA5_DATA_VARS[k] for k in self.pm.variables if k in ERA5_DATA_VARS.keys()]
        self.era5_data_vars = torch.tensor(vars)
        return self.era5_data_vars
    
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.scheduler.base_lrs = [lr for lr in self.scheduler.base_lrs]
        self.learning_rate = lr
        
    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def save_best_train(self, vali_metric):   # (C,), (C,)        
        if vali_metric is not None:
            wasBestEpoch = False

            weighted_metric = vali_metric * (self.era5_data_vars.to(vali_metric.device))

            if torch.mean(weighted_metric) < torch.mean(self.weighted_metric_old):
                wasBestEpoch = True
                self.weighted_metric_old = weighted_metric
                name = "best_model_" + str(self.md_id)
                logging.info("saving best train_state...model: {}, weighted_metric: {:.3f}, path: {}".format(self.md_id, torch.mean(weighted_metric).item(), name))
                self.save_state(name=name, model_only=True)

            self.epochs_since_improvement = self.epochs_since_improvement + 1 if not wasBestEpoch else 0
            
            keep_going = self.epochs_since_improvement < self.pm.epoch_patience
            logging.info("___Patience {} run out @ model {}".format(self.pm.epoch_patience, self.md_id))
            
            best_metric = torch.mean(self.weighted_metric_old).item()
        else:
            logging.info("Model {} has no validation metrics. Going ahead to next epoch...".format(self.md_id))
            
            keep_going = True
            best_metric = 1e9
            
        return keep_going, best_metric
    
    def get_best_loss_file_name(self):
        return "best_loss_model_" + str(self.md_id)
    
    def save_best_loss_model(self, loss):
        if loss < 0.4 and loss < self.best_loss:    # 0.4: hard-coded.
            self.best_loss = loss
            logging.info("saving best-loss model... md {}, loss {:.3f}, iter {}".format(self.md_id, loss, self.accum_iterations))
            self.save_state(best_loss=True)
       
    def save_state(self, name=None, train_finished=False, best_loss=False):
    #-------------- My version
        assert self.trainable, "Non-trainable train object cannot be saved."
        
        if name is None: name = "train_state_" + str(self.md_id)  # + '-' + self.get_train_id_string()
        if best_loss:
            name = self.get_best_loss_file_name()
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler':   self.scaler.state_dict(),
            'epoch': self.epoch,
            'weighted_metric_old': self.weighted_metric_old,
            'epochs_since_improvement': self.epochs_since_improvement,
            'train_finished' : train_finished,
            'packed_losses_metrics': self.packed_losses_metrics,
            'learning_rate': self.learning_rate,
            'best_loss' : self.best_loss,
            'accum_iterations' : self.accum_iterations,
            }
        utils.check_dir(self.project_path)
        temp_path = self.project_path / (name + '.pth.tmp')
        torch.save(checkpoint, temp_path)
        shutil.move(temp_path, self.project_path / (name + '.pth'))
    
        self.save_train_history()        
    
        # #------------------------- Test --------------------
        # checkpoint = torch.load(self.project_path / (name + '.pth'), map_location="cpu")
        # state_dict = checkpoint['model']
        # missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        # logging.info("Model loaded with saved state: \n_missing_keys: {}, \n_unexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))
    
    
        return
        
    def save_train_history(self, name=None):
        try:
            if name is None: name = "train_history_" + str(self.md_id)
            data = np.array(self.total_losses, dtype=np.float32)
            np.savez(str(self.project_path / name) + ".npz", total_losses = data)
        except:
            pass

    def load_history(self, name=None):
        try:
            if name is None: name = "train_history_"  + str(self.md_id)
            loaded = np.load(str(self.project_path / name) + ".npz")
            total_losses = list(loaded["total_losses"])
        except:
            total_losses = []
        return total_losses
    
    def try_load_state(self, name=None):    # Pairs with save_state
        choose_best_loss_model=self.pm.choose_best_loss_model
        
        if name is None: name = "train_state_" + str(self.md_id) #+ self.get_train_id_string()
        if choose_best_loss_model:
            name = self.get_best_loss_file_name()
            logging.info("!!!!!!!!!!!!!!!!!! Latest model chosen.     Mind of the switch.")
        elif name is None:
            name = "train_state_" + str(self.md_id) #+ self.get_train_id_string()
            logging.info("!!! Latest model chosen as usual.")
        else:
            assert name == "train_state_" + str(self.md_id), name
        
        file_not_found = False; train_finished = False; file_loaded = False

        # checkpoint_path = self.project_path / ("model_params" + '.pth')
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model"], strict=False)
        # logging.info("========== Model loaded with saved state: \n_missing_keys: {}, \n_unexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))

        try:
            checkpoint_path = self.project_path / (name + '.pth')
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logging.info("Checkpoint loaded from the .pth file")
        except Exception as e:
            logging.info(str(e))
            file_not_found = True

        if not file_not_found:
            if checkpoint.get('train_finished', None) is not None:
                train_finished = checkpoint['train_finished']
                train_finished = False  # Mike!!!
            else:
                logging.info("Train_finished not found")
                sys.exit()
        
        if file_not_found and not self.trainable:
            logging.info("Train checkpoint not found")
            sys.exit()
        
        if not file_not_found and (not train_finished or not self.trainable):
            try:
                # Load model parameters
                state_dict = checkpoint['model']
                
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
                    logging.info("Model loaded with saved state: \n_missing_keys: {}, \n_unexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))

                packed_losses_metrics = None
                if self.trainable:
                    
                    try:
                        epoch = checkpoint['epoch']                       
                        
                        if not choose_best_loss_model and self.optimizer is not None and "optimizer" in checkpoint:
                            self.optimizer.load_state_dict(checkpoint["optimizer"])

                            # Move optimizer state tensors to the same device as their parameters
                            for group in self.optimizer.param_groups:
                                for p in group["params"]:
                                    if p in self.optimizer.state:
                                        for k, v in self.optimizer.state[p].items():
                                            if torch.is_tensor(v):
                                                self.optimizer.state[p][k] = v.to(p.device, non_blocking=True)
                            logging.info("Optimizer loaded with saved state.")
                        else:
                            logging.info("!!! Optimizer NOT loaded with saved state.")

                        self.scheduler.load_state_dict(checkpoint['scheduler'])
                        logging.info("Scheduler loaded with saved state.")
                        
                        self.scaler.load_state_dict(checkpoint['scaler'])
                        logging.info("Scaler loaded with saved state")

                        self.weighted_metric_old = checkpoint['weighted_metric_old']
                        self.epochs_since_improvement = checkpoint['epochs_since_improvement']
                        logging.info("weighted_metric_old and epochs_since_improvement loaded with saved values")

                        if checkpoint.get('packed_losses_metrics', None) is not None:
                            packed_losses_metrics = checkpoint['packed_losses_metrics']
                            logging.info(f"packed_losses_metrics loaded")
                            
                        if self.pm.overwrite_saved_lr:
                            self.learning_rate = self.pm.lr
                        elif checkpoint.get('learning_rate', None) is not None:
                            self.learning_rate = checkpoint['learning_rate']
                            logging.info(f"learning_rate loaded {self.learning_rate}")
                        if self.learning_rate is None:
                            self.learning_rate = self.get_learning_rate()
                        else:
                            self.set_learning_rate(self.learning_rate)

                        if checkpoint.get('best_loss', None) is not None:
                            self.best_loss = checkpoint['best_loss']
                            
                            # self.best_loss = 1e9        # start over!
                            
                            logging.info(f"best_loss loaded {self.best_loss}")
                        if checkpoint.get('accum_iterations', None) is not None:
                            self.accum_iterations = checkpoint['accum_iterations']
                            logging.info(f"accum_iterations loaded {self.accum_iterations}")

                    except Exception as e:
                        logging.info(str(e))

                # print('----------- packed_losses_metrics loaded is ', packed_losses_metrics)
                if packed_losses_metrics is not None:  # The previous session was terminated after a new step was finished and before an epoch is finished.
                    epoch = epoch
                else:   # The previous session was terminated after an epoch was finished and before a new step was finished. 
                    epoch = epoch + 1

                logging.info(f"Checkpoint '{name}' loaded to {self.device}, including model, optimizer, scheduler, scaler, weighted_metric_old, and datase state")
                file_loaded = True
            except Exception as e:
                logging.info(str(e))
                file_loaded = False
        
        else:
            pass
        
        if file_not_found:               
            # Start a new training, creating the file.
            epoch = 1; packed_losses_metrics = None    # This is the moment where an existing train will be overriden !!!    
        elif train_finished and self.trainable:
            # Keep the finished train intact.
            logging.info("You tried to start/continue a finished training. Aborting... No! specially")
            sys.exit()
        elif not file_loaded:
            # Start a new training, creating the file.
            epoch = 1; packed_losses_metrics = None    # This is the moment where an existing train will be overriden !!!    
        else: # file found, train not finished, file loaded
            # Continue the loaded train.
            epoch = epoch   # This is actually the previous epoch 
            packed_losses_metrics = packed_losses_metrics
                        
        self.total_losses = self.load_history()
        logging.info("train history loaded. length: {}".format(len(self.total_losses)))

        self.unpack_losses_metrics(packed_losses_metrics)

        self.epoch, self.packed_losses_metrics = epoch, packed_losses_metrics        
        
        return self.epoch, self.packed_losses_metrics


    def get_epoch_display_batch(self, loss_batch, metrics_batch):       
        lead = "model {}, ".format(self.md_id)
        if loss_batch is not None and len(loss_batch) > 0:
            loss = np.array(loss_batch).mean()
            loss = "loss {:.4f}, ".format(loss)
        else:
            loss = "loss=None, "
        if metrics_batch is not None and len(metrics_batch) > 0:
            metrics = list(np.array(metrics_batch).mean(axis=0))
            metrics = self.pm.metrics + ": " + ", ".join(f"{x:.4f}" for x in metrics)                        
        else:
            metrics = "metrics=None"
        self.display = lead + loss + metrics
        return self.display

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
    
    def zero_losses_metrics(self):
        self.epoch_losses, self.epoch_metrics = [], []
        self.packed_losses_metrics = {}
        
    #---------------------------- protocol for loss and metrics
    #
    #      self.epoch_losses:   a list of scalars
    #      self.epoch_metrics:  a list of C-long lists of scalars
    #
    #-----------------------------------------------------------
        
    def pack_losses_metrics(self):
        assert self.epoch_losses is not None and self.epoch_metrics is not None
        self.packed_losses_metrics = {
            "epoch_losses": self.epoch_losses,          # each scalar -> float
            "epoch_metrics": self.epoch_metrics         # each (4,) tensor -> list    
        }
        return self.packed_losses_metrics

    def unpack_losses_metrics(self, packed_losses_metrics):
        self.epoch_losses, self.epoch_metrics = [], []
        if packed_losses_metrics is not None:
            self.epoch_losses = packed_losses_metrics["epoch_losses"]
            self.epoch_metrics = packed_losses_metrics["epoch_metrics"]
            
    def get_pbar_description_epoch(self):    # (steps,), (stept, C)
        if len(self.epoch_losses) <= 0 or len(self.epoch_metrics) <= 0: 
            self.pbar_description = "md.epoch_losses and md.epoch_metrics are empty"
        else:
            _epoch_loss = np.array(self.epoch_losses).mean().item()  # (,)
            _epoch_metrics = np.array(self.epoch_metrics).mean(axis=0)       # (C,)
            strC = ", ".join(f"{x:.5g}" for x in _epoch_metrics)
            self.pbar_description = "md: {}, loss: {:.4f}, {}: ({})".format(self.md_id, _epoch_loss, self.pm.metrics, strC)

    def get_pbar_description_step(self, step_loss, step_metrics):   # (,), (C,)
        strC = ", ".join(f"{x:.3g}" for x in step_metrics[-len(self.pm.targetVariables):])
        # self.pbar_description = "md {} loss {:.3f}, {} ({})".format(self.md_id, step_loss, self.pm.metrics, strC)
        self.pbar_description = "loss {:.3g}, {} ({}), acc {}".format(step_loss, self.pm.metrics, strC, self.accum_iterations)
        
    def stepOptimizer(self, loss_item):
        performed_step, lr, grad_norm = self.lr_controller.stepOptimizer(loss_item)
        

    def forward_backward_no_step(self,
            X0, X1, Y, cascade_step,
            fn_loss, fn_nornalize, fn_unnornalize, 
            batch_diff, era5_data_vars, trainable,
            type_metrics, predicted_value_required,
            pressure_indices, pressure_values,
        ):
        
        return self.model.forward_backward_no_step(
            X0, X1, Y, cascade_step,
            fn_loss, fn_nornalize, fn_unnornalize, 
            batch_diff, era5_data_vars, trainable,
            type_metrics, predicted_value_required,
            pressure_indices, pressure_values,
            self.scaler,
        )
        
    def store_step_loss_and_metrics(self, step_loss, step_metrics): # [substep_loss...], [substep_metrics...]
        loss = torch.stack(step_loss).mean().item()
        self.epoch_losses.append(loss)
        self.total_losses.append(loss)
        
        metrics = torch.stack(step_metrics, dim=0).mean(dim=0).tolist()
        # metrics = list(np.array(step_metrics).mean(axis=0))
        self.epoch_metrics.append(metrics)

        self.step_loss = loss
        self.get_pbar_description_step(loss, metrics)
    