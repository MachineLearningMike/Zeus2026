import torch
from torch import nn
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
        optimizer_name = getattr(self.pm, 'optimizer', 'adam').lower()

        if optimizer_name == 'adamw':
            OptimizerClass = torch.optim.AdamW
        elif optimizer_name == 'sgd':
            OptimizerClass = torch.optim.SGD
        else:
            OptimizerClass = torch.optim.Adam
            
        # Safely get trainable parameters as a list
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            raise ValueError("No trainable parameters found in the model.")

        # Define defaults clearly
        default_config = {
            'lr': 1e-6,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        }

        # Special default for weight_decay based on optimizer
        if optimizer_name.lower() == 'adam':
            default_config['weight_decay'] = 0.0
        else:
            default_config['weight_decay'] = 0.01

        # Retrieve values from self.pm or use defaults
        optimizer_config = {key: getattr(self.pm, key, default) for key, default in default_config.items()}

        # Create the optimizer
        optimizer = OptimizerClass(trainable_params, **optimizer_config)

        lr = optimizer.param_groups[0]['lr']
        wd = optimizer.param_groups[0]['weight_decay']
        logging.info(f"Optimizer created: {optimizer_name.upper()} with lr={lr}, weight_decay={wd}")

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
        
        if best_loss:
            name = self.get_best_loss_file_name()
        elif name is None: 
            name = "train_state_" + str(self.md_id)  # + '-' + self.get_train_id_string()
        else:
            assert name == "train_state_" + str(self.md_id)
        
        def get_complete_state_dict(model):
            """Get state dict including ALL buffers, not just registered ones"""
            state_dict = model.state_dict()
            
            # Also manually add any missing buffers
            for name, module in model.named_modules():
                for buffer_name, buffer in module.named_buffers():
                    full_name = name + ('.' if name else '') + buffer_name
                    if full_name not in state_dict:
                        state_dict[full_name] = buffer
            
            return state_dict
        
        checkpoint = {
            # 'model' : self.model,
            'model': self.model.state_dict(), # get_complete_state_dict(self.model), # self.model.state_dict(), # get_complete_state_dict(self.model),
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler':   self.scaler.state_dict(),
            'epoch': self.epoch,
            'weighted_metric_old': self.weighted_metric_old,
            'epochs_since_improvement': self.epochs_since_improvement,
            'train_finished' : train_finished,
            'packed_losses_metrics': self.packed_losses_metrics,
            'best_loss' : self.best_loss,
            'accum_iterations' : self.accum_iterations,
            }
        utils.check_dir(self.project_path)
        temp_path = self.project_path / (name + '.pth.tmp')
        torch.save(checkpoint, temp_path)
        shutil.move(temp_path, self.project_path / (name + '.pth'))
    
        self.save_train_history()        
        
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
        
        if choose_best_loss_model:
            name = self.get_best_loss_file_name()
            logging.info("!!!!!!!!!!!!!!!!!! Best loss model chosen.     Turn back off the switch.")
        elif name is None:
            name = "train_state_" + str(self.md_id) #+ self.get_train_id_string()
            logging.info("!!! Latest model chosen as usual.")
        else:
            assert name == "train_state_" + str(self.md_id), name
            logging.info("!!! Latest model chosen as usual.")
        
        file_not_found = False; train_finished = False; file_loaded = False

        try:
            checkpoint_path = self.project_path / (name + '.pth')
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # self.debug_scheduler_optimizer_alignment(checkpoint)
            logging.info("Checkpoint loaded from the .pth file")
        except Exception as e:
            logging.info(str(e))
            file_not_found = True

        if not file_not_found:
            train_finished = checkpoint['train_finished']
        
        if file_not_found and not self.trainable:
            logging.info("Train checkpoint not found")
            sys.exit()
        
        if not file_not_found and (not train_finished or not self.trainable):

            #------------------- model -------------------------
                
            # Load model parameters whetherr trainable or not.
            state_dict = checkpoint['model']
            
            if self.device != torch.device('cpu'):
                # Move ALL model tensors to correct device
                state_dict = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in state_dict.items()}
            
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            logging.info("Model loaded with saved state: \n_missing_keys: {}, \n_unexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))

            # Check if model has normalization layers
            norm_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                    nn.LayerNorm, nn.GroupNorm)):
                    norm_layers.append(name)
                    
            logging.info(f"Model has {len(norm_layers)} normalization layers: {norm_layers[:5]}...")

            # self.debug_normalization_issue()

            # Check running stats for a sample layer
            if norm_layers:
                sample_layer = norm_layers[0]
                for name, module in self.model.named_modules():
                    if name == sample_layer:
                        if hasattr(module, 'running_mean'):
                            logging.info(f"{name} running_mean mean: {module.running_mean.mean().item():.6f}")
                        if hasattr(module, 'running_var'):
                            logging.info(f"{name} running_var mean: {module.running_var.mean().item():.6f}")
                        break

            # Check what normalization stats are saved
            saved_keys = list(checkpoint['model'].keys())
            norm_stats_keys = [k for k in saved_keys if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
            logging.info(f"Found {len(norm_stats_keys)} normalization stat keys in checkpoint")
            if norm_stats_keys:
                logging.info(f"Sample: {norm_stats_keys[:5]}")

            #---------------------------------------------------------------------------------------
            packed_losses_metrics = None
            
            if self.trainable:

                epoch = checkpoint['epoch']
                
                #------------------- learning_rate -------------------------
                
                if self.learning_rate is None:
                    self.learning_rate = self.get_learning_rate()
                else:
                    self.set_learning_rate(self.learning_rate)
                    
                #------------------- optimizer -------------------------
                
                if self.optimizer is not None and "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    
                    correct_step, num_fixed = self.fix_mixed_step_values()  # to the most comon step value.

                    # Move optimizer state tensors to the same device as their parameters
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            if p in self.optimizer.state:
                                for k, v in self.optimizer.state[p].items():
                                    if torch.is_tensor(v):
                                        self.optimizer.state[p][k] = v.to(p.device, non_blocking=True)
                    logging.info("Optimizer loaded with saved state.")
                    
                    # self.debug_optimizer_state()
                    # self.debug_optimizer_parameter_references(checkpoint)
                    # self.debug_optimizer_state_existence_fixed()
                
                else:
                    logging.info("!!! Optimizer NOT loaded with saved state.")

                #------------------- scheduler -------------------------
                
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                logging.info("Scheduler loaded with saved state.")
                
                logging.info(f"Scheduler type: {type(self.scheduler).__name__}")

                # Check specific scheduler attributes
                if hasattr(self.scheduler, '_step_count'):
                    logging.info(f"Scheduler _step_count: {self.scheduler._step_count}")
                if hasattr(self.scheduler, 'last_epoch'):
                    logging.info(f"Scheduler last_epoch: {self.scheduler.last_epoch}")
                if hasattr(self.scheduler, 'get_last_lr'):
                    logging.info(f"Scheduler last_lr: {self.scheduler.get_last_lr()}")

                # Check against saved
                if 'scheduler' in checkpoint:
                    saved_scheduler = checkpoint['scheduler']
                    logging.info(f"Saved scheduler keys: {list(saved_scheduler.keys())}")
                
                #------------------- scaler -------------------------
                
                self.scaler.load_state_dict(checkpoint['scaler'])
                logging.info("Scaler loaded with saved state")

                #------------------ miscellaneous -------------------
                
                self.weighted_metric_old = checkpoint['weighted_metric_old']
                self.epochs_since_improvement = checkpoint['epochs_since_improvement']
                logging.info("weighted_metric_old and epochs_since_improvement loaded with saved values")

                packed_losses_metrics = checkpoint['packed_losses_metrics']
                logging.info(f"packed_losses_metrics loaded")
                    
                if self.pm.overwrite_saved_lr:
                    self.learning_rate = self.pm.lr
                else:
                    self.learning_rate = checkpoint['learning_rate']
                    logging.info(f"learning_rate loaded {self.learning_rate}")

                self.best_loss = checkpoint['best_loss']
                logging.info(f"best_loss loaded {self.best_loss}")

                self.accum_iterations = checkpoint['accum_iterations']
                logging.info(f"accum_iterations loaded {self.accum_iterations}")

            # print('----------- packed_losses_metrics loaded is ', packed_losses_metrics)
            if packed_losses_metrics is not None:  # The previous session was terminated after a new step was finished and before an epoch is finished.
                epoch = epoch
            else:   # The previous session was terminated after an epoch was finished and before a new step was finished. 
                epoch = epoch + 1

            logging.info(f"Checkpoint '{name}' loaded to {self.device}, including model, optimizer, scheduler, scaler, weighted_metric_old, and datase state")
            file_loaded = True
        
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
    
    
    def debug_normalization_issue(self):
        """Debug the normalization statistics issue"""
        
        print("\n" + "="*60)
        print("NORMALIZATION LAYER DEBUG")
        print("="*60)
        
        # List all normalization layers
        norm_layers = []
        for name, module in self.model.named_modules():
            if 'norm' in name.lower():
                norm_layers.append((name, type(module).__name__))
        
        print(f"Found {len(norm_layers)} layers with 'norm' in name")
        
        # Check first few
        for name, module_type in norm_layers[:10]:
            print(f"\n{name} ({module_type}):")
            
            module = None
            for m_name, m in self.model.named_modules():
                if m_name == name:
                    module = m
                    break
            
            if module is not None:
                # Check for BatchNorm attributes
                if hasattr(module, 'running_mean'):
                    print(f"  running_mean: shape={module.running_mean.shape}")
                    print(f"    mean value: {module.running_mean.mean().item():.6f}")
                else:
                    print(f"  No running_mean attribute")
                
                if hasattr(module, 'running_var'):
                    print(f"  running_var: shape={module.running_var.shape}")
                    print(f"    mean value: {module.running_var.mean().item():.6f}")
                else:
                    print(f"  No running_var attribute")
                
                if hasattr(module, 'weight'):
                    print(f"  weight: shape={module.weight.shape}")
                    print(f"    mean value: {module.weight.mean().item():.6f}")
                
                if hasattr(module, 'bias'):
                    print(f"  bias: shape={module.bias.shape if module.bias is not None else 'None'}")
                    if module.bias is not None:
                        print(f"    mean value: {module.bias.mean().item():.6f}")
    
    def fix_mixed_step_values(self):
        """Fix step values without bloating checkpoint size"""
        
        if not isinstance(self.optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            return
        
        logging.info("Fixing mixed step values efficiently...")
        
        # Find most common step as integer
        step_values = []
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param in self.optimizer.state and 'step' in self.optimizer.state[param]:
                    step = self.optimizer.state[param]['step']
                    # Convert to Python int
                    if hasattr(step, 'item'):
                        step_values.append(int(step.item()))
                    elif isinstance(step, torch.Tensor):
                        step_values.append(int(step.item()))
                    else:
                        step_values.append(int(step))
        
        if not step_values:
            return
        
        from collections import Counter
        most_common_step = Counter(step_values).most_common(1)[0][0]
        
        # Fix as integers
        fixed_count = 0
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param in self.optimizer.state and 'step' in self.optimizer.state[param]:
                    step = self.optimizer.state[param]['step']
                    
                    # Get current step as int
                    if hasattr(step, 'item'):
                        current_step = int(step.item())
                    elif isinstance(step, torch.Tensor):
                        current_step = int(step.item())
                    else:
                        current_step = int(step)
                    
                    if current_step != most_common_step:
                        # Store as Python integer (saves space!)
                        self.optimizer.state[param]['step'] = int(most_common_step)
                        fixed_count += 1
        
        logging.info(f"Fixed {fixed_count} step values to {most_common_step} (as integers)")
        return most_common_step, fixed_count
    
    
    def debug_optimizer_state_existence_fixed(self):
        """Check if optimizer state actually exists for parameters"""
        
        logging.info("=" * 60)
        logging.info("OPTIMIZER STATE EXISTENCE DEBUG V2")
        logging.info("=" * 60)
        
        total_params = 0
        params_with_state = 0
        state_keys_summary = {}
        
        # Check each parameter group
        for group_idx, group in enumerate(self.optimizer.param_groups):
            params_in_group = len(group['params'])
            logging.info(f"Group {group_idx}: {params_in_group} params, lr={group['lr']:.6f}")
            total_params += params_in_group
            
            # Check first few parameters in this group
            for param_idx, param in enumerate(group['params'][:3]):  # Check first 3 only
                if param in self.optimizer.state:
                    params_with_state += 1
                    state = self.optimizer.state[param]
                    
                    # Update key counts
                    for key in state.keys():
                        state_keys_summary[key] = state_keys_summary.get(key, 0) + 1
                    
                    # Log details for first parameter
                    if param_idx == 0:
                        logging.info(f"  First param state keys: {list(state.keys())}")
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                logging.info(f"    {key}: shape={value.shape}, "
                                            f"mean={value.mean().item():.6f}, "
                                            f"norm={value.norm().item():.6f}")
                            elif isinstance(value, (int, float)):
                                logging.info(f"    {key}: {value}")
        
        # Check remaining parameters (skip detailed logging)
        for group in self.optimizer.param_groups:
            for param in group['params'][3:]:  # Skip first 3 we already checked
                if param in self.optimizer.state:
                    params_with_state += 1
                    state = self.optimizer.state[param]
                    for key in state.keys():
                        state_keys_summary[key] = state_keys_summary.get(key, 0) + 1
        
        logging.info(f"\nSummary: {params_with_state}/{total_params} parameters have optimizer state")
        
        if state_keys_summary:
            logging.info("State keys distribution:")
            for key, count in sorted(state_keys_summary.items()):
                percentage = 100.0 * count / total_params
                logging.info(f"  {key}: {count} params ({percentage:.1f}%)")
        
        # Special check for Adam - FIXED VERSION
        if isinstance(self.optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            if 'step' in state_keys_summary:
                # Collect all step VALUES (not tensors)
                step_values = []
                for group in self.optimizer.param_groups:
                    for param in group['params']:
                        if param in self.optimizer.state and 'step' in self.optimizer.state[param]:
                            step_val = self.optimizer.state[param]['step']
                            # Convert tensor to scalar
                            if hasattr(step_val, 'item'):
                                step_values.append(step_val.item())
                            else:
                                step_values.append(step_val)
                
                if step_values:
                    # Check if all values are the same
                    unique_values = set(step_values)
                    if len(unique_values) == 1:
                        step_val = next(iter(unique_values))
                        logging.info(f"OK - All parameters at step {step_val}")
                        
                        # Estimate expected step
                        if hasattr(self, 'train_loader') and self.train_loader is not None:
                            batches_per_epoch = len(self.train_loader)
                            if hasattr(self, 'epoch'):
                                expected_step = self.epoch * batches_per_epoch
                                logging.info(f"  Based on epoch {self.epoch}, expected step ~{expected_step}")
                    else:
                        logging.warning(f"WARNING - Mixed step VALUES: {unique_values}")
                        logging.warning(f"  Found {len(unique_values)} different step values")
                        
                        # Show distribution
                        from collections import Counter
                        step_counts = Counter(step_values)
                        for step_val, count in sorted(step_counts.items()):
                            percentage = 100.0 * count / len(step_values)
                            logging.warning(f"    Step {step_val}: {count} params ({percentage:.1f}%)")
            else:
                logging.error("ERROR - ADAM OPTIMIZER HAS NO 'step' COUNTERS!")
                logging.error("  Bias correction will be wrong!")
        
        return params_with_state, state_keys_summary
    

    def debug_optimizer_parameter_references(self, checkpoint):
        """Check if optimizer points to same parameters as model"""
        
        logging.info("=" * 60)
        logging.info("OPTIMIZER PARAMETER REFERENCE DEBUG")
        logging.info("=" * 60)
        
        # 1. Get model parameter IDs
        model_param_ids = {id(p): name for name, p in self.model.named_parameters()}
        logging.info(f"Model has {len(model_param_ids)} parameters")
        
        # 2. Get optimizer parameter IDs
        optimizer_param_ids = set()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                optimizer_param_ids.add(id(p))
        
        logging.info(f"Optimizer references {len(optimizer_param_ids)} parameters")
        
        # 3. Check overlap
        common = optimizer_param_ids.intersection(model_param_ids)
        missing_in_optimizer = set(model_param_ids.keys()) - optimizer_param_ids
        extra_in_optimizer = optimizer_param_ids - set(model_param_ids.keys())
        
        logging.info(f"Common parameters: {len(common)}")
        logging.info(f"Parameters in model but NOT in optimizer: {len(missing_in_optimizer)}")
        logging.info(f"Parameters in optimizer but NOT in model: {len(extra_in_optimizer)}")
        
        if len(missing_in_optimizer) > 0:
            logging.error("OPTIMIZER DOESN'T REFERENCE ALL MODEL PARAMETERS!")
            # Show first few missing
            for pid in list(missing_in_optimizer)[:3]:
                name = model_param_ids[pid]
                logging.error(f"  Missing: {name}")
        
        # 4. Check if optimizer state matches parameters
        if len(missing_in_optimizer) == 0:
            # Check if state exists for parameters
            params_with_state = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p in self.optimizer.state:
                        params_with_state += 1
            
            logging.info(f"Parameters with optimizer state: {params_with_state}/{len(common)}")
            
            if params_with_state == 0:
                logging.warning("NO optimizer state exists yet!")
                logging.warning("  State will be created on first optimizer.step()")
                logging.warning("  Loaded checkpoint state will be IGNORED!")
        
        # 5. Check saved checkpoint - FIXED VERSION
        if 'optimizer' in checkpoint:
            saved_state = checkpoint['optimizer'].get('state', {})
            logging.info(f"Saved checkpoint has state for {len(saved_state)} parameters")
            
            # Do saved parameter IDs match current?
            if saved_state:
                # Convert model.parameters() generator to list
                model_params_list = list(self.model.parameters())
                
                # Check count
                if len(saved_state) != len(model_params_list):
                    logging.warning(f"Parameter count mismatch: saved={len(saved_state)}, current={len(model_params_list)}")
                
                # Also check if optimizer has same number of params as model
                optimizer_param_count = sum(len(g['params']) for g in self.optimizer.param_groups)
                if optimizer_param_count != len(model_params_list):
                    logging.warning(f"Optimizer-model count mismatch: optimizer={optimizer_param_count}, model={len(model_params_list)}")


    def debug_optimizer_state(self):
        """Log optimizer state details"""
        
        if not self.trainable or self.optimizer is None:
            return
        
        optimizer_type = type(self.optimizer).__name__
        logging.info(f"Optimizer type: {optimizer_type}")
        
        # Check momentum buffers
        total_params = 0
        params_with_state = 0
        momentum_sum = 0
        
        for i, group in enumerate(self.optimizer.param_groups):
            logging.info(f"  Group {i}: lr={group['lr']}, params={len(group['params'])}")
            
            for p in group['params']:
                total_params += 1
                if p in self.optimizer.state:
                    params_with_state += 1
                    state = self.optimizer.state[p]
                    
                    # Check what's in state
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            # Log first few values
                            if key in ['momentum_buffer', 'exp_avg', 'exp_avg_sq']:
                                flat_val = value.flatten()
                                if len(flat_val) > 0:
                                    momentum_sum += flat_val[:3].abs().mean().item()
        
        logging.info(f"  Parameters with state: {params_with_state}/{total_params}")
        logging.info(f"  Momentum magnitude indicator: {momentum_sum}")
        
        # Compare with saved checkpoint
        if hasattr(self, '_last_checkpoint'):
            saved_optimizer = self._last_checkpoint.get('optimizer', {})
            if 'state' in saved_optimizer:
                saved_states = len(saved_optimizer['state'])
                logging.info(f"  Saved had {saved_states} parameter states")

    def debug_scheduler_optimizer_alignment(self, checkpoint):
        """Check if scheduler and optimizer are aligned"""
        
        logging.info("=" * 60)
        logging.info("SCHEDULER-OPTIMIZER ALIGNMENT DEBUG")
        logging.info("=" * 60)
        
        # 1. Get optimizer step
        optimizer_step = None
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'step' in self.optimizer.state[p]:
                    optimizer_step = self.optimizer.state[p]['step']
                    break
            if optimizer_step is not None:
                break
        
        # 2. Get scheduler step/epoch
        scheduler_info = {}
        
        # Common scheduler attributes
        scheduler_attrs = ['_step_count', 'last_epoch', 'T_cur', 't_epoch', 'step_count']
        for attr in scheduler_attrs:
            if hasattr(self.scheduler, attr):
                scheduler_info[attr] = getattr(self.scheduler, attr)
        
        # 3. Check saved scheduler state
        saved_scheduler = checkpoint.get('scheduler', {})
        saved_step_count = saved_scheduler.get('_step_count', None)
        saved_last_epoch = saved_scheduler.get('last_epoch', None)
        
        # 4. Log everything
        logging.info(f"Optimizer step: {optimizer_step}")
        logging.info(f"Scheduler state: {scheduler_info}")
        logging.info(f"Saved scheduler: _step_count={saved_step_count}, last_epoch={saved_last_epoch}")
        
        # 5. Check consistency
        if optimizer_step is not None and '_step_count' in scheduler_info:
            if scheduler_info['_step_count'] != optimizer_step:
                logging.error(f"MISMATCH: scheduler._step_count={scheduler_info['_step_count']}, "
                            f"optimizer.step={optimizer_step}")
                
                # This is likely your problem!
                # The scheduler and optimizer are out of sync
                logging.error("  Scheduler and optimizer step counters are out of sync!")
                logging.error("  This causes wrong learning rate calculations!")
                
                # Calculate what it should be
                if saved_step_count is not None:
                    logging.info(f"  Saved scheduler had _step_count={saved_step_count}")
                    logging.info(f"  Difference: {scheduler_info['_step_count'] - saved_step_count}")
        
        # 6. Check learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        saved_lr = checkpoint.get('learning_rate', None)
        logging.info(f"Learning rate: current={current_lr}, saved={saved_lr}")
        
        if saved_lr is not None and abs(current_lr - saved_lr) > 1e-6:
            logging.warning(f"⚠️ Learning rate changed: {saved_lr} -> {current_lr}")


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
        

    def forward_backward_no_optimizer_step(self,
            X0, X1, Y, cascade_step,
            fn_loss, fn_nornalize, fn_unnornalize, 
            batch_diff, era5_data_vars, trainable,
            type_metrics, predicted_value_required,
            pressure_indices, pressure_values,
            scaler,
        ):
        
        return self.model.forward_backward_no_optimizer_step(
            X0, X1, Y, cascade_step,
            fn_loss, fn_nornalize, fn_unnornalize, 
            batch_diff, era5_data_vars, trainable,
            type_metrics, predicted_value_required,
            pressure_indices, pressure_values,
            scaler,
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
    