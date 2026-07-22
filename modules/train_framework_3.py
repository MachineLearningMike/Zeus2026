import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading
import queue
from tqdm import tqdm
import time
import os
import json
from pathlib import Path
import random
import signal
import sys
import atexit
import math


class ThreadedDataLoader:
    """DataLoader that pre-fetches next batch in a separate thread."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.current_epoch_indices = None
        self.current_batch_start = 0
        
        self.iterator = None
        self.prefetch_queue = queue.Queue(maxsize=1)
        self.prefetch_thread = None
        self.load_start_time = None
        
    def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
        self.epoch = epoch
        
        if resume_indices is not None:
            self.current_epoch_indices = resume_indices
        else:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(self.current_epoch_indices)
        
        self.current_batch_start = resume_batch_start
        self._create_iterator()
        self._start_prefetch()
    
    def _create_iterator(self):
        start = self.current_batch_start * self.batch_size
        remaining_indices = self.current_epoch_indices[start:]
        self.iterator = self._batch_generator(remaining_indices)
    
    def _batch_generator(self, indices):
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            X_list = []
            Y_list = []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                X_list.append(x)
                Y_list.append(y)
            
            X_batch = torch.stack(X_list)
            Y_batch = torch.stack(Y_list)
            
            yield X_batch, Y_batch
    
    def get_total_batches_in_epoch(self):
        """Return total number of batches in current epoch (including partial last batch)."""
        if self.current_epoch_indices is None:
            return math.ceil(len(self.dataset) / self.batch_size)
        return math.ceil(len(self.current_epoch_indices) / self.batch_size)
    
    def get_state(self):
        return {
            'epoch_indices': self.current_epoch_indices,
            'batch_start': self.current_batch_start,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
        }
    
    def restore_state(self, state):
        self.current_epoch_indices = state['epoch_indices']
        self.current_batch_start = state['batch_start']
    
    def _start_prefetch(self):
        self.load_start_time = time.time()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
    def _prefetch_worker(self):
        try:
            batch = next(self.iterator)
            load_time = time.time() - self.load_start_time
            self.prefetch_queue.put(('data', batch, load_time))
        except StopIteration:
            self.prefetch_queue.put(('stop', None, 0))
        except Exception as e:
            self.prefetch_queue.put(('error', e, 0))
    
    def get_next_batch(self):
        wait_start = time.time()
        status, data, load_time = self.prefetch_queue.get()
        wait_time = time.time() - wait_start
        
        if status == 'stop':
            return None, wait_time, load_time
        elif status == 'error':
            raise data
        
        self.current_batch_start += 1
        self._start_prefetch()
        return data, wait_time, load_time


class TrainingState:
    """Manages saving and loading of complete training state with atomic writes."""
    
    def __init__(self, save_dir='checkpoints', save_every_n_minibatches=100, 
                 max_checkpoints=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_minibatches = save_every_n_minibatches
        self.max_checkpoints = max_checkpoints
        self.state_file = self.save_dir / 'training_state.json'
        
    def save_checkpoint(self, model, optimizer, dataloader_state, epoch, batch_idx, 
                        total_minibatches_processed, loss, is_emergency=False):
        """Save checkpoint with atomic write to prevent corruption."""
        checkpoint = {
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataloader_state': dataloader_state,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
        }
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        prefix = 'emergency' if is_emergency else 'checkpoint'
        checkpoint_file = self.save_dir / f'{prefix}_minibatch{total_minibatches_processed}_{timestamp}.pt'
        
        # Atomic save: write to temp file, then rename
        temp_file = checkpoint_file.with_suffix('.tmp')
        torch.save(checkpoint, temp_file)
        os.replace(temp_file, checkpoint_file)
        
        state_info = {
            'latest_checkpoint': str(checkpoint_file),
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
        }
        
        # Atomic state file update
        temp_state = self.state_file.with_suffix('.tmp')
        with open(temp_state, 'w') as f:
            json.dump(state_info, f, indent=2)
        os.replace(temp_state, self.state_file)
        
        self._cleanup_old_checkpoints()
        
        if is_emergency:
            print(f"\n{'!'*60}")
            print(f"EMERGENCY CHECKPOINT SAVED at minibatch {total_minibatches_processed}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            print(f"{'!'*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Checkpoint saved at minibatch {total_minibatches_processed}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            print(f"{'='*60}")
        
    def load_latest_checkpoint(self, model, optimizer, dataloader):
        """Load the latest checkpoint. Returns None if nothing to resume."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state_info = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("⚠ State file corrupted. Starting fresh.")
            return None
        
        latest_checkpoint = Path(state_info['latest_checkpoint'])
        
        if not latest_checkpoint.exists():
            print(f"⚠ Checkpoint file not found. Starting fresh.")
            return None
        
        print(f"Found checkpoint: {latest_checkpoint.name}")
        
        try:
            checkpoint = torch.load(latest_checkpoint)
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}. Starting fresh.")
            return None
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        total_minibatches_processed = checkpoint['total_minibatches_processed']
        loss = checkpoint['loss']
        is_emergency = checkpoint.get('is_emergency', False)
        
        if 'dataloader_state' in checkpoint:
            dataloader_state = checkpoint['dataloader_state']
            dataloader.restore_state(dataloader_state)
            
            if is_emergency:
                print(f"⚠ Resuming from EMERGENCY checkpoint")
            else:
                print(f"Resuming from checkpoint")
            print(f"  Minibatch: {total_minibatches_processed}")
            print(f"  Epoch {epoch+1}, batch {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            
            return epoch, batch_idx, total_minibatches_processed, loss, dataloader_state, is_emergency
        
        return None
    
    def _cleanup_old_checkpoints(self):
        all_checkpoints = sorted(
            list(self.save_dir.glob('checkpoint_minibatch*.pt')) + 
            list(self.save_dir.glob('emergency_minibatch*.pt')),
            key=os.path.getmtime,
            reverse=True
        )
        
        for old_file in all_checkpoints[self.max_checkpoints:]:
            old_file.unlink()


class ResumableTrainer:
    """
    Trainer with robust handling of all interruption scenarios:
    - Normal run: Continuous training with periodic checkpoints
    - Ctrl+C once: Graceful stop after current batch (save & exit)
    - Ctrl+C twice: Force quit without saving
    - SIGTERM: Cloud preemption - emergency save immediately
    - Power failure: atexit emergency save (best effort)
    """
    
    def __init__(self, model, loss_fn, optimizer, dataloader, 
                 save_dir='checkpoints', device='cpu',
                 save_every_n_minibatches=100):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        
        self.state_manager = TrainingState(
            save_dir=save_dir,
            save_every_n_minibatches=save_every_n_minibatches
        )
        
        # Control flags
        self.stop_requested = False
        self.is_running = False
        self.ctrl_c_count = 0
        
        # Training state
        self.total_minibatches_processed = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        
        # Register handlers
        self._setup_signal_handlers()
        atexit.register(self._at_exit_handler)
        
    def _setup_signal_handlers(self):
        """Setup handlers for all interruption scenarios."""
        
        def sigint_handler(signum, frame):
            """Ctrl+C: once = graceful, twice = force quit"""
            self.ctrl_c_count += 1
            
            if self.ctrl_c_count == 1:
                print(f"\n\n⚠ Ctrl+C pressed once. Will save after current batch.")
                print("Press Ctrl+C again to force quit WITHOUT saving.")
                self.stop_requested = True
            else:
                print(f"\n\n⚠ Ctrl+C pressed twice. Force quitting NOW (no save).")
                sys.exit(1)
        
        def sigterm_handler(signum, frame):
            """SIGTERM: Cloud preemption - save immediately"""
            print(f"\n\n⚠ CLOUD PREEMPTION / SIGTERM received!")
            print("Saving emergency checkpoint NOW...")
            self._emergency_save()
            print("Exiting.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigterm_handler)
    
    def _at_exit_handler(self):
        """Called on Python exit - last resort for power failure."""
        if self.is_running and self.total_minibatches_processed > 0:
            try:
                print(f"\n⚠ ATEXIT: Attempting emergency save...")
                self._emergency_save()
                print("Emergency checkpoint saved.")
            except Exception as e:
                print(f"Could not save on exit: {e}")
    
    def _emergency_save(self):
        """Save checkpoint immediately for emergencies."""
        try:
            dataloader_state = self.dataloader.get_state()
            
            self.state_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                dataloader_state=dataloader_state,
                epoch=self.current_epoch,
                batch_idx=self.current_batch,
                total_minibatches_processed=self.total_minibatches_processed,
                loss=self.current_loss,
                is_emergency=True
            )
        except Exception as e:
            print(f"Emergency save failed: {e}")
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.current_epoch,
                    'total_minibatches_processed': self.total_minibatches_processed,
                }, self.state_manager.save_dir / 'last_resort.pt')
                print("Bare minimum checkpoint saved.")
            except:
                print("CRITICAL: Could not save anything!")
    
    def train(self, num_epochs, resume=True):
        """Train with full interruption resilience."""
        
        # Try to resume from checkpoint
        if resume:
            loaded = self.state_manager.load_latest_checkpoint(
                self.model, self.optimizer, self.dataloader
            )
            
            if loaded:
                (self.start_epoch, self.start_batch, 
                 self.total_minibatches_processed, _, _, is_emergency) = loaded
                
                # Calculate total batches in the saved epoch
                total_batches_in_epoch = self.dataloader.get_total_batches_in_epoch()
                
                # Check if training is actually complete
                training_complete = False
                
                # Case 1: All epochs finished (start_epoch is 0-indexed)
                if self.start_epoch >= num_epochs:
                    training_complete = True
                
                # Case 2: Last epoch had all batches processed
                elif self.start_epoch == num_epochs - 1 and self.start_batch >= total_batches_in_epoch:
                    training_complete = True
                
                if training_complete:
                    print("\n" + "="*60)
                    print("✓ TRAINING ALREADY COMPLETE")
                    print("="*60)
                    print(f"All {num_epochs} epochs finished in previous run.")
                    print(f"Last checkpoint: minibatch {self.total_minibatches_processed}")
                    print(f"Epoch {self.start_epoch+1}/{num_epochs}")
                    print(f"Loss: {self.current_loss:.4f}")
                    print("="*60)
                    print("Nothing to train.")
                    print("Use --clean for fresh start, or increase --epochs.")
                    print("="*60)
                    return
                
                if is_emergency:
                    print(f"\n⚠ Resuming from emergency save.")
                    # Emergency save happens AFTER batch completion (SIGTERM, Ctrl+C)
                    # So the saved batch was already fully processed
                    self.start_batch += 1
                    print(f"  Batch {self.start_batch-1} was completed before save.")
                    print(f"  Starting from batch {self.start_batch}.")
            else:
                print("\nStarting fresh training.")
        
        self.is_running = True
        self.ctrl_c_count = 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs} (starting from {self.start_epoch+1})")
        print(f"Total minibatches processed so far: {self.total_minibatches_processed}")
        print(f"Save checkpoint every: {self.state_manager.save_every_n_minibatches} minibatches")
        print(f"Checkpoint directory: {self.state_manager.save_dir}/")
        print(f"{'='*60}")
        print(f"Interruption handling:")
        print(f"  Ctrl+C once  → Graceful stop (save & exit)")
        print(f"  Ctrl+C twice → Force quit (no save)")
        print(f"  Cloud preempt → Emergency save & exit")
        print(f"  Power failure → Emergency save on exit")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(self.start_epoch, num_epochs):
                if not self.is_running:
                    break
                    
                self.current_epoch = epoch
                avg_loss = self.train_epoch(epoch)
                
                if not self.is_running:
                    break
                    
                print(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}')
                
        except KeyboardInterrupt:
            self._emergency_save()
            
        except Exception as e:
            print(f"\n⚠ Unexpected error: {e}")
            self._emergency_save()
            raise
        
        finally:
            self.is_running = False
            
            if self.stop_requested:
                print(f"\n⏸ Training stopped at minibatch {self.total_minibatches_processed}")
                print(f"   Epoch {self.current_epoch+1}, batch {self.current_batch}")
                print(f"   Resume with: python train.py")
            elif self.current_epoch + 1 >= num_epochs:
                print(f"\n✓ Training complete!")
                print(f"   Total minibatches: {self.total_minibatches_processed}")
                print(f"   All {num_epochs} epochs finished.")
            else:
                print(f"\n⚠ Training ended unexpectedly at minibatch {self.total_minibatches_processed}")
                print(f"   Epoch {self.current_epoch+1}/{num_epochs}")
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        
        # Resume from checkpoint if needed
        if epoch == self.start_epoch and self.start_batch > 0:
            resume_indices = self.dataloader.current_epoch_indices
            resume_batch_start = self.start_batch
        else:
            resume_indices = None
            resume_batch_start = 0
            self.start_batch = 0
        
        self.dataloader.start_epoch(
            epoch, 
            resume_indices=resume_indices,
            resume_batch_start=resume_batch_start
        )
        
        total_batches = self.dataloader.get_total_batches_in_epoch()
        total_samples = len(self.dataloader.current_epoch_indices)
        
        pbar = tqdm(total=total_batches, desc=f'Epoch {epoch+1}',
                    initial=resume_batch_start)
        
        batch_idx = resume_batch_start
        total_loss = 0.0
        batches_in_epoch = 0
        
        while True:
            # Check for stop request (Ctrl+C once)
            if self.stop_requested:
                self._emergency_save()
                pbar.close()
                self.is_running = False
                return total_loss / batches_in_epoch if batches_in_epoch > 0 else 0
            
            # Get batch
            batch, wait_time, load_time = self.dataloader.get_next_batch()
            
            if batch is None:
                break
            
            # Compute
            compute_start = time.time()
            
            X, Y = batch
            X, Y = X.to(self.device), Y.to(self.device)
            
            self.optimizer.zero_grad()
            P = self.model(X)
            L = self.loss_fn(P, X, Y)
            L.backward()
            self.optimizer.step()
            
            compute_time = time.time() - compute_start
            
            # Update counters
            batch_idx += 1
            batches_in_epoch += 1
            self.total_minibatches_processed += 1
            total_loss += L.item()
            avg_loss = total_loss / batches_in_epoch
            self.current_batch = batch_idx
            self.current_loss = avg_loss
            
            # Progress bar - cap sample count at total
            samples_used = min(batch_idx * self.dataloader.batch_size, total_samples)
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'wait': f'{wait_time*1000:.0f}ms',
                'comp': f'{compute_time*1000:.0f}ms',
                'data': f'{samples_used}/{total_samples}',
                'total': self.total_minibatches_processed
            })
            
            # Periodic checkpoint
            if (self.total_minibatches_processed % 
                self.state_manager.save_every_n_minibatches == 0):
                
                dataloader_state = self.dataloader.get_state()
                
                self.state_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    dataloader_state=dataloader_state,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_minibatches_processed=self.total_minibatches_processed,
                    loss=avg_loss,
                    is_emergency=False
                )
        
        # Ensure progress bar shows completed
        pbar.update(total_batches - pbar.n)
        pbar.close()
        
        return total_loss / batches_in_epoch if batches_in_epoch > 0 else 0


# ═══════════════════════════════════════════════════════════
# DATASETS AND MODELS
# ═══════════════════════════════════════════════════════════

class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=1000, input_dim=10, output_dim=5):
        self.size = size
        self.X = torch.randn(size, input_dim)
        self.Y = torch.randn(size, output_dim)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SimpleModel(nn.Module):
    """Simple neural network model."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def custom_loss(P, X, Y):
    """Custom loss function."""
    return nn.functional.mse_loss(P, Y)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resilient Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=10, help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', help='Delete old checkpoints')
    parser.add_argument('--dataset-size', type=int, default=500, help='Dataset size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    
    CHECKPOINT_DIR = 'training_checkpoints'
    
    # Clean start if requested
    if args.clean and Path(CHECKPOINT_DIR).exists():
        import shutil
        shutil.rmtree(CHECKPOINT_DIR)
        print("Cleaned previous checkpoints.\n")
    
    # Create dataset, model, dataloader, optimizer
    dataset = SimpleDataset(size=args.dataset_size, input_dim=10, output_dim=5)
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    dataloader = ThreadedDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create trainer
    trainer = ResumableTrainer(
        model=model,
        loss_fn=custom_loss,
        optimizer=optimizer,
        dataloader=dataloader,
        save_dir=CHECKPOINT_DIR,
        device='cpu',
        save_every_n_minibatches=args.save_every
    )
    
    # Show status
    print("=" * 60)
    print("RESILIENT TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.dataset_size} samples, batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Minibatches per epoch: {math.ceil(args.dataset_size / args.batch_size)}")
    print(f"Save every: {args.save_every} minibatches")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print("=" * 60)
    
    # Check for existing checkpoint
    state_file = Path(CHECKPOINT_DIR) / 'training_state.json'
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            print(f"Found previous checkpoint at minibatch {state['total_minibatches_processed']}")
            print(f"Epoch {state['epoch']+1}, batch {state['batch_idx']}")
        except:
            pass
    else:
        print("No previous checkpoint - fresh start")
    
    print()
    
    # Train
    trainer.train(num_epochs=args.epochs, resume=True)