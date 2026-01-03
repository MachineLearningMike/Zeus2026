import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

class AdaptiveLossStatLR(_LRScheduler):
    """
    PyTorch Learning Rate Scheduler that adapts based on moving statistics of losses.
    
    Adjusts LR based on:
    1. Distance of moving average from minimum loss
    2. Loss variability (standard deviation)
    3. Patience (steps without improvement)
    4. Recent improvements
    """
    
    def __init__(self, optimizer, 
                 initial_lr=None,
                 min_lr=1e-6,
                 max_lr=0.1,
                 window_size=50,
                 patience=10,
                 improvement_threshold=0.001,
                 std_scaling_factor=0.5,
                 mean_scaling_factor=0.1,
                 warmup_steps=100,
                 last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            initial_lr: Starting learning rate (uses optimizer's lr if None)
            min_lr: Minimum allowed learning rate
            max_lr: Maximum allowed learning rate
            window_size: Number of recent losses to track for statistics
            patience: Steps without improvement before reducing LR aggressively
            improvement_threshold: Relative improvement threshold for counting progress
            std_scaling_factor: How much standard deviation affects LR adjustment
            mean_scaling_factor: How much mean deviation from min affects LR adjustment
            warmup_steps: Steps before starting adaptive adjustments
            last_epoch: The index of last epoch
        """
        self.optimizer = optimizer
        
        # If initial_lr is not specified, use the optimizer's current lr
        if initial_lr is None:
            self.initial_lr = optimizer.param_groups[0]['lr']
        else:
            self.initial_lr = initial_lr
            # Set optimizer's initial lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.window_size = window_size
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.std_factor = std_scaling_factor
        self.mean_factor = mean_scaling_factor
        self.warmup_steps = warmup_steps
        
        # State tracking
        self.min_loss = float('inf')
        self.loss_window = []
        self.steps_since_improvement = 0
        self.step_count = 0
        
        # EMA statistics
        self.ema_mean = None
        self.ema_std = None
        self.ema_beta = 0.9
        
        # Call parent constructor
        super().__init__(optimizer, last_epoch)
    
    def step_loss(self, current_loss):
        """
        Update learning rate based on current loss.
        Call this after computing loss in your training loop.
        
        Args:
            current_loss: Current loss value (float or tensor)
        """
        self.step_count += 1
        
        # Convert to float if it's a tensor
        if torch.is_tensor(current_loss):
            current_loss = current_loss.item()
        
        # Update minimum loss
        improved = False
        if current_loss < self.min_loss * (1 - self.improvement_threshold):
            improved = True
            self.min_loss = current_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        # Track losses for statistics
        self.loss_window.append(current_loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
        
        # During warmup, use simple linear warmup
        if self.step_count < self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
            new_lr = self.initial_lr * warmup_factor
            self._update_lr(new_lr)
            return new_lr
        
        # Skip if we don't have enough data
        if len(self.loss_window) < self.window_size // 2:
            return self.get_last_lr()[0]
        
        # Calculate moving statistics
        recent_losses = np.array(self.loss_window)
        moving_mean = np.mean(recent_losses)
        moving_std = np.std(recent_losses) + 1e-8  # Avoid division by zero
        
        # Update EMA statistics
        if self.ema_mean is None:
            self.ema_mean = moving_mean
            self.ema_std = moving_std
        else:
            self.ema_mean = self.ema_beta * self.ema_mean + (1 - self.ema_beta) * moving_mean
            self.ema_std = self.ema_beta * self.ema_std + (1 - self.ema_beta) * moving_std
        
        # Calculate adjustment factors
        mean_from_min = self.ema_mean - self.min_loss
        normalized_std = self.ema_std / (self.min_loss + 1e-8)
        
        # 1. Mean distance factor (closer to min → potentially lower LR)
        if mean_from_min <= 0:
            mean_factor = 1.1  # Slight increase if below min (rare)
        else:
            mean_factor = np.exp(-self.mean_factor * mean_from_min / self.min_loss)
        
        # 2. Variability factor (high variance → lower LR for stability)
        std_factor = np.exp(-self.std_factor * normalized_std)
        
        # 3. Patience factor (no improvement → reduce LR)
        patience_factor = 1.0
        if self.steps_since_improvement > self.patience:
            patience_exceedance = self.steps_since_improvement - self.patience
            patience_factor = np.exp(-0.1 * patience_exceedance)
        
        # 4. Improvement boost (recent improvement → slight LR boost)
        improvement_factor = 1.05 if improved else 1.0
        
        # Combine all factors
        adjustment = mean_factor * std_factor * patience_factor * improvement_factor
        
        # Get current LR and apply adjustment
        current_lr = self.get_last_lr()[0]
        new_lr = current_lr * adjustment
        
        # Clamp to bounds
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        # Optional: Add small noise if stuck for a long time
        if self.steps_since_improvement > self.patience * 2:
            if np.random.random() < 0.1:
                noise = np.random.uniform(0.9, 1.1)
                new_lr = new_lr * noise
                new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        # Update learning rate
        self._update_lr(new_lr)
        
        return new_lr
    
    def _update_lr(self, new_lr):
        """Update all parameter groups with new learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def get_state(self):
        """Get current state for debugging/monitoring."""
        return {
            'learning_rate': self.get_last_lr()[0],
            'min_loss': self.min_loss,
            'steps_since_improvement': self.steps_since_improvement,
            'step_count': self.step_count,
            'window_size': len(self.loss_window),
            'loss_window_mean': np.mean(self.loss_window) if self.loss_window else 0,
            'loss_window_std': np.std(self.loss_window) if self.loss_window else 0,
            'ema_mean': self.ema_mean,
            'ema_std': self.ema_std
        }
    
    # Override base class methods
    def step(self, epoch=None):
        """This scheduler uses step_loss instead of epoch-based steps."""
        pass
    
    def get_last_lr(self):
        """Get current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


# Alternative: Simpler version for quick integration
class SimpleAdaptiveLR:
    """Simpler PyTorch-compatible adaptive LR scheduler."""
    
    def __init__(self, optimizer, window_size=20, min_lr=1e-5, max_lr=0.01):
        self.optimizer = optimizer
        self.window_size = window_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.min_loss = float('inf')
        self.losses = []
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, loss):
        """Update LR based on current loss."""
        if torch.is_tensor(loss):
            loss = loss.item()
        
        # Track losses
        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
        
        # Update minimum
        self.min_loss = min(self.min_loss, loss)
        
        # Adjust LR if we have enough data
        if len(self.losses) >= self.window_size // 2:
            mean_loss = np.mean(self.losses)
            std_loss = np.std(self.losses)
            
            # Calculate adjustment factors
            mean_ratio = (mean_loss - self.min_loss) / (self.min_loss + 1e-8)
            cv = std_loss / (mean_loss + 1e-8)  # Coefficient of variation
            
            # Combined adjustment
            adjustment = np.exp(-0.1 * mean_ratio - 0.05 * cv)
            
            # Get current LR and apply adjustment
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = current_lr * adjustment
            new_lr = max(self.min_lr, min(self.max_lr, new_lr))
            
            # Update optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        return self.optimizer.param_groups[0]['lr']


# Example training loop with integration
def example_training_loop():
    """
    Example showing how to integrate the adaptive LR scheduler into training.
    """
    import torch.nn as nn
    import torch.optim as optim
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create our adaptive scheduler
    scheduler = AdaptiveLossStatLR(
        optimizer=optimizer,
        initial_lr=2e-5,
        min_lr=1e-8,
        max_lr=1e-2,
        window_size=30,
        warmup_steps=100
    )
    
    # Example data (replace with your actual data)
    dataloader = []  # Your DataLoader here
    
    print("Starting training with adaptive learning rate...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(10):  # Your number of epochs
        epoch_losses = []
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Forward pass
            output = model(data)
            loss = nn.functional.mse_loss(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update learning rate based on current loss
            current_lr = scheduler.step_loss(loss)
            
            # Optimizer step
            optimizer.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            
            # Log progress
            if batch_idx % 50 == 0:
                state = scheduler.get_state()
                print(f"Epoch {epoch:2d} | Batch {batch_idx:3d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Min Loss: {state['min_loss']:.4f}")
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch} complete - Avg Loss: {avg_loss:.4f}")
        print("-" * 40)


# Test function
def test_adaptive_scheduler():
    """Test the scheduler with simulated training."""
    print("Testing AdaptiveLossStatLR with PyTorch...")
    print("=" * 60)
    
    # Create a mock optimizer and scheduler
    mock_params = [torch.randn(10, requires_grad=True)]
    optimizer = optim.Adam(mock_params, lr=0.001)
    
    scheduler = AdaptiveLossStatLR(
        optimizer=optimizer,
        initial_lr=0.001,
        min_lr=1e-5,
        max_lr=0.01,
        window_size=20,
        warmup_steps=10
    )
    
    # Simulate loss patterns
    np.random.seed(42)
    
    # Initial improvement phase
    print("\nPhase 1: Improvement")
    loss = 2.0
    for i in range(30):
        loss = loss * 0.97 + np.random.normal(0, 0.05)
        lr = scheduler.step_loss(loss)
        
        if i % 5 == 0:
            state = scheduler.get_state()
            print(f"Step {i:3d}: Loss={loss:.4f}, LR={lr:.6f}, Min={state['min_loss']:.4f}")
    
    # Plateau phase
    print("\nPhase 2: Plateau")
    for i in range(30, 60):
        loss = 0.5 + np.random.normal(0, 0.08)
        lr = scheduler.step_loss(loss)
        
        if i % 5 == 0:
            state = scheduler.get_state()
            print(f"Step {i:3d}: Loss={loss:.4f}, LR={lr:.6f}, Min={state['min_loss']:.4f}")
    
    # Final improvement
    print("\nPhase 3: Final improvement")
    loss = 0.45
    for i in range(60, 90):
        loss = loss * 0.99 + np.random.normal(0, 0.03)
        lr = scheduler.step_loss(loss)
        
        if i % 5 == 0:
            state = scheduler.get_state()
            print(f"Step {i:3d}: Loss={loss:.4f}, LR={lr:.6f}, Min={state['min_loss']:.4f}")
    
    print("\n" + "=" * 60)
    print("Final scheduler state:")
    state = scheduler.get_state()
    for key, value in state.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.6f}")
        else:
            print(f"{key:25s}: {value}")


if __name__ == "__main__":
    # Run test
    import torch.optim as optim
    test_adaptive_scheduler()
    
    print("\n" + "=" * 60)
    print("Quick Integration Guide:")
    print("=" * 60)
    print("""
# 1. Import and initialize
from adaptive_lr import AdaptiveLossStatLR

# 2. Create optimizer as usual
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Create adaptive scheduler
scheduler = AdaptiveLossStatLR(
    optimizer=optimizer,
    initial_lr=0.001,      # Optional, defaults to optimizer's lr
    min_lr=1e-5,           # Minimum LR
    max_lr=0.01,           # Maximum LR
    window_size=30,        # Losses to track
    warmup_steps=100       # Steps before adaptive adjustments
)

# 4. In training loop:
for batch in dataloader:
    # Forward pass
    loss = compute_loss(batch)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update LR based on current loss (IMPORTANT: before optimizer.step())
    current_lr = scheduler.step_loss(loss)
    
    # Optimizer step
    optimizer.step()
    
    # Monitor (optional)
    if step % 100 == 0:
        state = scheduler.get_state()
        print(f"Step {step}: Loss={loss.item():.4f}, LR={current_lr:.6f}")
    """)