import torch

class AdaptiveLRLearner:
    """
    Adaptive LR controller for noisy small-batch training with gradient accumulation.
    - Gradients are accumulated for `accum_steps` before optimizer.step().
    - LR adapts based on smoothed accumulated gradient norm.
    """
    def __init__(self,
                 optimizer,
                 accum_steps=4,
                 init_lr=1e-5,
                 grad_target=0.35,   # 0.1
                 grad_tolerance=2.0,
                 ema_decay=0.9,
                 lr_min=1e-9,
                 lr_max=1e-3,
                 step_down=0.8,
                 step_up=1.05,
                 stagnation_boost=1.02,
                 stagnation_thresh=0.002):
        self.optimizer = optimizer
        self.accum_steps = accum_steps
        self.accum_iterations = 0
        self.ema_grad_norm = 0.0
        self.ema_loss = None
        self.ema_decay = ema_decay
        self.grad_target = grad_target
        self.grad_tolerance = grad_tolerance
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_down = step_down
        self.step_up = step_up
        self.stagnation_boost = stagnation_boost
        self.stagnation_thresh = stagnation_thresh

        for g in self.optimizer.param_groups:
            g['lr'] = init_lr

    def stepOptimizer(self, loss_value):
        """
        Call AFTER loss.backward() on every mini-batch.
        Calls optimizer.step() automatically every `accum_steps`.
        Updates LR based on accumulated gradients.
        Returns: (performed_step:bool, current_lr, grad_norm)
        """
        self.accum_iterations += 1

        # Compute gradient norm of accumulated gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]['params'], max_norm=float('inf')
        ).item()

        # EMA smoothing
        self.ema_grad_norm = self.ema_decay * self.ema_grad_norm + (1 - self.ema_decay) * grad_norm
        if self.ema_loss is None:
            self.ema_loss = loss_value
        else:
            self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * loss_value

        performed_step = False
        if self.accum_iterations >= self.accum_steps:
            # --- Adjust LR based on accumulated gradient ---
            lr = self.optimizer.param_groups[0]['lr']
            if self.ema_grad_norm > self.grad_target * self.grad_tolerance:
                lr *= self.step_down
            elif self.ema_grad_norm < self.grad_target / self.grad_tolerance:
                lr *= self.step_up
            elif abs(loss_value - self.ema_loss) / (self.ema_loss + 1e-8) < self.stagnation_thresh:
                lr *= self.stagnation_boost

            # Clamp LR
            lr = max(min(lr, self.lr_max), self.lr_min)
            for g in self.optimizer.param_groups:
                g['lr'] = lr

            # Perform optimizer step and zero gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accum_iterations = 0
            performed_step = True

        return performed_step, self.optimizer.param_groups[0]['lr'], grad_norm
