import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- Create 4 simple modules ----------------
class SimpleModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

# Four submodules
m1 = SimpleModule(10, 20)
m2 = SimpleModule(20, 30)
m3 = SimpleModule(30, 40)
m4 = SimpleModule(40, 1)
modules = [m1, m2, m3, m4]

# ---------------- Freeze first N modules before training ----------------
freeze_until = 2  # freeze first 2 modules
for i, module in enumerate(modules):
    freeze = i < freeze_until
    for param in module.parameters():
        param.requires_grad = not freeze

# ---------------- Dummy dataset ----------------
X = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ---------------- Optimizer ----------------
optimizer = torch.optim.Adam(
    [p for module in modules for p in module.parameters() if p.requires_grad],
    lr=1e-3
)
loss_fn = nn.MSELoss()

# ---------------- Training Parameters ----------------
acts_cpu = []

# ---------------- Training Loop ----------------
for data, target in dataloader:
    data, target = data.to("cpu"), target.to("cpu")
    acts_cpu.clear()
    x = data.to("cuda:0")  # input on GPU

    # ---------------- Forward Pass ----------------
    for i, module in enumerate(modules):
        module.to("cuda:0")

        # Only last module tracks grad
        with torch.set_grad_enabled(i == len(modules)-1):
            x = module(x)

        # Save activations of non-last modules to CPU
        if i != len(modules)-1:
            acts_cpu.append(x.cpu())
            module.to("cpu")       # move module back to CPU
            x = x.detach()         # keep tensor but remove autograd
            torch.cuda.empty_cache()

    final_out = x  # last module output exists

    # ---------------- Backward Pass (recompute m3..m1) ----------------
    grad = None
    for i in reversed(range(len(modules)-1)):
        if not any(p.requires_grad for p in modules[i].parameters()):
            continue  # skip frozen modules

        module = modules[i]
        module.to("cuda:0")

        prev_act = acts_cpu[i-1].to("cuda:0") if i > 0 else data.to("cuda:0")
        prev_act.requires_grad_(True)

        with torch.set_grad_enabled(True):
            out = module(prev_act)
            if grad is None:
                grad = torch.ones_like(out)
            out.backward(grad.to(out.device))

        grad = prev_act.grad.detach().cpu()
        del out, prev_act
        torch.cuda.empty_cache()
        module.to("cpu")

    # ---------------- Last Module Backward ----------------
    loss = loss_fn(final_out, target.to("cuda:0"))
    loss.backward()

    # ---------------- Optimizer Step ----------------
    optimizer.step()
    optimizer.zero_grad()

print("Test run complete!")
