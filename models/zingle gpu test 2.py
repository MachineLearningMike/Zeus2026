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

# ---------------- Training Loop ----------------
nFirstFrozenModules = 2

acts_cpu = []

for data, target in dataloader:
    data, target = data.to("cpu"), target.to("cpu")
    acts_cpu.clear()
    x = data.to("cuda:0")  # input on GPU

    # ---------------- Forward Pass ----------------
    for i, module in enumerate(modules):
        module.to("cuda:0")

        if i < len(modules) - 1:
            with torch.no_grad():
                x = module(x)       # intermediate module, no grad
            x = x.detach()          # safe, disconnect from graph
            acts_cpu.append(x.cpu())  # save for recomputation
            module.to("cpu")
            torch.cuda.empty_cache()
        else:
            # Last module
            final_input = x
            final_module = module
            final_input.requires_grad_(True)   # leaf tensor
            final_out = final_module(final_input)    # compute output with grad

    # ---------------- Backward Pass ----------------
    # 1) Last module and loss
    loss = loss_fn(final_out, target.to("cuda:0"))
    loss.backward()                         # grads for last module & final_input
    final_module.to("cpu")                  # Safe place to move back.
    grad_next = final_input.grad.detach().cpu()


    # 2) Recompute previous modules in reverse
    for i in reversed(range(len(modules)-1)):
        if i < nFirstFrozenModules: continue
        
        # Comment out this block, as modules[i] is not frozen now.
        # if not any(p.requires_grad for p in modules[i].parameters()):
        #     # frozen: recompute output without tracking grads
        #     with torch.no_grad():
        #         out = module(prev_act.to("cuda:0"))
        #     # propagate gradient to input
        #     prev_act.requires_grad_(True)
        #     out.backward(grad_next.to(out.device))
        #     grad_next = prev_act.grad.detach().cpu()
        #     del out, prev_act            
        #     continue

        module = modules[i]
        module.to("cuda:0")

        prev_act = acts_cpu[i-1].to("cuda:0") if i > 0 else data.to("cuda:0")
        prev_act.requires_grad_(True)

        with torch.set_grad_enabled(True):
            out = module(prev_act)
            if grad_next is None:
                grad_next = torch.ones_like(out)
            out.backward(grad_next.to(out.device))

        grad_next = prev_act.grad.detach().cpu()
        del out, prev_act
        torch.cuda.empty_cache()
        module.to("cpu")

    # ---------------- Optimizer Step ----------------
    for i, param_group in enumerate(optimizer.param_groups):
        for j, p in enumerate(param_group['params']):
            print(f"param_group {i}, param {j} device: {p.device}")
    
    optimizer.step()
    optimizer.zero_grad()

print("Memory-efficient single-GPU training complete!")
