import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaOperatorMultiChannel(nn.Module):
    """
    Low-rank Delta operator for multi-channel maps:
    Δ(x) = U diag(g(x)) V, applied across channels and spatial points.
    Input/output shape: [B, H, W, C]
    """
    def __init__(self, H, W, C, latent_rank=16, gate_init_scale=0.01, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.H = H
        self.W = W
        self.C = C
        self.D = H * W * C  # total flattened size
        self.R = latent_rank

        # Low-rank factors
        self.U = nn.Parameter(torch.randn(self.D, latent_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(latent_rank, self.D) * 0.01)

        # Gate network
        hidden = max(latent_rank*2, 32)
        self.g_net = nn.Sequential(
            nn.Linear(self.D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_rank),
            nn.Sigmoid()
        )

        # Scale last layer weights and bias to make g(x) ~ 0 initially
        final_linear = self.g_net[-2]  # nn.Linear(hidden, R)
        final_linear.weight.data *= gate_init_scale
        final_linear.bias.data *= gate_init_scale

        self.to(self.device)

    def forward(self, x_phys):
        """
        x_phys: [B, H, W, C]
        returns: Δ(x) @ x_phys, same shape [B, H, W, C]
        """
        B, H, W, C = x_phys.shape
        assert (H, W, C) == (self.H, self.W, self.C)

        # Flatten spatial and channels
        x_flat = x_phys.reshape(B, -1)  # [B, D]

        # Compute gate g(x)
        g = self.g_net(x_flat)  # (B, R)

        # Low-rank multiplication Δ(x) @ x_phys
        # Step 1: scale V by g per batch: [B, R, D]
        V_g = self.V.unsqueeze(0) * g.unsqueeze(2)

        # Step 2: multiply by U: [B, D, R] @ [B, R, D] -> [B, D]
        U_exp = self.U.unsqueeze(0).expand(B, -1, -1)
        delta_x = torch.bmm(U_exp, V_g)  # [B, D, D] @ flattened x
        # Actually we want delta @ x_phys directly without forming D x D
        # So compute as (U * g) @ (V @ x)
        Vx = torch.matmul(x_flat, self.V.t())   # [B, R]
        gated = Vx * g                           # [B, R]
        delta_x_flat = torch.matmul(gated, self.U.t())  # [B, D]

        # Reshape back to [B, H, W, C]
        delta_x = delta_x_flat.view(B, H, W, C)
        return delta_x


if __name__ == "__main__":
    # Grid and batch parameters
    B, H, W, C = 1, 180, 360, 1  # batch, height, width, channels
    latent_rank = 16
    gate_init_scale = 0.01

    # Random input: e.g., t2m, u100, v100 stacked
    x_phys = torch.randn(B, H, W, C).cuda()

    # Create DeltaOperator
    delta_op = DeltaOperatorMultiChannel(H, W, C, latent_rank, gate_init_scale, device='cuda')

    import time
    s = time.time()
    loops = 5
    for _ in range(loops):
        # Compute Δ(x) @ x_phys
        delta_x = delta_op(x_phys)
    e = time.time()
    print(f"time taken: {(e-s)/loops:.3f}")

    # Apply to x_phys
    x_corrected = x_phys + delta_x

    # Checks
    print("Input shape:", x_phys.shape)
    print("Delta shape:", delta_x.shape)
    print("Corrected shape:", x_corrected.shape)
    print("Delta stats: min {:.3f}, max {:.3f}, mean {:.3f}".format(
        delta_x.min().item(), delta_x.max().item(), delta_x.mean().item()
    ))
    print("Device:", x_corrected.device)
