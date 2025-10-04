import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from OffChunkEncLayer import OffChunkEncLayer

class OffChunkTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        num_chunks: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.relu,
        device: Optional[str] = None,
        norm_layer: Optional[nn.Module] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        layers = []
        for _ in range(num_layers):
            layers.append(
                OffChunkEncLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_chunks=num_chunks,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    device=device,
                    eps=eps,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
        
        self.src_saved = None

    def forward(
        self,
        src: torch.Tensor,                       # (B, S, E) typically on CPU
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        train: bool = False,
        modules_to_cpu: bool = True
    ) -> torch.Tensor:
        output = src.to(self.device)
        for layer in self.layers:
            output = layer.forward(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, train=train, modules_to_cpu=modules_to_cpu)
        if self.norm is not None:
            norm = self.norm.to(self.device)
            output = self.norm(output.to(self.device))
            norm.to("cpu")
        # output is on GPU (self.device)
        return output
    
    def backward(self, gradient: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
        
    
if __name__ == "__main__":
    """
    1. For large seq_len, the more layers the more loss.
    """
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    factor10 = 2
    batch_size = 24         # it's also the data simplicity in this test.
    seq_len = round(1 * 72 * factor10 * 144 * factor10 /16/16)
    embed_dim = 1024
    num_heads = 8
    num_chunks = 4
    lr = 1e-3
    steps = 500

    model = OffChunkTransformer(
        num_layers = 3,
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        num_chunks=num_chunks, 
        dim_feedforward=4*embed_dim, 
        dropout=0.0, 
        activation=F.relu, 
        device=device
    )

    x = torch.randn(batch_size, seq_len, embed_dim)  # (B, S, E)
    out = model(x)
    print("Output:", out.shape, out.device)
    
    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, seq_len, embed_dim)
    y_train = torch.randn(batch_size, seq_len, embed_dim)

    for step in range(steps):
        optimizer.zero_grad()

        # forward pass (offloads chunks to GPU internally)
        with torch.no_grad():
            out = model.forward(x_train, train=True)
            
        # target -> same device as out for loss
        out.requires_grad_(True)
        out.retain_grad()
        with torch.enable_grad():   # redundant for readability
            loss = criterion(out, y_train.to(out.device))
            loss.backward()
        gradient = out.grad

        gradient = model.backward(gradient)

        optimizer.step()

        print(f"Step {step+1}/{steps} - Loss: {loss.item():.6f}")

    print("Training done.")