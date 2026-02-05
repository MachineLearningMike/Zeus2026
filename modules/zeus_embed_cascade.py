import torch
import torch.nn as nn

class Embed_Cascade(nn.Module):
    def __init__(self, H, W, max_cascade_depth, dims_cascade_emb=2): 
        super().__init__()
        self.H = H
        self.W = W
        self.max_cascade_depth = max_cascade_depth
        self.dims_cascade_emb = dims_cascade_emb
        
        # Only these weights will have gradients
        self.cascade_embeds = nn.Embedding(max_cascade_depth, dims_cascade_emb)

    def forward(self, cascade_depth):
        # Input should already be integer(s) without gradients
        # But let's make sure by converting to LongTensor without grad
        if isinstance(cascade_depth, int):
            indices = torch.tensor(cascade_depth - 1, 
                                  device=self.cascade_embeds.weight.device,
                                  dtype=torch.long)
        else:
            # Assuming cascade_depth is already a tensor
            indices = cascade_depth.long() - 1
        
        # Ensure indices are valid
        indices = torch.clamp(indices, 0, self.max_cascade_depth - 1)
        
        # Forward pass through embedding layer
        # Gradients will only flow through cascade_embeds.weight
        return self.cascade_embeds(indices)[None, None, :].repeat(self.H, self.W, 1)
    
if __name__ == "__main__":

    # During training
    model = Embed_Cascade(max_cascade_depth=10, dims_cascade_emb=4)

    # cascade_depth comes from data loader (integers, no grad)
    depth_batch = torch.tensor([1, 2], dtype=torch.long)  # No requires_grad=True!

    embeddings = model(depth_batch)  # shape: (4, 4)

    print(f"embeddings: {embeddings}")

    # Compute loss
    loss = embeddings.sum()  # Gradients flow back to embedding weights, NOT depth_batch
    loss.backward()

    # Check gradients
    print(model.cascade_embeds.weight.grad is not None)  # True
    print(depth_batch.grad is None)  # True (no gradients for indices)
    