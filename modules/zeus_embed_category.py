import torch
import torch.nn as nn
import math

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.zeus_dataset_Snapshot import Snapshot

# First, define the corrected class
class Embed_Category(nn.Module):
    def __init__(self): 
        super().__init__()
        
        # Get data on CPU first
        cat_cpu = Snapshot.cat
        cat_indices = cat_cpu.to(torch.long)
        C, H, W = cat_indices.shape
        
        # Get vocabulary sizes (CPU computation)
        vocab_sizes = (cat_cpu.amax(dim=(1, 2)) + 1).long().tolist()
        dims_list = [max(1, math.ceil(math.log2(nc))) for nc in vocab_sizes]
        # land_sea_maks: ceil log2(2) = 1, soil_type: ceil log2(7) = 3
        self.total_dims = sum(dims_list)
        
        # Create random embedding tensor (will move to device with model)
        # Note: don't specify device here, let .to(device) handle it
        embedding_tensor = torch.randn(H, W, self.total_dims) * 0.02
        self.embedding = nn.Parameter(embedding_tensor)
        
        # Store references
        self.vocab_sizes = vocab_sizes
        self.dims_list = dims_list
        self.dims_category_emb = self.total_dims
    
    def forward(self):
        return self.embedding


# Run the test
if __name__ == "__main__":
    # Mock the Snapshot module if it doesn't exist
    
    try:
        from modules.zeus_dataset_Snapshot import Snapshot
    except ImportError:
        print("Snapshot module not found, creating mock...")
        
        # Create a mock Snapshot class
        class MockSnapshot:
            cat = torch.stack([
                torch.randint(0, 4, (4, 4)),
                torch.randint(0, 2, (4, 4)),
                torch.randint(0, 8, (4, 4)),
            ], dim=0)
        
        # Create the module structure
        import sys
        from types import ModuleType
        
        # Create mock module
        mock_module = ModuleType('modules.zeus_dataset_Snapshot')
        mock_module.Snapshot = MockSnapshot
        sys.modules['modules.zeus_dataset_Snapshot'] = mock_module
        
        from modules.zeus_dataset_Snapshot import Snapshot

    # Now the corrected test function
    def test_embed_category_fixed_corrected():
        print("=" * 60)
        print("Testing Embed_Category Class (CORRECTED)")
        print("=" * 60)
        
        # Create instance
        model = Embed_Category()
        
        # Check learnable parameters
        params = list(model.parameters())
        print(f"1. Number of parameter groups: {len(params)}")
        print(f"   Total learnable parameters: {sum(p.numel() for p in params)}")
        
        # Test forward pass
        output = model()
        print(f"\n2. Forward pass:")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output requires_grad: {output.requires_grad}")
        
        # Test different instances
        model2 = Embed_Category()
        output2 = model2()
        same = torch.allclose(output, output2, rtol=1e-5)
        print(f"\n3. Different instances:")
        print(f"   Two instances produce same output: {same}")
        
        # Test gradients flow
        loss = output.sum()
        loss.backward()
        has_gradients = model.embedding.grad is not None
        print(f"\n4. Gradient flow:")
        print(f"   Gradients computed: {has_gradients}")
        if has_gradients:
            print(f"   Gradient norm: {model.embedding.grad.norm().item():.6f}")
        
        # Test training updates - CORRECTED VERSION
        print(f"\n5. Training updates (CORRECTED TEST):")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Store initial value (MAKE A COPY!)
        initial_value = model.embedding.data.clone()
        
        # Training step
        optimizer.zero_grad()
        current_output = model()  # Use for loss, but don't store as reference
        loss = current_output.sum()
        loss.backward()
        optimizer.step()
        
        # Store new value (MAKE A COPY!)
        new_value = model.embedding.data.clone()
        
        # Check if values changed
        embedding_changed = not torch.allclose(initial_value, new_value, rtol=1e-5)
        print(f"   Embedding changed: {embedding_changed}")
        
        if embedding_changed:
            max_change = (new_value - initial_value).abs().max().item()
            print(f"   Max change: {max_change:.6f}")
        
        # Verify dimensions
        print(f"\n6. Dimensions:")
        print(f"   Vocabulary sizes: {model.vocab_sizes}")
        print(f"   Embedding dims: {model.dims_list}")
        print(f"   Total dims: {model.total_dims}")
        print(f"   Output channels: {output.shape[0]}")
        
        print("\n" + "=" * 60)
        if (output.requires_grad and 
            has_gradients and 
            embedding_changed and 
            output.shape[0] == model.total_dims):
            print("✅ All tests PASSED! Embeddings are learnable.")
        else:
            print("❌ Some tests FAILED!")
        
        return model

    
    # Run test
    model = test_embed_category_fixed_corrected()
        
        
    def correct_test():
        model = Embed_Category()
        
        # Get initial value (MAKE A COPY!)
        initial_copy = model.embedding.data.clone()
        
        # Do optimization
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        
        loss = model().sum()
        loss.backward()
        optimizer.step()
        
        # Get new value (MAKE A COPY!)
        new_copy = model.embedding.data.clone()
        
        # Compare COPIES
        changed = not torch.allclose(initial_copy, new_copy, rtol=1e-5)
        
        print(f"Initial copy id: {id(initial_copy)}")
        print(f"New copy id: {id(new_copy)}")
        print(f"model.embedding id: {id(model.embedding)}")
        print(f"Are copies same object? {initial_copy is new_copy}")
        print(f"Changed? {changed}")
        
        if changed:
            print(f"Max difference: {(new_copy - initial_copy).abs().max().item()}")
        
        return changed

    correct_test()