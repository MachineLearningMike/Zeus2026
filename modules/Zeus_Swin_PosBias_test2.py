import torch

def verify_original_implementation():
    # Small test case
    wd, wh, ww = 2, 2, 3
    D, H, W = 4, 4, 9  # Input resolution
    num_heads = 4
    
    # Calculate dimensions
    G = (D // wd) * (H // wh)  # 2*2 = 4
    N = wd * wh * ww  # 2*2*3 = 12
    num_lon_windows = W // ww  # 9//3 = 3
    
    print(f"G={G}, N={N}, num_lon_windows={num_lon_windows}")
    print(f"Parameter shape: ({G}, {wd*wd}, {wh*wh}, {2*ww-1}, {num_heads})")
    
    # Create synthetic parameter
    param = torch.randn(G, wd*wd, wh*wh, 2*ww-1, num_heads)
    
    # Generate coordinates
    d = torch.arange(wd)
    h = torch.arange(wh)
    w = torch.arange(ww)
    coords = torch.stack(torch.meshgrid(d, h, w, indexing="ij")).view(3, -1)
    
    # Calculate indices (original method)
    d1, d2 = coords[0][:, None], coords[0][None, :]
    h1, h2 = coords[1][:, None], coords[1][None, :]
    w1, w2 = coords[2][:, None], coords[2][None, :]
    
    idx_d = d1 * wd + d2
    idx_h = h1 * wh + h2
    idx_w = (w1 - w2 + (ww - 1)) % (2*ww-1)
    
    # Create indexing tensors
    G_idx = torch.arange(G, device=param.device)[:, None, None, None]
    W_idx = torch.arange(num_lon_windows, device=param.device)[None, :, None, None]
    D_idx = idx_d[None, None, :, :]
    H_idx = idx_h[None, None, :, :]
    W_idx_tokens = idx_w[None, None, :, :]
    
    # Expand parameter
    param_expanded = param[:, None, :, :, :, :].repeat(1, num_lon_windows, 1, 1, 1, 1)
    
    # Original indexing
    result = param_expanded[G_idx, W_idx, D_idx, H_idx, W_idx_tokens]
    
    print(f"\nResult shape: {result.shape}")
    print(f"Expected shape: ({G}, {num_lon_windows}, {N}, {N}, {num_heads})")
    
    # Verify a few values manually
    print("\nManual verification (first few elements):")
    for g in range(min(2, G)):
        for lon in range(min(2, num_lon_windows)):
            for i in range(min(2, N)):
                for j in range(min(2, N)):
                    # Get from result
                    val_result = result[g, lon, i, j]
                    
                    # Calculate manually
                    d1_val, h1_val, w1_val = coords[:, i]
                    d2_val, h2_val, w2_val = coords[:, j]
                    
                    idx_d_manual = d1_val * wd + d2_val
                    idx_h_manual = h1_val * wh + h2_val
                    idx_w_manual = (w1_val - w2_val + (ww - 1)) % (2*ww-1)
                    
                    val_manual = param[g, idx_d_manual, idx_h_manual, idx_w_manual]
                    
                    if torch.allclose(val_result, val_manual, atol=1e-6):
                        print(f"  ✓ ({g},{lon},{i},{j}): values match")
                    else:
                        print(f"  ✗ ({g},{lon},{i},{j}): MISMATCH!")
                        print(f"    Result: {val_result}")
                        print(f"    Manual: {val_manual}")
                        return False
    
    print("\n✅ All tests passed! Original implementation is correct.")
    return True

if __name__ == "__main__":
    verify_original_implementation()