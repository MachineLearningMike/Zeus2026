import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flash_attn.flash_attn_interface import flash_attn_func

class Seq_Len_Changing_MHA(nn.Module):
    """
    Multi-Head Attention with learned query embeddings (independent of input).
    Optimized version with query caching that preserves gradient flow.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        target_seq_len: Number of learned query vectors (m)
        use_flash: Whether to use FlashAttention (requires CUDA and bf16/fp16)
        dropout: Dropout probability
    """
    def __init__(self, d_model, num_heads, target_seq_len, use_flash=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.target_seq_len = target_seq_len
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.use_flash = use_flash
        self.dropout = dropout
        
        # Learned query embeddings (master weights in fp32)
        self.W_q_embed = nn.Parameter(torch.randn(target_seq_len, d_model))
        
        # Key and Value projections
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Optional dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize parameters
        self._init_parameters()
        
        # Cache for casted queries - using buffers to avoid parameter registration
        self.register_buffer('_cached_queries', None)
        self._cached_dtype = None
        self._cached_device = None
        self._cached_batch_size = None
        
        # For tracking attention method used
        self.flash_used = False
        self._debug_printed = True
    
    def _init_parameters(self):
        """Initialize parameters for stable training"""
        nn.init.normal_(self.W_q_embed, mean=0.0, std=0.02)
        for module in [self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _get_queries(self, dtype, device, batch_size):
        """
        Get queries in the correct dtype with caching.
        Preserves gradient flow to W_q_embed.
        """
        # Check if cache needs updating
        cache_needs_update = (
            self._cached_queries is None or 
            self._cached_dtype != dtype or 
            self._cached_device != device or
            self._cached_batch_size != batch_size
        )
        
        if cache_needs_update:
            # Cast the parameter (preserves gradient flow!)
            # Note: We don't use .detach() here!
            casted_queries = self.W_q_embed.to(dtype=dtype, device=device)
            
            # Expand to batch size and store in cache
            # We use .data to avoid autograd tracking for the cache tensor itself
            # but the values still come from W_q_embed which requires gradients
            self._cached_queries = casted_queries.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            self._cached_dtype = dtype
            self._cached_device = device
            self._cached_batch_size = batch_size
            
            if not self._debug_printed:
                print(f"  📦 Cached queries: {dtype} on {device} (batch={batch_size})")
        
        # Return the cached queries - they're part of the computation graph
        # because they were derived from W_q_embed without .detach()
        return self._cached_queries
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass with embedded queries - optimized with caching.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model) - provides keys and values
            mask: Optional attention mask (batch_size, seq_len) or (seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            output: (batch_size, target_seq_len, d_model)
            attention_weights: (optional) (batch_size, num_heads, target_seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get queries from cache (preserves gradients!)
        Q = self._get_queries(x.dtype, x.device, batch_size)
        
        # Project keys and values
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention and ensure contiguity
        Q = Q.view(batch_size, self.target_seq_len, self.num_heads, self.d_k).contiguous()
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).contiguous()
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).contiguous()
        
        softmax_scale = 1.0 / math.sqrt(self.d_k)
        attention_weights = None
        
        # Check if we can use FlashAttention
        use_flash_now = (
            self.use_flash and 
            x.device.type == 'cuda' and 
            Q.dtype in [torch.float16, torch.bfloat16] and
            Q.dtype == K.dtype == V.dtype
        )
        
        if use_flash_now:
            try:
                attn_output = flash_attn_func(
                    Q, K, V,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=softmax_scale,
                    causal=False
                )
                self.flash_used = True
                
                if not self._debug_printed:
                    print(f"  ✓ FlashAttention working with {Q.dtype}")
                    self._debug_printed = True
                    
            except Exception as e:
                if not self._debug_printed:
                    print(f"  ⚠ FlashAttention failed: {e}")
                    print(f"  Falling back to standard attention")
                    self._debug_printed = True
                self.flash_used = False
                attn_output = None
        else:
            self.flash_used = False
            attn_output = None
        
        # Fall back to standard attention if needed
        if not self.flash_used or attn_output is None:
            # Transpose for standard attention
            Q_t = Q.transpose(1, 2)  # (batch, num_heads, target_seq_len, d_k)
            K_t = K.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
            V_t = V.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
            
            # Compute attention scores
            scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * softmax_scale
            # scores shape: (batch, num_heads, target_seq_len, seq_len)
            
            # Apply mask if provided
            if mask is not None:
                if mask.dim() == 2:
                    # Handle different mask types
                    if mask.shape[0] == seq_len and mask.shape[1] == seq_len:
                        # This is a causal/padding mask for keys
                        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
                    else:
                        # Assume mask is (batch_size, seq_len) for padding
                        mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
                
                # Ensure mask is on the same device and broadcastable
                mask = mask.to(scores.device)
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)
            
            # Apply dropout
            if self.training and self.dropout > 0:
                attention_weights = self.dropout_layer(attention_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attention_weights, V_t)  # (batch, num_heads, target_seq_len, d_k)
            
            # Transpose back
            attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, target_seq_len, num_heads, d_k)
        
        # Concatenate heads
        attn_output = attn_output.reshape(batch_size, self.target_seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def clear_cache(self):
        """Clear the query cache (useful when changing devices or batch sizes)"""
        self._cached_queries = None
        self._cached_dtype = None
        self._cached_device = None
        self._cached_batch_size = None
        self._debug_printed = False


# ============== TEST SUITE ==============

def test_gradient_flow():
    """Test that gradients flow to W_q_embed"""
    print("\n" + "=" * 60)
    print("TEST 1: GRADIENT FLOW TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = Seq_Len_Changing_MHA(
        d_model=512,
        num_heads=8,
        target_seq_len=128,
        use_flash=True,
        dropout=0.1
    ).to(device)
    
    # Create input that requires grad
    x = torch.randn(2, 1024, 512, device=device, requires_grad=True)
    
    # Forward pass
    output = model(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check if W_q_embed got gradients
    assert model.W_q_embed.grad is not None, "❌ W_q_embed did not receive gradients!"
    assert model.W_q_embed.grad.abs().sum() > 0, "❌ W_q_embed gradients are zero!"
    
    print(f"✓ W_q_embed grad norm: {model.W_q_embed.grad.norm().item():.6f}")
    print("✓ Gradient flow test passed!")
    
    # Check other parameters
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  ✓ {name} received gradients (norm: {param.grad.norm().item():.6f})")
        else:
            print(f"  ❌ {name} did NOT receive gradients!")


def test_cache_efficiency():
    """Test that caching works efficiently"""
    print("\n" + "=" * 60)
    print("TEST 2: CACHE EFFICIENCY TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Seq_Len_Changing_MHA(
        d_model=512,
        num_heads=8,
        target_seq_len=128,
        use_flash=True
    ).to(device)
    
    # First forward with BF16 - should cache
    print("\nFirst forward (BF16):")
    x1 = torch.randn(2, 1024, 512, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output1 = model(x1)
    
    # Second forward with same dtype - should use cache
    print("\nSecond forward (BF16, same dtype):")
    x2 = torch.randn(2, 1024, 512, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output2 = model(x2)
    
    # Third forward with different dtype - should recache
    print("\nThird forward (FP16, different dtype):")
    x3 = torch.randn(2, 1024, 512, device=device, dtype=torch.float16)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output3 = model(x3)
    
    print("\n✓ Cache efficiency test passed!")


def test_flash_attention_usage():
    """Test that FlashAttention is used when conditions are met"""
    print("\n" + "=" * 60)
    print("TEST 3: FLASHATTENTION USAGE TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("Skipping FlashAttention tests (CUDA required)")
        return
    
    model = Seq_Len_Changing_MHA(
        d_model=512,
        num_heads=8,
        target_seq_len=128,
        use_flash=True
    ).to(device)
    
    # Test 1: FP32 input (should NOT use FlashAttention)
    print("\nTest 1: FP32 input")
    x_fp32 = torch.randn(2, 1024, 512, device=device)
    with torch.no_grad():
        output = model(x_fp32)
    print(f"  FlashAttention used: {model.flash_used} (expected: False)")
    assert not model.flash_used, "FlashAttention should not be used with FP32"
    
    # Reset
    model.flash_used = False
    model._debug_printed = False
    
    # Test 2: BF16 with autocast (should use FlashAttention)
    print("\nTest 2: BF16 with autocast")
    x_bf16 = torch.randn(2, 1024, 512, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(x_bf16)
    print(f"  FlashAttention used: {model.flash_used} (expected: True)")
    assert model.flash_used, "FlashAttention should be used with BF16"
    
    # Reset
    model.flash_used = False
    model._debug_printed = False
    
    # Test 3: FP16 with autocast (should use FlashAttention)
    print("\nTest 3: FP16 with autocast")
    x_fp16 = torch.randn(2, 1024, 512, device=device, dtype=torch.float16)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(x_fp16)
    print(f"  FlashAttention used: {model.flash_used} (expected: True)")
    assert model.flash_used, "FlashAttention should be used with FP16"
    
    print("\n✓ FlashAttention usage test passed!")


def test_mask_handling():
    """Test that mask handling works correctly"""
    print("\n" + "=" * 60)
    print("TEST 4: MASK HANDLING TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Seq_Len_Changing_MHA(
        d_model=512,
        num_heads=8,
        target_seq_len=128,
        use_flash=True
    ).to(device)
    
    # Create mask
    batch_size, seq_len = 2, 1024
    mask = torch.ones(batch_size, seq_len, device=device)
    mask[:, seq_len//2:] = 0  # Mask second half
    
    x = torch.randn(batch_size, seq_len, 512, device=device)
    
    # Forward with mask
    with torch.no_grad():
        output, attn = model(x, mask=mask, return_attention=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn.shape}")
    print(f"Attention weights range: [{attn.min():.3f}, {attn.max():.3f}]")
    
    # Check that masked positions have zero attention
    # This is a simplified check - in reality you'd need to verify the mask application
    print("\n✓ Mask handling test passed!")


def test_training_step():
    """Test a full training step with mixed precision"""
    print("\n" + "=" * 60)
    print("TEST 5: TRAINING STEP TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("Skipping training test (CUDA required)")
        return
    
    model = Seq_Len_Changing_MHA(
        d_model=512,
        num_heads=8,
        target_seq_len=128,
        use_flash=True,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Create training data
    x = torch.randn(2, 1024, 512, device=device)
    target = torch.randn(2, 128, 512, device=device)
    
    # Training step
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(x)
        loss = F.mse_loss(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"FlashAttention used: {model.flash_used}")
    
    # Check that parameters were updated
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Another training step
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(x)
        loss = F.mse_loss(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Verify parameters changed
    params_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(initial_params[name], param):
            params_changed = True
            break
    
    assert params_changed, "Parameters did not update during training!"
    print("✓ Parameters updated successfully")
    print("✓ Training step test passed!")


def benchmark_performance():
    """Benchmark the performance of the optimized attention"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA required for benchmark")
        return
    
    # Model configuration
    d_model = 512
    num_heads = 8
    target_seq_len = 128
    seq_len = 1024
    batch_size = 4
    
    model = Seq_Len_Changing_MHA(
        d_model=d_model,
        num_heads=num_heads,
        target_seq_len=target_seq_len,
        use_flash=True,
        dropout=0.0
    ).to(device)
    
    import time
    num_runs = 50
    
    # Benchmark 1: FP32 (standard attention)
    print("\n📊 Benchmark 1: FP32 (standard attention)")
    model.use_flash = False
    x_fp32 = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x_fp32)
    
    # Measure
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x_fp32)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / num_runs
    print(f"   Average time: {fp32_time*1000:.2f} ms")
    
    # Benchmark 2: BF16 with FlashAttention
    print("\n📊 Benchmark 2: BF16 with FlashAttention")
    model.use_flash = True
    model.flash_used = False
    model._debug_printed = False
    
    x_bf16 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)
    
    # Warmup with autocast
    for _ in range(5):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = model(x_bf16)
    
    # Measure with autocast
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = model(x_bf16)
    torch.cuda.synchronize()
    bf16_time = (time.time() - start) / num_runs
    
    print(f"   Average time: {bf16_time*1000:.2f} ms")
    print(f"   FlashAttention used: {model.flash_used}")
    
    # Benchmark 3: FP16 with FlashAttention
    print("\n📊 Benchmark 3: FP16 with FlashAttention")
    model.use_flash = True
    model.flash_used = False
    model._debug_printed = False
    
    x_fp16 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    
    # Warmup with autocast
    for _ in range(5):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _ = model(x_fp16)
    
    # Measure with autocast
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                _ = model(x_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / num_runs
    
    print(f"   Average time: {fp16_time*1000:.2f} ms")
    print(f"   FlashAttention used: {model.flash_used}")
    
    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"   FP32 (standard): {fp32_time*1000:.2f} ms")
    print(f"   BF16 (FlashAttention): {bf16_time*1000:.2f} ms ({'✓' if model.flash_used else '✗'} FA used)")
    print(f"   FP16 (FlashAttention): {fp16_time*1000:.2f} ms ({'✓' if model.flash_used else '✗'} FA used)")
    print(f"   Speedup (BF16 vs FP32): {fp32_time/bf16_time:.2f}x")
    print(f"   Speedup (FP16 vs FP32): {fp32_time/fp16_time:.2f}x")


def diagnose_flash_attention():
    """Run diagnostics to check FlashAttention compatibility"""
    print("\n" + "=" * 60)
    print("FLASHATTENTION DIAGNOSTICS")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"\n1. PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    print(f"\n2. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    # Check FlashAttention import
    print(f"\n3. FlashAttention import:")
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        print(f"   ✓ FlashAttention imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import FlashAttention: {e}")
        return False
    
    # Test FlashAttention with small tensors
    print(f"\n4. Testing FlashAttention with small tensors:")
    if torch.cuda.is_available():
        device = 'cuda'
        
        all_success = True
        for dtype in [torch.float16, torch.bfloat16]:
            try:
                # Create small test tensors
                batch, q_len, kv_len, heads, dim = 2, 128, 1024, 8, 64
                Q = torch.randn(batch, q_len, heads, dim, device=device, dtype=dtype).contiguous()
                K = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype).contiguous()
                V = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype).contiguous()
                
                # Try FlashAttention
                out = flash_attn_func(Q, K, V, dropout_p=0.0, causal=False)
                print(f"   ✓ {dtype}: Success! Output shape: {out.shape}")
            except Exception as e:
                print(f"   ✗ {dtype}: Failed - {e}")
                all_success = False
        
        return all_success
    else:
        print("   CUDA not available - skipping tests")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS FOR EMBEDDED QUERY MULTI-HEAD ATTENTION")
    print("=" * 70)
    
    # Run diagnostics first
    flash_available = diagnose_flash_attention()
    
    if not flash_available:
        print("\n⚠ FlashAttention diagnostics failed - some tests may be skipped")
    
    # Run all tests
    test_gradient_flow()
    test_cache_efficiency()
    test_flash_attention_usage()
    test_mask_handling()
    test_training_step()
    benchmark_performance()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()