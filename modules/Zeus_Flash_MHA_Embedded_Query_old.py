import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flash_attn.flash_attn_interface import flash_attn_func

class Seq_Len_Changing_MHA(nn.Module):
    def __init__(self, d_model, num_heads, num_queries, use_flash=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.use_flash = use_flash
        self.dropout = dropout
        
        # Learned query embeddings (master weights in fp32)
        self.W_q_embed = nn.Parameter(torch.randn(num_queries, d_model))
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_parameters()
        
        # Cache for casted queries
        self._cached_dtype = None
        self._cached_queries = None
        self._cached_device = None
        self.flash_used = False
        self._debug_printed = False
    
    def _init_parameters(self):
        nn.init.normal_(self.W_q_embed, mean=0.0, std=0.02)
        for module in [self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _get_queries(self, dtype, device, batch_size):
        if (self._cached_queries is None or 
            self._cached_dtype != dtype or 
            self._cached_device != device):
            
            self._cached_queries = self.W_q_embed.detach().to(dtype=dtype, device=device)
            self._cached_dtype = dtype
            self._cached_device = device
        
        return self._cached_queries.unsqueeze(0).expand(batch_size, -1, -1)
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Get queries, project K/V
        Q = self._get_queries(x.dtype, x.device, batch_size)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head and ensure contiguity
        Q = Q.view(batch_size, self.num_queries, self.num_heads, self.d_k).contiguous()
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).contiguous()
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).contiguous()
        
        softmax_scale = 1.0 / math.sqrt(self.d_k)
        attention_weights = None
        
        # FlashAttention condition
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
            except:
                self.flash_used = False
                attn_output = None
        else:
            self.flash_used = False
            attn_output = None
        
        # Fallback to standard attention
        if not self.flash_used or attn_output is None:
            Q_t = Q.transpose(1, 2)
            K_t = K.transpose(1, 2)
            V_t = V.transpose(1, 2)
            
            scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * softmax_scale
            
            if mask is not None:
                if mask.dim() == 2:
                    if mask.shape[0] == seq_len and mask.shape[1] == seq_len:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    else:
                        mask = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask.to(scores.device) == 0, float('-inf'))
            
            attention_weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attention_weights = self.dropout_layer(attention_weights)
            
            attn_output = torch.matmul(attention_weights, V_t)
            attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Output projection
        attn_output = attn_output.reshape(batch_size, self.num_queries, self.d_model)
        output = self.W_o(attn_output)
        
        if return_attention:
            return output, attention_weights
        return output


# Complete test suite
def test_embedded_query_attention():
    """Comprehensive test of the optimized embedded query attention"""
    
    print("=" * 70)
    print("TESTING OPTIMIZED EMBEDDED QUERY MULTI-HEAD ATTENTION")
    print("=" * 70)
    
    # Test parameters
    d_model = 512
    num_heads = 8
    num_queries = 128
    seq_len = 1024
    batch_size = 2
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")
    
    # Create model
    print("\n2. Creating model...")
    model = Seq_Len_Changing_MHA(
        d_model=d_model,
        num_heads=num_heads,
        num_queries=num_queries,
        use_flash=True,
        dropout=0.1
    ).to(device)
    
    print(f"   Model parameter dtype: {next(model.parameters()).dtype}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 1: FP32 input (standard attention)
    print("\n3. Test 1: FP32 input")
    x_fp32 = torch.randn(batch_size, seq_len, d_model, device=device)
    with torch.no_grad():
        output_fp32 = model(x_fp32)
    print(f"   Output shape: {output_fp32.shape}")
    print(f"   FlashAttention used: {model.flash_used}")
    
    # Reset debug flag
    model._debug_printed = False
    model.flash_used = False
    
    # Test 2: BF16 input with autocast
    print("\n4. Test 2: BF16 input WITH autocast")
    if device.type == 'cuda':
        x_bf16 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)
        print(f"   Input dtype: {x_bf16.dtype}")
        
        with torch.no_grad():
            # Use autocast for the forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_bf16 = model(x_bf16)
        
        print(f"   Output shape: {output_bf16.shape}")
        print(f"   Output dtype: {output_bf16.dtype}")
        print(f"   FlashAttention used: {model.flash_used}")
    else:
        print("   Skipping (CUDA required)")
    
    # Reset debug flag
    model._debug_printed = False
    model.flash_used = False
    
    # Test 3: FP16 input with autocast
    print("\n5. Test 3: FP16 input WITH autocast")
    if device.type == 'cuda':
        x_fp16 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
        print(f"   Input dtype: {x_fp16.dtype}")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output_fp16 = model(x_fp16)
        
        print(f"   Output shape: {output_fp16.shape}")
        print(f"   Output dtype: {output_fp16.dtype}")
        print(f"   FlashAttention used: {model.flash_used}")
    else:
        print("   Skipping (CUDA required)")
    
    # Reset debug flag
    model._debug_printed = False
    model.flash_used = False
    
    # Test 4: With mask
    print("\n6. Test 4: With padding mask")
    mask = torch.ones(batch_size, seq_len, device=device)
    mask[:, seq_len//2:] = 0  # Mask second half
    
    x_masked = torch.randn(batch_size, seq_len, d_model, device=device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_masked, attn = model(x_masked, mask=mask, return_attention=True)
    print(f"   Output shape: {output_masked.shape}")
    print(f"   FlashAttention used: {model.flash_used}")
    if attn is not None:
        print(f"   Attention weights shape: {attn.shape}")
        print(f"   Attention weights range: [{attn.min():.3f}, {attn.max():.3f}]")
    
    # Reset debug flag
    model._debug_printed = False
    model.flash_used = False
    
    # Test 5: Cache efficiency test
    print("\n7. Test 5: Cache efficiency test")
    if device.type == 'cuda':
        # First forward - should cache
        x1 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output1 = model(x1)
        print(f"   First forward - FlashAttention used: {model.flash_used}")
        
        # Reset flag for second forward
        model.flash_used = False
        
        # Second forward - should use cache
        x2 = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output2 = model(x2)
        
        print(f"   Second forward - FlashAttention used: {model.flash_used}")
        print(f"   Both outputs valid: {output1.shape} and {output2.shape}")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETED")
    print("=" + "=" * 70)


# Training example
def training_example():
    """Example of training with the optimized attention"""
    
    print("\n" + "=" * 70)
    print("TRAINING EXAMPLE WITH OPTIMIZED ATTENTION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA required for this example")
        return
    
    # Model configuration
    d_model = 512
    num_heads = 8
    num_queries = 128
    seq_len = 1024
    batch_size = 2
    num_steps = 5
    
    # Create model
    model = Seq_Len_Changing_MHA(
        d_model=d_model,
        num_heads=num_heads,
        num_queries=num_queries,
        use_flash=True,
        dropout=0.1
    ).to(device)
    
    # Use the new GradScaler API
    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"\nTraining for {num_steps} steps...")
    
    for step in range(num_steps):
        # Create random data in fp32 (autocast will convert)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        target = torch.randn(batch_size, num_queries, d_model, device=device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast - this ensures Q, K, V are in bf16
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(x)
            loss = F.mse_loss(output, target)
        
        # Print FlashAttention status for first step
        if step == 0:
            print(f"   FlashAttention used in training: {model.flash_used}")
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"   Step {step+1}: Loss = {loss.item():.6f}")
    
    print("Training completed!")


# Fixed performance benchmark
def benchmark_performance():
    """Benchmark the performance of the optimized attention"""
    
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA required for benchmark")
        return
    
    # Model configuration
    d_model = 512
    num_heads = 8
    num_queries = 128
    seq_len = 1024
    batch_size = 4
    
    model = Seq_Len_Changing_MHA(
        d_model=d_model,
        num_heads=num_heads,
        num_queries=num_queries,
        use_flash=True,
        dropout=0.0
    ).to(device)
    
    import time
    num_runs = 50
    
    # Benchmark 1: FP32 (standard attention)
    print("\n📊 Benchmark 1: FP32 (standard attention)")
    model.use_flash = False  # Force standard attention
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
    
    # Create fresh BF16 tensor (don't reuse FP32 tensor!)
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
    
    # Create fresh FP16 tensor
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


# Diagnostic function
def diagnose_flash_attention():
    """Run diagnostics to check FlashAttention compatibility"""
    
    print("=" * 60)
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
        return
    
    # Test FlashAttention with small tensors
    print(f"\n4. Testing FlashAttention with small tensors:")
    if torch.cuda.is_available():
        device = 'cuda'
        
        # Test with different dtypes
        for dtype in [torch.float16, torch.bfloat16]:
            try:
                # Create small test tensors with the exact same shape as our model would use
                batch, q_len, kv_len, heads, dim = 2, 128, 1024, 8, 64
                Q = torch.randn(batch, q_len, heads, dim, device=device, dtype=dtype).contiguous()
                K = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype).contiguous()
                V = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype).contiguous()
                
                # Try FlashAttention
                out = flash_attn_func(Q, K, V, dropout_p=0.0, causal=False)
                print(f"   ✓ {dtype}: Success! Output shape: {out.shape}")
            except Exception as e:
                print(f"   ✗ {dtype}: Failed - {e}")
    else:
        print("   CUDA not available - skipping tests")


if __name__ == "__main__":
    # First run diagnostics
    diagnose_flash_attention()
    
    # Then run tests
    print("\n")
    test_embedded_query_attention()
    
    # Run training example
    print("\n")
    training_example()
    
    # Run benchmark
    print("\n")
    benchmark_performance()