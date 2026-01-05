"""
Enhanced Zeus Swin 3D Transformer with parallel bottleneck blocks.
Standalone module that loads original checkpoints and creates enhanced models.
Preserves the original Zeus_Swin_Block_3D_CDHW.py without modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Tuple, Optional

# We'll import your original module to use its architecture
try:
    from Zeus_Swin_Block_3D_CDHW import (
        Zeus_SwinTransformerBlock3D,
        BasicLayer,
        Encoder,
        Decoder,
        Encoder_Decoder,
        Mlp,
        WindowAttention3D,
        PositionBiasLearner3D_Shpere,
        PatchMerging3D,
        PatchUnmerging3D
    )
    ORIGINAL_MODULE_AVAILABLE = True
except ImportError:
    print("Warning: Original Zeus_Swin_Block_3D_CDHW.py not found in path")
    print("Will create placeholder classes for demonstration")
    ORIGINAL_MODULE_AVAILABLE = False
    # Placeholder classes for demonstration
    class Zeus_SwinTransformerBlock3D(nn.Module):
        pass
    class BasicLayer(nn.Module):
        pass
    class Encoder(nn.Module):
        pass
    class Decoder(nn.Module):
        pass
    class Encoder_Decoder(nn.Module):
        pass


# ============================================================================
# ENHANCED COMPONENTS
# ============================================================================

class ExploratorySwinBlock(nn.Module):
    """
    Exploratory parallel Swin block with smart initialization.
    Creates new parameters that explore around the original's wisdom.
    """
    
    def __init__(self, original_block: Zeus_SwinTransformerBlock3D, 
                 exploration_strength: float = 0.1):
        super().__init__()
        
        # Store reference to original block (will be frozen)
        self.original_block = copy.deepcopy(original_block)
        
        # Freeze original block to preserve learned wisdom
        for param in self.original_block.parameters():
            param.requires_grad = False
        
        # Create explorative parallel block
        self.parallel_block = self._create_explorative_copy(original_block, exploration_strength)
        
        # Zero-initialized gate (parallel block starts silent)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Store dimensions for reference
        self.dim = original_block.dim
        self.num_heads = original_block.num_heads
        self.sWin = original_block.sWin
        
        print(f"Created exploratory block: dim={self.dim}, heads={self.num_heads}, "
              f"exploration={exploration_strength}")
    
    def _create_explorative_copy(self, original_block, exploration_strength):
        """Create a copy with smart exploratory perturbations"""
        parallel_block = copy.deepcopy(original_block)
        
        with torch.no_grad():
            # No need to copy data - deepcopy already did it
            # But we DO need to ensure gradients are enabled for the parallel block
            for param in parallel_block.parameters():
                param.requires_grad = True  # Ensure parallel block is trainable
            
            # Now add exploratory perturbations to the already-copied parameters
            for name, param in parallel_block.named_parameters():
                # Add exploratory perturbation
                if param.dim() >= 2:  # Weight matrices
                    # Get corresponding original parameter for std calculation
                    orig_param = dict(original_block.named_parameters())[name]
                    param_std = orig_param.std().item()
                    noise_std = exploration_strength * max(param_std, 1e-6)
                    
                    # Orthogonal exploration
                    noise = torch.randn_like(param) * noise_std
                    
                    if param.numel() > 1:
                        # Project out component parallel to original direction
                        orig_flat = orig_param.flatten()
                        noise_flat = noise.flatten()
                        
                        parallel_component = (noise_flat @ orig_flat) / (orig_flat @ orig_flat + 1e-8)
                        orthogonal_noise = noise_flat - parallel_component * orig_flat
                        
                        # Reshape back and apply with adjusted scale
                        # Note: orthogonal noise might have different magnitude
                        ortho_norm = orthogonal_noise.norm()
                        orig_norm = noise_flat.norm()
                        if ortho_norm > 1e-8 and orig_norm > 1e-8:
                            scale = orig_norm / ortho_norm
                            orthogonal_noise = orthogonal_noise * scale
                        
                        noise = orthogonal_noise.reshape_as(param) * noise_std
                    
                    param.data.add_(noise)
                
                elif param.dim() == 1:  # Biases
                    noise_std = exploration_strength * 0.01
                    param.data.add_(torch.randn_like(param) * noise_std)
        
        return parallel_block
    
    def forward(self, x):
        """Forward pass: original + gate * (explorative parallel)"""
        # Original (frozen, proven) computation
        with torch.no_grad():
            original_out = self.original_block(x)
        
        # Exploratory parallel computation
        parallel_out = self.parallel_block(x)
        
        # Gated combination
        # When gate=0: output = original_out (exact match)
        # When gate>0: adds exploratory signal
        output = original_out + self.gate * parallel_out
        
        return output


class EnhancedBasicLayer(nn.Module):
    """
    Enhanced BasicLayer with parallel exploratory blocks in bottleneck.
    Standalone - contains both original and enhanced components.
    """
    
    def __init__(self, original_layer: BasicLayer, layer_index: int = 2,
                 enhance_only_bottleneck: bool = True):
        super().__init__()
        
        # Deep copy the original layer structure
        self.original_layer = copy.deepcopy(original_layer)
        
        # Determine if this layer should be enhanced
        self.is_enhanced = (enhance_only_bottleneck and layer_index == 2)
        self.layer_index = layer_index
        
        if self.is_enhanced:
            print(f"🚀 Enhancing bottleneck BasicLayer (index {layer_index})")
            
            # Create exploratory blocks for each original block
            self.exploratory_blocks = nn.ModuleList()
            self.gates = nn.ParameterList()
            
            for i, orig_block in enumerate(original_layer.blocks):
                # Create exploratory parallel block
                exp_block = ExploratorySwinBlock(orig_block, exploration_strength=0.15)
                self.exploratory_blocks.append(exp_block)
                
                # Zero-initialized gate for this block
                gate = nn.Parameter(torch.zeros(1))
                self.gates.append(gate)
            
            # Freeze original layer to preserve wisdom
            for param in self.original_layer.parameters():
                param.requires_grad = False
            
            print(f"  Added {len(self.exploratory_blocks)} exploratory blocks "
                  f"with zero-initialized gates")
        
        # Copy resampler if exists
        if hasattr(original_layer, 'resampler') and original_layer.resampler is not None:
            self.resampler = copy.deepcopy(original_layer.resampler)
        else:
            self.resampler = None
        
        # Copy other attributes
        self.dim = original_layer.dim
        self.i_reso = original_layer.i_reso
        self.depth = original_layer.depth
        self.use_checkpoint = original_layer.use_checkpoint
        self.o_reso = original_layer.o_reso
        self.padding = original_layer.padding
        self.in_encoder = original_layer.in_encoder
        self.blocks = self.original_layer.blocks  # For compatibility
    
    def forward(self, x):
        """Forward pass through enhanced layer"""
        if not self.is_enhanced:
            # For non-enhanced layers, use original forward
            return self.original_layer(x)
        
        # Enhanced forward: iterate through blocks
        layer_output = []
        
        # Process through each block with exploratory enhancement
        for i, (orig_block, exp_block, gate) in enumerate(zip(
            self.original_layer.blocks,
            self.exploratory_blocks,
            self.gates
        )):
            if self.use_checkpoint:
                import torch.utils.checkpoint as checkpoint
                # Original path (frozen)
                with torch.no_grad():
                    orig_out = checkpoint.checkpoint(orig_block, x)
                # Exploratory path
                exp_out = checkpoint.checkpoint(exp_block, x)
            else:
                # Original path (frozen)
                with torch.no_grad():
                    orig_out = orig_block(x)
                # Exploratory path
                exp_out = exp_block(x)
            
            # Gated combination
            x = orig_out + gate * exp_out
            
            # Store intermediate if needed (for skip connections)
            if i == 0:  # First block output
                layer_output.append(x)
        
        # Apply resampler if exists
        if self.resampler is not None:
            x = self.resampler(x)
            layer_output.append(x)
        else:
            layer_output.append(x)
        
        return layer_output
    
    def extra_repr(self):
        return (f"EnhancedBasicLayer(dim={self.dim}, i_reso={self.i_reso}, "
                f"depth={self.depth}, enhanced={self.is_enhanced})")


# ============================================================================
# ENHANCED ENCODER-DECODER
# ============================================================================

class EnhancedEncoder(nn.Module):
    """
    Enhanced Encoder with parallel blocks in bottleneck BasicLayer.
    """
    
    def __init__(self, original_encoder: Encoder):
        super().__init__()
        
        # Copy all attributes from original encoder
        self.num_layers = original_encoder.num_layers
        self.embed_dim = original_encoder.embed_dim
        self.ape = original_encoder.ape
        self.mlp_ratio = original_encoder.mlp_ratio
        self.skippable = original_encoder.skippable
        
        # Copy absolute position embedding if exists
        if hasattr(original_encoder, 'absolute_pos_embed'):
            self.absolute_pos_embed = nn.Parameter(
                original_encoder.absolute_pos_embed.data.clone()
            )
        
        # Copy position dropout
        self.pos_drop = copy.deepcopy(original_encoder.pos_drop)
        
        # Create enhanced layers
        self.layers = nn.ModuleList()
        for i, original_layer in enumerate(original_encoder.layers):
            # Only enhance the bottleneck layer (index 2)
            enhanced_layer = EnhancedBasicLayer(
                original_layer, 
                layer_index=i,
                enhance_only_bottleneck=True
            )
            self.layers.append(enhanced_layer)
        
        print(f"Enhanced Encoder created: {self.num_layers} layers, "
              f"bottleneck layer enhanced")
    
    def forward(self, x):
        """Forward pass through enhanced encoder"""
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        encoder_output = []
        
        for idx, layer in enumerate(self.layers):
            layer_output = layer(x)
            x = layer_output[-1]  # Get the output after resampler
            
            if self.skippable or (not self.skippable and idx == len(self.layers)-1):
                encoder_output.append(layer_output)
        
        return encoder_output


class EnhancedDecoder(nn.Module):
    """
    Enhanced Decoder with parallel blocks in bottleneck BasicLayer.
    """
    
    def __init__(self, original_decoder: Decoder):
        super().__init__()
        
        # Copy all attributes
        self.num_layers = original_decoder.num_layers
        self.embed_dim = original_decoder.embed_dim
        self.ape = original_decoder.ape
        self.mlp_ratio = original_decoder.mlp_ratio
        self.skip_learnable = original_decoder.skip_learnable
        self.skippable = original_decoder.skippable
        
        # Copy skip learners if they exist
        if self.skippable and self.skip_learnable:
            self.skip_learners = nn.ModuleList()
            for skip_learner in original_decoder.skip_learners:
                self.skip_learners.append(copy.deepcopy(skip_learner))
        
        # Create enhanced layers
        self.layers = nn.ModuleList()
        for i, original_layer in enumerate(original_decoder.layers):
            # Only enhance the bottleneck layer (index 2)
            enhanced_layer = EnhancedBasicLayer(
                original_layer,
                layer_index=i,
                enhance_only_bottleneck=True
            )
            self.layers.append(enhanced_layer)
        
        print(f"Enhanced Decoder created: {self.num_layers} layers, "
              f"bottleneck layer enhanced")
    
    def forward(self, enc_x_by_layer):
        """Forward pass through enhanced decoder"""
        if self.skippable:
            dec_x_after_us = torch.zeros_like(enc_x_by_layer[-1][0])
            
            if self.skip_learnable:
                for idx, (dec_layer, (enc_x_before_ds, _), skip_learner) in enumerate(
                    zip(reversed(self.layers), reversed(enc_x_by_layer), reversed(self.skip_learners))
                ):
                    input_tensor = dec_x_after_us + skip_learner(enc_x_before_ds)
                    (_, dec_x_after_us) = dec_layer(input_tensor)
            else:
                for idx, (dec_layer, (enc_x_before_ds, _)) in enumerate(
                    zip(reversed(self.layers), reversed(enc_x_by_layer))
                ):
                    input_tensor = dec_x_after_us + enc_x_before_ds
                    (_, dec_x_after_us) = dec_layer(input_tensor)
        else:
            dec_x_after_us = enc_x_by_layer[-1][0]
            for layer in reversed(self.layers):
                (_, dec_x_after_us) = layer(dec_x_after_us)
        
        return dec_x_after_us


class EnhancedEncoderDecoder(nn.Module):
    """
    Complete enhanced encoder-decoder model.
    Standalone - can be saved/loaded independently.
    """
    
    def __init__(self, original_model: Encoder_Decoder):
        super().__init__()
        
        print("=" * 60)
        print("Creating Enhanced Zeus Swin 3D Model")
        print("=" * 60)
        
        # Create enhanced encoder and decoder
        self.encoder = EnhancedEncoder(original_model.encoder)
        self.decoder = EnhancedDecoder(original_model.decoder)
        
        # Copy skip connection if exists
        self.skippable = original_model.skippable
        self.skip_learnable = original_model.skip_learnable
        
        if self.skippable and self.skip_learnable:
            self.skip_learner = copy.deepcopy(original_model.skip_learner)
        
        print(f"Enhanced model created: skippable={self.skippable}, "
              f"skip_learnable={self.skip_learnable}")
        print("=" * 60)
    
    def forward(self, x_flat):
        """Forward pass through enhanced model"""
        if self.skippable:
            x_skip = x_flat
        
        x_by_layer = self.encoder(x_flat)
        x = self.decoder(x_by_layer)
        
        if self.skippable:
            if self.skip_learnable:
                x = x + self.skip_learner(x_skip)
            else:
                x = x + x_skip
        
        return x


# ============================================================================
# CHECKPOINT HANDLING AND TRANSITION
# ============================================================================

def load_original_checkpoint(checkpoint_path: str, device='cpu'):
    """
    Load original model checkpoint.
    
    Args:
        checkpoint_path: Path to original checkpoint file
        device: Device to load onto
    
    Returns:
        Dictionary with model_state_dict, optimizer_state_dict, etc.
    """
    print(f"Loading original checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loss: {checkpoint.get('loss', 'N/A'):.6f}")
    
    return checkpoint


def create_enhanced_from_checkpoint(
    original_model_class,  # Your original Encoder_Decoder class
    checkpoint_path: str,
    model_config: dict,
    save_enhanced_path: Optional[str] = None,
    device: str = 'cpu'
) -> EnhancedEncoderDecoder:
    """
    Create enhanced model from original checkpoint in one step.
    
    Args:
        original_model_class: The original Encoder_Decoder class
        checkpoint_path: Path to original checkpoint
        model_config: Configuration dictionary for model initialization
        save_enhanced_path: Optional path to save enhanced model
        device: Device to create model on
    
    Returns:
        EnhancedEncoderDecoder model ready for training
    """
    print("\n" + "=" * 60)
    print("TRANSITIONING TO ENHANCED MODEL")
    print("=" * 60)
    
    # Step 1: Load original checkpoint
    checkpoint = load_original_checkpoint(checkpoint_path, device)
    
    # Step 2: Create original model with saved weights
    print("\nCreating original model...")
    original_model = original_model_class(**model_config).to(device)
    
    # Load weights into original model
    model_state_dict = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = original_model.load_state_dict(
        model_state_dict, strict=True
    )
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    print("✓ Original model loaded successfully")
    
    # Step 3: Create enhanced model
    print("\nCreating enhanced model with exploratory parallel blocks...")
    enhanced_model = EnhancedEncoderDecoder(original_model).to(device)
    
    # Verify enhancement
    print("\nVerifying enhancement...")
    
    # Create test input
    B, C, D, H, W = 1, model_config.get('embed_dim', 192), 8, 45, 90
    test_input = torch.randn(B, D * H * W, C).to(device)
    
    # Set gates to zero initially
    with torch.no_grad():
        for name, param in enhanced_model.named_parameters():
            if 'gate' in name:
                param.zero_()
    
    # Run both models
    original_model.eval()
    enhanced_model.eval()
    
    with torch.no_grad():
        orig_output = original_model(test_input)
        enh_output = enhanced_model(test_input)
    
    # Check that outputs match initially (gates = 0)
    diff = torch.abs(orig_output - enh_output).max().item()
    
    if diff < 1e-6:
        print(f"✓ Enhancement verified! Max difference: {diff:.2e}")
        print("  Enhanced model matches original when gates=0")
    else:
        print(f"⚠ Warning: Models differ by {diff:.2e}")
        print("  This may be due to minor numerical differences")
    
    # Count parameters
    orig_params = sum(p.numel() for p in original_model.parameters())
    enh_params = sum(p.numel() for p in enhanced_model.parameters())
    added_params = enh_params - orig_params
    
    print(f"\nParameter counts:")
    print(f"  Original model: {orig_params:,} parameters")
    print(f"  Enhanced model: {enh_params:,} parameters")
    print(f"  Added: {added_params:,} parameters ({added_params/orig_params*100:.2f}% increase)")
    
    # Step 4: Create enhanced checkpoint
    enhanced_checkpoint = {
        'model_state_dict': enhanced_model.state_dict(),
        'original_checkpoint': checkpoint_path,
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'config': {
            'model_config': model_config,
            'enhanced': True,
            'enhancement_type': 'exploratory_parallel_blocks',
            'enhanced_layers': 'bottleneck_only',
            'zero_initialized_gates': True,
            'parameter_increase_pct': f"{added_params/orig_params*100:.2f}%"
        }
    }
    
    # Add optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        enhanced_checkpoint['original_optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        print("✓ Original optimizer state preserved in checkpoint")
    
    # Save if requested
    if save_enhanced_path:
        torch.save(enhanced_checkpoint, save_enhanced_path)
        print(f"\n✓ Enhanced model saved to: {save_enhanced_path}")
    
    print("\n" + "=" * 60)
    print("TRANSITION COMPLETE")
    print("=" * 60)
    
    return enhanced_model, enhanced_checkpoint


def create_enhanced_optimizer(enhanced_model, original_optimizer_state=None, 
                             lr_gates=1e-3, lr_exploration=1e-4, lr_original=1e-5):
    """
    Create optimizer for enhanced model with differential learning rates.
    
    Args:
        enhanced_model: The enhanced model
        original_optimizer_state: State dict from original optimizer (optional)
        lr_gates: Learning rate for gate parameters
        lr_exploration: Learning rate for exploratory block parameters
        lr_original: Learning rate for original (frozen) parameters if unfrozen later
    
    Returns:
        Optimizer configured for phased training
    """
    # Separate parameters by type
    gate_params = []
    exploration_params = []
    original_params = []
    
    for name, param in enhanced_model.named_parameters():
        if param.requires_grad:
            if 'gate' in name:
                gate_params.append(param)
            elif 'exploratory' in name or 'parallel_block' in name:
                exploration_params.append(param)
            else:
                original_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = []
    
    if gate_params:
        param_groups.append({
            'params': gate_params,
            'lr': lr_gates,
            'weight_decay': 0.01,
            'name': 'gates'
        })
    
    if exploration_params:
        param_groups.append({
            'params': exploration_params,
            'lr': lr_exploration,
            'weight_decay': 0.01,
            'name': 'exploration'
        })
    
    if original_params:
        param_groups.append({
            'params': original_params,
            'lr': lr_original,
            'weight_decay': 0.01,
            'name': 'original'
        })
    
    # Create optimizer
    optimizer = torch.optim.AdamW(param_groups)
    
    # Optionally load original optimizer state (for momentum, etc.)
    if original_optimizer_state:
        # Note: This requires careful matching of parameter shapes
        print("Warning: Loading original optimizer state may not align perfectly")
        # optimizer.load_state_dict(original_optimizer_state)  # Use with caution
    
    print(f"Created optimizer with {len(param_groups)} parameter groups:")
    for group in param_groups:
        print(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']}")
    
    return optimizer


# ============================================================================
# TRAINING MANAGER FOR ENHANCED MODEL
# ============================================================================

class EnhancedModelTrainer:
    """
    Manages phased training of the enhanced model.
    """
    
    def __init__(self, enhanced_model, device='cuda'):
        self.model = enhanced_model
        self.device = device
        self.current_phase = 0
        
        # Track gate statistics
        self.gate_history = []
        
    def set_training_phase(self, phase):
        """
        Phase 0: Train only gates (discover useful explorations)
        Phase 1: Train gates + exploration blocks (refine explorations)
        Phase 2: Train everything (full fine-tuning)
        """
        print(f"\nSetting training phase: {phase}")
        
        # Freeze everything first
        self._freeze_all()
        
        if phase == 0:
            # Phase 0: Only train gates
            self._unfreeze_only_gates()
            print("  Training: Gates only")
            print("  Purpose: Discover which exploratory directions are useful")
            
        elif phase == 1:
            # Phase 1: Train gates + exploration blocks
            self._unfreeze_gates_and_exploration()
            print("  Training: Gates + exploration blocks")
            print("  Purpose: Refine useful exploratory directions")
            
        elif phase == 2:
            # Phase 2: Train everything
            self._unfreeze_all()
            print("  Training: Full model")
            print("  Purpose: Full fine-tuning")
        
        self.current_phase = phase
    
    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _unfreeze_only_gates(self):
        for name, param in self.model.named_parameters():
            if 'gate' in name:
                param.requires_grad = True
    
    def _unfreeze_gates_and_exploration(self):
        for name, param in self.model.named_parameters():
            if 'gate' in name or 'exploratory' in name or 'parallel_block' in name:
                param.requires_grad = True
    
    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def monitor_gates(self):
        """Monitor and record gate statistics"""
        gate_values = []
        gate_names = []
        
        for name, param in self.model.named_parameters():
            if 'gate' in name and param.requires_grad:
                gate_values.append(param.data.mean().item())
                gate_names.append(name)
        
        if gate_values:
            avg_gate = sum(gate_values) / len(gate_values)
            max_gate = max(gate_values)
            min_gate = min(gate_values)
            
            self.gate_history.append({
                'phase': self.current_phase,
                'avg': avg_gate,
                'max': max_gate,
                'min': min_gate,
                'values': gate_values.copy()
            })
            
            print(f"Gate statistics:")
            print(f"  Average: {avg_gate:.4f}")
            print(f"  Range: [{min_gate:.4f}, {max_gate:.4f}]")
            
            # Suggest phase transitions
            if self.current_phase == 0 and avg_gate > 0.2:
                print("  → Consider moving to Phase 1 (gates finding useful directions)")
            elif self.current_phase == 1 and avg_gate > 0.5:
                print("  → Consider moving to Phase 2 (full fine-tuning)")
        
        return gate_values
    
    def save_checkpoint(self, epoch, optimizer, loss, path):
        """Save enhanced model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'phase': self.current_phase,
            'gate_history': self.gate_history,
            'enhanced': True
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path, optimizer=None):
        """Load enhanced model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'gate_history' in checkpoint:
            self.gate_history = checkpoint['gate_history']
        
        print(f"Checkpoint loaded: epoch={checkpoint.get('epoch', 'N/A')}, "
              f"loss={checkpoint.get('loss', 'N/A'):.6f}")
        
        return checkpoint.get('epoch', 0)


# ============================================================================
# MAIN TRANSITION SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the enhancement system.
    Run this to transition your saved checkpoint to an enhanced model.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Transition to enhanced Zeus Swin 3D model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to original checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Model configuration (as Python dict string)')
    parser.add_argument('--save_path', type=str, default='enhanced_model.pth',
                       help='Path to save enhanced model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Parse model config (example)
    # Example config string: "{'i_reso': (8, 180, 360), 'embed_dim': 192, ...}"
    try:
        import ast
        model_config = ast.literal_eval(args.config)
    except:
        print("Error parsing config. Using example config.")
        model_config = {
            'i_reso': (8, 180, 360),
            'embed_dim': 192,
            'depths': [4, 8, 4],
            'num_heads': [4, 8, 16],
            'sWin': (2, 6, 12),
            'skippable': True,
            'skip_learnable': True,
            'pair_bias_dim': 5
        }
    
    # Import your original model class
    # This assumes Zeus_Swin_Block_3D_CDHW.py is in the path
    if ORIGINAL_MODULE_AVAILABLE:
        from Zeus_Swin_Block_3D_CDHW import Encoder_Decoder
        original_model_class = Encoder_Decoder
    else:
        print("Original module not found. Using demonstration mode.")
        original_model_class = Encoder_Decoder  # Placeholder
    
    # Create enhanced model from checkpoint
    enhanced_model, enhanced_checkpoint = create_enhanced_from_checkpoint(
        original_model_class=original_model_class,
        checkpoint_path=args.checkpoint,
        model_config=model_config,
        save_enhanced_path=args.save_path,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("ENHANCED MODEL READY FOR TRAINING")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create trainer: trainer = EnhancedModelTrainer(enhanced_model)")
    print("2. Create optimizer: optimizer = create_enhanced_optimizer(enhanced_model)")
    print("3. Train in phases:")
    print("   - Phase 0: trainer.set_training_phase(0)  # Gates only")
    print("   - Phase 1: trainer.set_training_phase(1)  # Gates + exploration")
    print("   - Phase 2: trainer.set_training_phase(2)  # Full model")
    print("\nThe enhanced model is now standalone and can be saved/loaded independently.")