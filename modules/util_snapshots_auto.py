import torch
import inspect
from typing import Any, Optional
from pathlib import Path
import json
import re
from collections import defaultdict

# ========== HIERARCHY TRACKING ==========
_call_stack = []

class ctx:
    def __init__(self, name, module=None, index=None):
        self.name = name
        self.module = module
        self.index = index
        
    def __enter__(self):
        _call_stack.append({
            'name': self.name,
            'module': self.module,
            'index': self.index
        })
        return self
        
    def __exit__(self, *args):
        _call_stack.pop()

def current_path():
    if not _call_stack:
        return 'root'
    
    parts = []
    for frame in _call_stack:
        name = frame['name']
        if frame['index'] is not None:
            name = f"{name}{frame['index']}"
        parts.append(name)
    
    return '.'.join(parts)

# ========== RECURSIVE SAVE FUNCTIONS ==========
def save(ss: Optional[int], x: Any, name: str = None, max_depth: int = 3):
    """Save with FULL hierarchical path and recursive module traversal"""
    if ss is None:
        return
    
    if isinstance(ss, torch.Tensor):
        ss = int(ss.item())
    ss = int(ss)
    
    # Get path from call stack
    stack_path = current_path()
    
    # Build full path
    if stack_path and stack_path != 'root':
        full_path = f"{stack_path}.{name}" if name else stack_path
    else:
        full_path = name if name else 'tensor'
    
    # Start recursive saving
    ss_str = f"snapshot_{ss:06d}"
    _save_down(ss_str, x, full_path, 0, max_depth)

def _save_down(ss: str, x: Any, path: str, depth: int, max_depth: int):
    """Recursively traverse DOWNWARD through structure"""
    if depth > max_depth:
        return
    
    if isinstance(x, torch.nn.Module):
        _save_module(ss, x, path, depth, max_depth)
    elif isinstance(x, dict):
        _save_dict(ss, x, path, depth, max_depth)
    elif isinstance(x, (list, tuple)):
        _save_seq(ss, x, path, depth, max_depth)
    elif isinstance(x, torch.Tensor):
        stats = _get_tensor_stats(x)
        if stats:
            _record(ss, path, stats)

def _save_module(ss: str, mod: torch.nn.Module, path: str, depth: int, max_depth: int):
    """Save module and its parameters WITHOUT recording the module itself"""
    
    # Parameters
    for n, p in mod.named_parameters(recurse=False):
        if p.requires_grad:
            # Use the parameter name directly
            param_path = f"{path}.{n}"
            _save_down(ss, p, param_path, depth + 1, max_depth)
    
    # Buffers (like running_mean in BatchNorm)
    for n, b in mod.named_buffers(recurse=False):
        buffer_path = f"{path}.{n}"
        _save_down(ss, b, buffer_path, depth + 1, max_depth)
    
    # Child modules
    for n, child in mod.named_children():
        child_path = f"{path}.{n}"
        _save_down(ss, child, child_path, depth + 1, max_depth)

def _save_dict(ss: str, d: dict, path: str, depth: int, max_depth: int):
    """Save dictionary contents"""
    for k, v in list(d.items())[:10]:  # Limit to first 10 items
        dict_path = f"{path}.{str(k)}"
        _save_down(ss, v, dict_path, depth + 1, max_depth)

def _save_seq(ss: str, seq: Any, path: str, depth: int, max_depth: int):
    """Save sequence contents"""
    for i, v in enumerate(seq[:10]):  # Limit to first 10 items
        seq_path = f"{path}[{i}]"
        _save_down(ss, v, seq_path, depth + 1, max_depth)

def _get_tensor_stats(x):
    """Extract statistics from tensor"""
    try:
        if x.numel() == 0:
            return None
        
        with torch.no_grad():
            if x.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
                xf = x.float()
            else:
                xf = x
            
            return {
                'mean': float(xf.mean()),
                'std': float(xf.std()),
                'min': float(xf.min()),
                'max': float(xf.max()),
                'shape': list(x.shape)
            }
    except:
        return None

def _record(ss: str, name: str, stats: dict):
    """Record variable stats to file"""
    Path('./snapshots').mkdir(exist_ok=True)
    with open(f'./snapshots/{ss}.jsonl', 'a') as f:
        f.write(json.dumps({'var': name, 'stats': stats}) + '\n')

# ========== READER ==========
class Reader:
    def __init__(self, d='./snapshots'): 
        self.dir = Path(d)
    
    def get(self, ss):
        f = self.dir / f"snapshot_{ss:06d}.jsonl"
        if not f.exists():
            return {}
        data = {}
        with open(f, 'r') as file:
            for line in file:
                record = json.loads(line)
                data[record['var']] = record['stats']
        return data
    
    def list(self):
        return [int(f.stem.replace('snapshot_', '')) for f in sorted(self.dir.glob('*.jsonl'))]

# ========== EXAMPLE ==========
if __name__ == "__main__":
    import torch.nn as nn
    
    class FlashMultiHeadAttention(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.W_q = nn.Linear(dim, dim)
            self.W_k = nn.Linear(dim, dim)
            self.W_v = nn.Linear(dim, dim)
            self.W_o = nn.Linear(dim, dim)
        
        def forward(self, x, ss=None):
            # Save individual weight matrices
            save(ss, self.W_q, 'W_q')
            save(ss, self.W_k, 'W_k')
            save(ss, self.W_v, 'W_v')
            
            # Simulate attention
            att_out = x
            save(ss, att_out, 'att_out')
            
            return self.W_o(att_out)
    
    class FlashEncoderLayer(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.normal = FlashMultiHeadAttention(dim)
            self.ffn = nn.Linear(dim, dim)
        
        def forward(self, x, ss=None):
            with ctx('normal'):
                x = self.normal(x, ss=ss)
            with ctx('ffn'):
                x = self.ffn(x)
                save(ss, self.ffn.weight, 'w')
                save(ss, self.ffn.bias, 'b')
            return x
    
    class ShiftEncoderLayer(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.L0 = FlashEncoderLayer(dim)
            self.L1 = FlashEncoderLayer(dim)
        
        def forward(self, x, ss=None):
            with ctx('L0'):
                x = self.L0(x, ss=ss)
            with ctx('L1'):
                x = self.L1(x, ss=ss)
            return x
    
    class uBlock(nn.Module):
        def __init__(self, block_id=1, dim=64):
            super().__init__()
            self.e = ShiftEncoderLayer(dim)  # encoder
            self.d = ShiftEncoderLayer(dim)  # decoder
        
        def forward(self, x, ss=None):
            with ctx(f'B{3}'):
                with ctx('e'):
                    x = self.e(x, ss=ss)
                with ctx('d'):
                    x = self.d(x, ss=ss)
            return x
    
    # Test with multiple blocks
    model = nn.ModuleList([
        uBlock(1, 64),
        uBlock(2, 64)
    ])
    
    x = torch.randn(4, 64)
    
    # Take snapshot
    for i, block in enumerate(model):
        with ctx(f'B{i+1}'):
            x = block(x, ss=42)
    
    # Read results
    reader = Reader()
    snap = reader.get(42)
    
    print("\n📁 Saved variables:")
    for var in sorted(snap.keys()):
        if any(key in var for key in ['W_q', 'W_k', 'W_v', 'att_out']):
            stats = snap[var]
            print(f"  {var}")
            print(f"     mean={stats['mean']:.4f}, shape={stats['shape']}")