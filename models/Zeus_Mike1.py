import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import math
from einops import rearrange
import random

from pathlib import Path
import os, sys
from SphericalCodec import SphericalEncoder, SphericalDecoder

sys.path.append(str(Path(__file__).resolve().parent.parent))

class Zeus(nn.Module):
    def __init__(self, pm, device='cpu'):
        pass
