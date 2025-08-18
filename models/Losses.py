from typing import List, Optional
import numpy as np
import torch
from einops import rearrange
from pathlib import Path

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules_zeus.constants import (
    REWARD_DIFFICULTY_SCALER,  
    REWARD_IMPROVEMENT_WEIGHT,
    REWARD_IMPROVEMENT_MIN_DELTA,
)

class Losses():
    def __init__(self, env, variables):
        self.env = env
        self.variables = variables

        # improve this line regarding missing variables.
        min_sota_delta = torch.tensor([value for (key, value) in REWARD_IMPROVEMENT_MIN_DELTA.items() if key in variables])  # (C,)
        print('=====', min_sota_delta)
        self.min_sota_delta = rearrange(min_sota_delta, 'c -> c 1 1')

    def get_curved_scores(self, raw_rmse: torch.Tensor, gamma: float) -> torch.Tensor:
        # gamma: (B, C, H, W)
        min_rmse = rearrange(torch.tensor((0.1, 0.0, 0.4, 0.4)), 'c -> c 1 1').to(self.env.device)    # Mike!!! Rough. They come from w&d
        max_rmse = raw_rmse * 1.2   # Mike!!! A rouph assumption

        min_rmse = torch.minimum(min_rmse, raw_rmse)
        eq = (min_rmse == max_rmse)

        amin = torch.minimum(min_rmse, max_rmse)
        amax = torch.maximum(min_rmse, max_rmse)
        den  = (amax - amin)
        norm = (raw_rmse - amin) / torch.clamp(den, min=1e-12)
        norm = torch.where(eq, torch.ones_like(norm), norm)

        return torch.pow(norm, gamma)  # gamma.shape = (B, C, H, W)
    
    # def loss(self, target, pred, batch_diff, era5_data_vars):
    def loss(self, target, pred, batch_diff, era5_data_vars):
        # (B,FG,C,H,W), (B,FG,C,H,W), (B, C, H, W), (C,)
        device = pred.device
        rmse = ((pred - target) ** 2).mean(dim=1).sqrt()     # (B, C, H, W)

        if batch_diff is not None:
            baseline_rmse = 0
            print('----', rmse.shape, rmse.device, rmse.dtype, self.min_sota_delta.shape, self.min_sota_delta.dtype)
            improvement = baseline_rmse - rmse  - self.min_sota_delta.to(device)   # (B, C, H, W)  the-higher-the-better
            baseline_improvement = torch.maximum(torch.tensor(0.0), improvement)    # 0.0, (B, C, H, W)  the-higher-the-better
            gamma = torch.pow(REWARD_DIFFICULTY_SCALER, batch_diff * 2 - 1)  # (B, C, H, W)  the-higher-the-more-difficult
            quality_score = self.get_curved_scores(rmse, gamma)       # (B, C, H, W) the-higher-the-worse
            baseline_improvement = rearrange(baseline_improvement, 'b c -> b c 1 1')                # 0.0, (B, C, 1, 1)
            improvement_score = self.get_curved_scores(-baseline_improvement, gamma) # (B, C, H, W) the-higher-the-worse
            rmse = (1 - REWARD_IMPROVEMENT_WEIGHT) * quality_score + REWARD_IMPROVEMENT_WEIGHT * improvement_score    # (B, C, H, W)
        
        reward = rmse * rearrange(era5_data_vars, 'c -> c 1 1')   # (B, C, H, W) the-higher-the-worse
        reward = reward.mean()  # (,)
        rmse = rmse.mean(dim=(0,2,3))   # (C,)

        return reward, rmse # (,), (C,)
