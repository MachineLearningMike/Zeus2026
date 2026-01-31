import torch
import torch.nn as nn
from datetime import datetime, timedelta, timezone
import time

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.zeus_dataset_Snapshot import Snapshot
from modules.zeus_dateset_geo_tempo import Geo_Tempo
from modules.Flash_Nonlinear_Attention import PhysicalFlashAttention
from modules.RoPE_LatLon import RoPE


# -----------------------
# Zeus model (pre-rotated inputs)
# -----------------------
class Zeus(nn.Module):
    def __init__(self, pm, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.pm = pm
        
        self.H = pm.height
        self.W = pm.width
        self.C = Snapshot.meteo_dims + Snapshot.tempo_dims + Snapshot.geo_dense_dims
        self.nHeads = pm.nHeads
        self.device = device
        self.dtype = dtype
        self.nLayers = pm.nLayers
        
        # Stack multiple PhysicalFlashAttention layers
        self.attn_layers = nn.ModuleList([
            PhysicalFlashAttention(dim=self.C, phys_channels=1,  H=self.H, W=self.W, nHeads=self.nHeads, dropout=0.0).to(device=self.device, dtype=self.dtype)
            for _ in range(self.nLayers)
        ])
        
        
        #-----------------------
        self.predict_fun_dict = None

    def forward(self, p, x):
        """
        p, x: (B, H, W, C)
        p: simple physical prediction of y
        """
        B, H, W, C = x.shape
        assert (B, H, W, C) == p.shape
        assert (H, W, C) == (self.H, self.W, self.C)
                                
        # Flatten features
        x_flat = x.reshape(B, N, C)
        p_flat = p.reshape(B, N, C)

        # Apply RoPE
        x_rot = rope.apply_rope(x_flat).reshape(B, H, W, C)
        p_rot = rope.apply_rope(p_flat).reshape(B, H, W, C)
        
        for i, attn in enumerate(self.attn_layers):
            x_rot = attn(p_rot, x_rot)
            p_rot = x_rot
        
        return x_rot + p
    
    def physical_predict(self, ss_x, ss_y, modulo, variables):

        from datetime import datetime, timedelta
        from light.predict_t2m_n_hours import T2mPredictor
        from light.predict_tp_n_hours import TpPredictor    
        from light.predict_u100m_n_hours import U100mPredictor
        from light.predict_v100m_n_hours import V100mPredictor
        from light.predict_d2m_n_hours import D2mPredictor
        from light.predict_sp_n_hours import SpPredictor

        if self.predict_fun_dict is None:
            self.predict_fun_dict = {"t2m": T2mPredictor(), "tp": TpPredictor(), "u100m": U100mPredictor(), "v100m": V100mPredictor(), "d2m": D2mPredictor(), "sp": SpPredictor()}

        varwise_tensors = []
        for var in variables:
            dt_x = ss_x.get_datetime()
            var_tensor = self.fun_dict[var].predict(ss_x.meteo_raw, ss_x.geo_dense, ss_x.geo_cat, ss_x.tempo, ss_y.tempo, dt_x, delta_hours=modulo).cpu().numpy()
            varwise_tensors.append(var_tensor)
        
        predict = torch.stack(varwise_tensors, dim=0)

        return predict
    
    # def loss(self, target, pred, batch_diff, zeus_loss_weights_on_target_vars):
    def loss(self, target, pred, batch_diff=None, zeus_loss_weights_on_target_vars=None, vars_to_mask_on_loss=None, area_weights=None):
        # (C,H,W), (C,H,W), (C, H, W), (C,)
        # 32, cuda, False ; 32, cuda, True !!!, 32, cuda, False, 32, cuda, False
        device = pred.device
        target = target.to(device)
        vars_to_mask_on_loss = vars_to_mask_on_loss.to(device)

        C, H, W = target.shape
        error_sqaured = (pred - target) ** 2 * (area_weights if area_weights is not None else 1.0)   # (C, H, W)
        sum_weights = area_weights.sum() * W if area_weights is not None else H * W
        rmse = torch.sqrt(error_sqaured.sum(dim=(1,2)) / sum_weights)  # (C,)
        rmse = rmse * (vars_to_mask_on_loss if vars_to_mask_on_loss is not None else 1.0)     # (C,)
        loss = (rmse * self.pm.loss_amplification_on_target_vars * (zeus_loss_weights_on_target_vars if zeus_loss_weights_on_target_vars is not None else 1.0)).mean()

        # self._print_tensor_state(1, target)                 # 
        # self._print_tensor_state(2, pred)
        # self._print_tensor_state(3, batch_diff)
        # self._print_tensor_state(4, zeus_loss_weights_on_target_vars)
        # self._print_tensor_state(4.5, rmse)

        # if batch_diff is not None:
        #     baseline_rmse = 0
        #     improvement = baseline_rmse - rmse  - self.min_sota_delta.to(device)   # (B, C, H, W), 32, cuda, True,  the-higher-the-better
        #     # self._print_tensor_state(5, improvement)
        #     baseline_improvement = torch.maximum(torch.tensor(0.0).to(device), improvement)    # 0.0, (B, C, H, W), 32, cuda, True !!!,  the-higher-the-better
        #     # self._print_tensor_state(6, baseline_improvement)
        #     gamma = torch.pow(REWARD_DIFFICULTY_SCALER, batch_diff * 2 - 1).to(device)  # (B, C, H, W), 32, cuda, False,  the-higher-the-more-difficult
        #     # self._print_tensor_state(7, gamma)
        #     quality_score = self.get_curved_scores(rmse, gamma)       # (B, C, H, W), 32, cuda, True !!!, the-higher-the-worse
        #     # self._print_tensor_state(8, quality_score)
        #     improvement_score = self.get_curved_scores(-baseline_improvement, gamma) # (B, C, H, W), 32, cuda, True !!!, the-higher-the-worse
        #     # self._print_tensor_state(10, improvement_score)
        #     rmse = (1 - REWARD_IMPROVEMENT_WEIGHT) * quality_score + REWARD_IMPROVEMENT_WEIGHT * improvement_score    # (B, C, H, W), 32, cuda, True !!!
        #     # self._print_tensor_state(11, rmse)
        
        # zeus_loss_weights_on_target_vars = zeus_loss_weights_on_target_vars.to(device)
        # reward = rmse * zeus_loss_weights_on_target_vars   # (C,), 32, cuda, True !!!, the-higher-the-worse
        # self._print_tensor_state(12, reward)
        # reward = reward.mean()  # (,)

        # return rmse.mean(), rmse # (,), (C,)     32, cuda, True

        # zeus_loss_weights_on_target_vars = zeus_loss_weights_on_target_vars.to(device)
        
        return loss, rmse    
        
    def forward_backward_no_optimizer_step(self,       
            DnSS0, DnSS1, DnSSY, batch_diff, cascade_step, trainable, scaler, predicted_snapshot_required,
            zeus_loss_weights_on_target_vars, train_dict,
        ):
        
        assert DnSS0 == DnSS1   # for this task.
        device = next(self.attn_layers[0].parameters()).device        
        
        # 1. get physical prediction of y_meteo_raw                
        # used: DnSS0.meteo_raw, DnSS0.geo_dense, DnSS0.geo_cat, DnSS0.tempo, DnSSY.tempo, DnSS0.datatime
        physical_prediction_raw = self.physical_predict(self, DnSS0, DnSSY, self.pm.modulo, ["t2m"])   # 

        # 2. prediction_normal
        physical_prediction_normal = train_dict['fn_normalize_meteo_phdn'](physical_prediction_raw)        
        
        # 3. Enrich prediction_normal with geo-info
        PhDn = Snapshot(DnSSY.year, DnSSY.month, DnSSY.day, DnSSY.hour, meteo_normal=physical_prediction_normal, meteo_raw=None)
        surface_meteo_normal_tempo_dense_p = Snapshot.pack_pair_snapshots(PhDn, PhDn, device=device)
        surface_meteo_normal_tempo_dense_p = self.enrich_mixer(surface_meteo_normal_tempo_dense_p)

        # 4. get surface_meteo_normal_tempo_dense_x
        surface_meteo_normal_tempo_dense_x = Snapshot.pack_pair_snapshots(DnSS0, DnSS0, device=device)
        
        # 4. forward through model
        surface_meteo_normal_tempo_dense_pred = self(surface_meteo_normal_tempo_dense_p, surface_meteo_normal_tempo_dense_x)
        
        # 5. Derich surface_meteo_normal_tempo_dense_pred learnably
        meteo_normal_pred = self.derich_mixer(surface_meteo_normal_tempo_dense_pred)
        
        #--------------------------------------------------------------------------------------------------------
        
        lossAndMetrics = self.loss(DnSSY.meteo_normal, meteo_normal_pred, batch_diff, 
                zeus_loss_weights_on_target_vars, train_dict['area_weights'])        
        normalized_loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics   #---- [0] picks loss

        if trainable:
            scaler.scale(normalized_loss).backward(retain_graph=True)

        meteo_normal_pred = meteo_normal_pred.detach()
            
        #------------------------------ pred_normal is no longer user below -------------------
        meteo_raw_pred = meteo_normal_pred
        train_dict['fn_rawize_meteo'](meteo_normal_pred, meteo_raw_pred)

        if predicted_snapshot_required:
            
            predicted_ss = Snapshot(DnSSY.year, DnSSY.month, DnSSY.day, DnSSY.hour, meteo_normal=meteo_normal_pred.to(DnSSY.meteo_normal.device), meteo_raw=meteo_raw_pred)
        else:
            predicted_ss = None

        
        if train_dict['type_metrics'] == "rmse":
            lossAndMetrics = self.loss(DnSSY.meteo_raw, meteo_raw_pred, batch_diff, area_weights=train_dict['area_weights'])
            rawized_metrics = lossAndMetrics[1] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics   #---- [1] picks rmse
        elif train_dict['type_metrics'] == "mae":
            rawized_metrics = torch.abs(DnSSY.meteo_raw - meteo_raw_pred).mean(dim=(1, 2))    # (C,)
        else:
            raise Exception("type_metrics is invalid")
                
        return predicted_ss, normalized_loss.detach().to("cpu"), rawized_metrics.detach().to("cpu")

# -----------------------
# Test code
# -----------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    from models.config_Mike1 import config
    from modules.params_Mike1 import Params, flatten_pm_single
    pm = Params(config)    
    flatten_pm = flatten_pm_single(pm, 0)
        
    model = Zeus(flatten_pm, device=device)
    model.to(device)

    
    from modules.zeus_dateset_geo_tempo import Geo_Tempo
    from modules.RoPE_LatLon import RoPE
    
    factor10 = 2.5
    num_heads = 2
    B = 1

    H, W = int(721*factor10/10), int(1440*factor10/10)
    C = 10 * int(10/factor10 * 10/factor10)
    
    # -----------------------------
    # RoPE
    # -----------------------------
    latlon, _, _ = Geo_Tempo.lat_lon_grid_tensor(H, W)
    latlon = torch.deg2rad(latlon).to(device=device, dtype=dtype)
    lat_grid, lon_grid = latlon[0], latlon[1]

    N = H * W
    lat_flat = lat_grid.reshape(1, N).expand(B, N)
    lon_flat = lon_grid.reshape(1, N).expand(B, N)

    rope = RoPE(lat_flat, lon_flat, dims=C)

    # -----------------------------
    # Dummy inputs
    # -----------------------------
    p = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)

    # -----------------------------
    # Forward through Zeus
    # -----------------------------
    model = Zeus(H, W, C, num_heads=num_heads, num_layers=1, device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )

    print(f"starting...")

    loops = 10
    torch.cuda.synchronize()
    s = time.time()
    
    for loop in range(loops):
        optimizer.zero_grad(set_to_none=True)
        y = model(p, x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        # print(f"loop {loop}")
        
    torch.cuda.synchronize()
    e = time.time()
    print(f"time taken: {(e-s)/loops:.3f}")

    # print("Output shape:", y.shape)
    # print("Output dtype:", y.dtype)
    # print("Output mean:", y.mean().item())

    # print("x.grad:", x.grad is not None)
    # print("p.grad:", p.grad is not None)
    print("✅ Pre-RoPE Zeus test passed successfully.")
