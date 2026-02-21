import torch
import torch.nn as nn
from datetime import datetime, timedelta, timezone
import time

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.zeus_dataset_Snapshot import Snapshot
from modules.uNet import uNet
# from modules.Flash_Nonlinear_Attention import PhysicalFlashAttention
# from modules.Zeus_Axial_Non_Lin_Attn_From_Phys_FA_Axial import AxialFlashAttention
from modules.Zeus_Axial_Attn_Phys_FA_Cross_PE import AxialFlashAttention
from modules.Reshape_HX_4X import Reshape_HX_4X, Reshape_2X_QX
from modules.utils import resize_continuous_bilinear_channel_wise_cylindric
from modules.zeus_embed_cascade import Embed_Cascade
from modules.zeus_embed_category import Embed_Category
# from modules.RoPE_LatLon import RoPE

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -----------------------
# Zeus model (pre-rotated inputs)
# -----------------------
class Zeus(nn.Module):
    def __init__(self, pm, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.pm = pm
        
        self.H = pm.height
        self.W = pm.width
        self.C = pm.attn_dim         #--------------------------------------------
        self.nHeads = pm.nHeads
        self.device = device
        self.dtype = dtype
        self.nLayers = pm.nLayers
        axial_loops = pm.axial_loops
        axial_not_total_pe = pm.axial_not_total_pe
        
        pe_type = "learnable"
        
        print("------------------------------------------ Zeus parameters ... ---------------------------------")
        print(f"factor10 {pm.factor10}, pair_not_single_snapshot {pm.pair_not_single_snapshot}, steps_to_save {pm.steps_to_save}, batch_size {pm.batch_size}")
        print(f"(H W C) ({self.H} {self.W} {self.C}), nHeads {self.nHeads} dtype {self.dtype} nLayers {self.nLayers} axial_loops {pm.axial_loops}")
        print(f"axial_not_total_pe {axial_not_total_pe}, pos_type {pe_type}, nSubstepsPerStep {pm.nSubstepsPerStep}, max_cascade_depth {pm.max_cascade_depth}")
        print("------------------------------------------ ... Zeus parameters ---------------------------------")

        
        Snapshot.set_original_settings(factor10=self.pm.factor10,
            device=self.device, dtype=self.pm.torch_dtype,
            dense_variables = self.pm.dense_variables,
            cat_variables = self.pm.cat_variables,
            tempo_variables = self.pm.tempo_variables,
            tempo_dense_cat_rel_size=(1/2, 1/2, 1/2),       # 10//self.pm.factor10)
            )
        
        
        self.uNet = uNet(self.H//2, self.W//2, self.C, pm.block_dicts)   #.to(device=device, dtype=dtype)

        
        
        self.cascade_depth_embeder = Embed_Cascade(self.H//2, self.W//2, max_cascade_depth=10, dims_cascade_emb=2)
        self.catetory_embedder = Embed_Category()
        self.cascade_depth_emb_dim = self.cascade_depth_embeder.dims_cascade_emb
        self.category_emb_dim = self.catetory_embedder.dims_category_emb
        
        in_features = self.get_meteo4_tempo_dense_catemb_dim()
        out_features = self.C
        self.m4tdc_to_attn_dims = Mlp(
            in_features = in_features,
            hidden_features = 4 * in_features,
            out_features = out_features,      # Add in pm
            act_layer = nn.Tanh,
            drop = 0    
        )
        in_features = out_features
        out_features = self.get_meteo4_dim()
        self.attn_dim_to_meteo4 = Mlp(
            in_features = in_features,
            hidden_features = 4 * in_features,
            out_features = Snapshot.meteo_dims * 4,      # Add in pm
            act_layer = nn.Tanh,
            drop = 0    
        )
        
        self.patch_merger = Reshape_HX_4X(dim=self.C)         # dim is not used actually
        self.patch_splitter = Reshape_2X_QX(dim=4*self.C)   # dim is not used actually.
        
        self.predict_fun_dict = None
        self.meteo_dims = len(self.pm.targetVariables)
        
        # self.to(self.pm.torch_dtype) # This line will causes no float32 master copy, breaks autocast's automatic management.

    def get_meteo4_dim(self):
        return Snapshot.meteo_dims * 4
        
    def get_meteo4_tempo_dense_catemb_dim(self):
        return Snapshot.meteo_dims * 4 + Snapshot.tempo_dims + Snapshot.dense_dims + self.category_emb_dim + self.cascade_depth_emb_dim

    def forward(self, x_mtdc, py_mtdc, cascase_depth=1):   # (H, W, ...) => (H, W, meteo_dim)

        category_emb = self.catetory_embedder()
        cascade_emb = self.cascade_depth_embeder(cascase_depth)

        #----------- patch_merging ------------
        if x_mtdc is py_mtdc:     # or if id(x) == id(py)       
            meteo_mormal, _, tempo, dense, _ = x_mtdc     # all normalized, except cat
            # (H, W, meteo_dim), (H, W, meteo_dim), (H//2, W//2, tempo_dim), (H//2, W//2, dense_dim), (H//2, W//2, cat_dim)
            H, W, C = meteo_mormal.shape
            meteo_mormal_4 = self.patch_merger(meteo_mormal.unsqueeze(0)).squeeze(0)    # (H, W, d) => (H//2, W//2, 4*d)
            m4tdc = torch.concat([meteo_mormal_4, tempo, dense, category_emb, cascade_emb], dim=-1).unsqueeze(0)   # 6*4 + 9 + cat_emb.dim
            x = self.m4tdc_to_attn_dims(m4tdc)
            py = x
        else:
            meteo_mormal, _, tempo, dense, cat = py_mtdc     # all normalized, except cat
            # (H, W, meteo_dim), (H, W, meteo_dim), (H//2, W//2, tempo_dim), (H//2, W//2, dense_dim), (H//2, W//2, cat_dim)
            H, W, C = meteo_mormal.shape
            meteo_mormal_4 = self.patch_merger(meteo_mormal.unsqueeze(0)).squeeze(0)    # (H, W, d) => (H//2, W//2, 4*d)
            m4tdc = torch.concat([meteo_mormal_4, tempo, dense, category_emb, cascade_emb], dim=-1).unsqueeze(0)
            py = self.m4tdc_to_attn_dims(m4tdc)
                                                                        
        #---------- attentions -----------------    
        # for i, attn in enumerate(self.attn_layers):
        #     py = attn(x, py)
        #     x = py        

        x = self.uNet(x)

        py_meteo4 = self.attn_dim_to_meteo4(x)

        #----------- patch_splitting ------------
        py = self.patch_splitter(py_meteo4).squeeze(0)      # (H, W, meteo_dim)
        
        # return
        return py
    
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

        # varwise_tensors = []
        # for idx, var in enumerate(variables):
        #     if var in ["t2m", "v100m"]:         # OBSERVATION: predict(x) vs. x is better for "t2m" and "v100m" only.
        #         dt_x = ss_x.get_datetime()
        #         var_tensor = self.predict_fun_dict[var].predict(ss_x.mr, ss_x.dense, ss_x.cat, ss_x.tempo, ss_y.tempo, dt_x, delta_hours=modulo).to(device=self.device, dtype=self.pm.torch_dtype)
        #     else:
        #         var_tensor = ss_x.mr[idx]

        #     varwise_tensors.append(var_tensor)
        
        # predict = torch.stack(varwise_tensors, dim=0)
        
        predict = ss_x.mr

        return predict
    
    # def loss(self, target, pred, batch_diff, zeus_weights):
    def loss(self, target, pred, batch_diff=None, zeus_weights=None, vars_to_mask_on_loss=None, area_weights=None):
        # (C,H,W), (C,H,W), (C, H, W), (C,)
        # 32, cuda, False ; 32, cuda, True !!!, 32, cuda, False, 32, cuda, False
        device = pred.device
        target = target.to(device)
        # vars_to_mask_on_loss = vars_to_mask_on_loss.to(device)
        
        C, H, W = target.shape
        error_sqaured = (pred - target) ** 2 * (area_weights[None, :, None] if area_weights is not None else 1.0)   # (C, H, W)
        sum_weights = area_weights.sum() * W  if area_weights is not None else H * W
        rmse = torch.sqrt(error_sqaured.sum(dim=(1, 2)) / sum_weights)  # (C,)
        # rmse = rmse * (vars_to_mask_on_loss if vars_to_mask_on_loss is not None else 1.0)     # (C,)
        loss = (rmse * (zeus_weights if zeus_weights is not None else 1.0)).mean()

        # self._print_tensor_state(1, target)                 # 
        # self._print_tensor_state(2, pred)
        # self._print_tensor_state(3, batch_diff)
        # self._print_tensor_state(4, zeus_weights)
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
        
        # zeus_weights = zeus_weights.to(device)
        # reward = rmse * zeus_weights   # (C,), 32, cuda, True !!!, the-higher-the-worse
        # self._print_tensor_state(12, reward)
        # reward = reward.mean()  # (,)

        # return rmse.mean(), rmse # (,), (C,)     32, cuda, True

        # zeus_weights = zeus_weights.to(device)
        
        return loss, rmse    
        
    def forward_backward_no_optimizer_step(self,       
            DnSS0, DnSS1, DnSSY, batch_diff, cascade_depth, trainable, scaler, predicted_snapshot_required,
            zeus_weights, train_dict,
        ):
        
        assert DnSS0 == DnSS1   # for this task.
        device = next(self.cascade_depth_embeder.parameters()).device        
        
        # # 1. get physical prediction of y_mr                
        # # used: DnSS0.mr, DnSS0.dense, DnSS0.cat, DnSS0.tempo, DnSSY.tempo, DnSS0.datatime
        # physical_prediction_raw = self.physical_predict(DnSS0, DnSSY, self.pm.modulo, self.pm.meteo_variables)   # 

        # # 2. prediction_normal
        # physical_prediction_normal = physical_prediction_raw.clone()
        # # physical_prediction_normal = train_dict['fn_normalize_meteo_phdn'](physical_prediction_raw, physical_prediction_normal)    
        # physical_prediction_normal = train_dict['fn_normalize_meteo'](physical_prediction_raw, physical_prediction_normal)        
        
        # # 3. Enrich prediction_normal with geo-info
        # PhDn = Snapshot(DnSSY.year, DnSSY.month, DnSSY.day, DnSSY.hour, meteo_normal=physical_prediction_normal, mr=None)
        # mntd_p = Snapshot.pack_pair_snapshots(PhDn, PhDn, device=device)
        # # mntd_p = self.enrich_mixer(mntd_p)

        # 4. get mntd_x                                                   #-------------  mntd = meteo_normal_tempo_dense
        # mntdc_x = Snapshot.pack_pair_snapshots(DnSS0, DnSS0, device=device)
        # mntdc_x = mntdc_x.permute(1, 2, 0).unsqueeze(0)   # from CHW to BHWC

        meteo_mormal, meteo_raw, tempo, dense, cat = DnSS0.meteo_normal, DnSS0.meteo_raw, DnSS0.tempo, DnSS0.dense, DnSS0.cat
        # (meteo_dim, H, W), (meteo_dim, H, W), (tempo_dim, H//2, W//2), (dense_dim, H//2, W//2), (cat_dim, H//2, W//2)

        #====================== from CHW to HWC
        # Remember they are not contiguouse after this line. Don't call .view() on them, .but reshape().
        x_mtdc = (meteo_mormal.permute(1, 2, 0), meteo_raw.permute(1, 2, 0), tempo.permute(1, 2, 0), dense.permute(1, 2, 0), cat.permute(1, 2, 0))
        # (H, W, meteo_dim), (H, W, meteo_dim), (H//2, W//2, tempo_dim), (H//2, W//2, dense_dim), (H//2, W//2, cat_dim)
        
        py_mtdc = x_mtdc
        
        pred_mn = self(x_mtdc, py_mtdc, cascade_depth)        #------------ forward --------- HWC space. forward: (H, W, _) => (H, W, m)
        
        #======================= from BHWC to CHW
        pred_mn = pred_mn.permute(2, 0, 1).squeeze(0)
        
        #--------------------------------------------------------------------------------------------------------
        
        truth_mn = DnSSY.meteo_normal
                
        lossAndMetrics = self.loss(truth_mn, pred_mn, batch_diff, zeus_weights=zeus_weights, area_weights=train_dict['area_weights'])        
        normalized_loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics   #---- [0] picks loss

        if trainable:
            # normalized_loss.backward(retain_graph=True) # instead of the below line.
            scaler.scale(normalized_loss).backward(retain_graph=True) # scaler doesn't support bf16

        pred_mn = pred_mn.detach()
            
        #------------------------------ pred_normal is no longer user below -------------------
        pred_mr = pred_mn.clone()
        train_dict['fn_rawize_meteo'](pred_mn, pred_mr)

        if predicted_snapshot_required:
            predicted_ss = Snapshot(DnSSY.year, DnSSY.month, DnSSY.day, DnSSY.hour, meteo_normal=pred_mn, meteo_raw=pred_mr)
        else:
            predicted_ss = None

        if train_dict['type_metrics'] == "rmse":
            truth_mr = DnSSY.meteo_raw
            lossAndMetrics = self.loss(truth_mr, pred_mr, batch_diff, area_weights=train_dict['area_weights'])
            rawized_metrics = lossAndMetrics[1] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics   #---- [1] picks rmse
        elif train_dict['type_metrics'] == "mae":
            rawized_metrics = torch.abs(truth_mr - pred_mr).mean(dim=(1, 2))    # (C,)
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
    
    factor10 = 2.5
    num_heads = 2
    B = 1

    H, W = int(721*factor10/10), int(1440*factor10/10)
    C = 10 * int(10/factor10 * 10/factor10)
    
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
