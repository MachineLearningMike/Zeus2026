import torch
import torch.nn as nn
from datetime import datetime, timedelta, timezone
import time

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.zeus_patch_unpatch_individual import PatchFeatureExtractor, UnpatchReconstructor
from modules.Zeus_Swin_Block_3D_CDHW import Encoder_Decoder
from modules.zeus_dataset_Snapshot import Snapshot


# -----------------------
# 4️⃣ Full Model
# -----------------------
class Zeus(nn.Module):
    def __init__(self, pm, device='cpu'):
        super().__init__()

        self.factor10 = pm.factor10

        factor10 = pm.factor10
        num_heads = pm.nHeads
        patch_kernel = pm.patch_kernel
        patch_stride = pm.patch_stride
        swin_dim = pm.swin_dim
        window_size = pm.window_size
        num_encoder_decoder = pm.num_swin_block
        nLevels = pm.nLevels
        increasing = pm.increasing
        type_pos_bias = pm.type_pos_bias
        depth_list = pm.depth_list
        head_list = pm.head_list
        skippable = pm.skippable
        skip_learnable = pm.skip_learnable
        qkv_bias = pm.qkv_bias
        
        pair_bias_dim = pm.pair_bias_dim

        H, W = int(720/10 * factor10), int(1440/10 * factor10)
    
        print("Zeus model being created..............................................................")
        print(f"factor10 {factor10}, num_heads {num_heads}, patch_kernel {patch_kernel}, patch_stride {patch_stride}, swin_dim {swin_dim}")
        print(f"window_size {window_size}, num_encoder_decoder {num_encoder_decoder}, nLevels {nLevels}, increasing {increasing}, type_pos_bias {type_pos_bias}")
        print(f"depth_list {depth_list}, head_list {head_list}, skippable {skippable}, skip_learnable {skip_learnable}")
        print(".......................................................................................")

        self.device = device
    
        #---------------- Patch extractor -----------------
        
        Snapshot.set_original_settings(factor10=factor10, increasing=increasing, packed_levels=nLevels - 1)  #------------- SETUP
        
        now = datetime.now(timezone.utc)
        ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
        
        if pm.pair_not_single_snapshot:
            then = now + timedelta(hours=6)
            ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo=None, meteo_unnormal=None)
        else:
            ss2 = ss1

        levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
        
        self.patch_kernel=(patch_kernel,patch_kernel)
        self.patch_stride=(patch_stride,patch_stride)
        
        self.patch_extractor = PatchFeatureExtractor(
                levels, surface, 
                c_out=swin_dim, 
                patch_kernel=self.patch_kernel, 
                patch_stride=self.patch_stride
            )
        
        #---------------- Swin blocks ----------------------
        
        input_resolution = (self.patch_extractor.nTotalLevels, H//patch_stride, W//patch_stride)
        qkv_bias = qkv_bias
        use_checkpoint = False
        self.encoder_decoder = Encoder_Decoder(
                input_resolution = input_resolution, embed_dim=swin_dim, depths=depth_list, num_heads=head_list,    #--------- num_heads to revise for deployment.
                window_size=window_size, mlp_ratio=4., qkv_bias=qkv_bias,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True,
                use_checkpoint=use_checkpoint, pretrained_window_sizes=[0, 0, 0, 0], skippable=skippable, skip_learnable=skip_learnable)
                
        #--------------- Unpatch reconstructor        
        self.unpatch_reconstructor = UnpatchReconstructor(
                in_channels=self.patch_extractor.nTotalLevels * swin_dim, 
                # We have to reduce losses for all the meteo variables, for auto-regressive (cascaded) inference.
                # Consider placing heavier weights on our target variables in the loss function.
                out_channels=Snapshot.meteo_dims,   
                kernel_size=self.patch_kernel, 
                stride=self.patch_stride
            )

    def forward(self, ss1, ss2, cascade_step):    # Not batch-processing, as hard drives are always bottlenecks.       
        # print(f"x befor patching: {ss1.meteo.shape, ss1.tempo.shape, ss1.geo_dense.shape}")

        #----------- patch/extract level features ------------
        # s = time.time()
        levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
        x = self.patch_extractor(levels, surface, cascade_step)
        # e = time.time()
        # print(f"x after patching: {x.shape}, time: {e-s}")

        #----------- undergo swin blocks ----------------
        # s = time.time()
        x = x.unsqueeze(0)
        B, C, D, H, W = x.shape
        x_flat = x.permute(0,2,3,4,1).reshape(B, D*H*W, C)

        x_flat = self.encoder_decoder(x_flat)

        x_out = x_flat.view(B, D, H, W, C).permute(0,4,1,2,3).contiguous()
        x = x_out.squeeze(0)
        # e = time.time()
        # print(f"x after swin blocks: {x.shape}, time: {e-s}")
        
        #------------ unpatch/reconstruct --------------

        # s = time.time()
        x = x.view(-1, H, W)
        x = x.unsqueeze(0)
        x = self.unpatch_reconstructor(x)
        x = x.squeeze(0)
        # e = time.time()
        # print(f"x after unpatching: {x.shape}, time: {e-s}")

        return x
    
    def forward_backward_no_step(self,
            ss0, ss1, ssY, cascade_step,
            fn_loss, fn_nornalize, fn_unnornalize, 
            batch_diff, era5_data_vars, trainable,
            type_metrics, continued,
            pressure_indices, pressure_values,
        ):
        
        # with autocast(device_type=self.device_type, dtype=torch.bfloat16):
        #     with torch.set_grad_enabled(trainable):         
        
        pred = self(ss0, ss1, cascade_step)
        pred[pressure_indices, :, :] = pressure_values.view(-1, 1, 1).to(pred.dtype)

        # pred = torch.index_select(pred, dim=0, index=target_indices)

        if continued:
            result_ss = Snapshot(ssY.year, ssY.month, ssY.day, ssY.hour, meteo=pred.detach().to(ssY.meteo.device), meteo_unnormal=None, device=ssY.device)
        else:
            result_ss = None

        pred_unnormal = pred.clone()
        fn_unnornalize(pred, pred_unnormal)

        target_normal = ssY.meteo
        target_unnormal = ssY.meteo_unnormal
        
        lossAndMetrics = fn_loss(target_normal, pred, batch_diff, era5_data_vars)
        loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics

        if trainable:
            loss.backward(retain_graph=True)
                # grads = [p.grad.detach().clone() for p in self.parameters()]
        
        if type_metrics == "rmse":
            lossAndMetrics = fn_loss(target_unnormal, pred_unnormal, batch_diff, era5_data_vars)
            unnormalized_metrics = lossAndMetrics[1] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
        elif type_metrics == "mae":
            unnormalized_metrics = torch.abs(target_unnormal - pred_unnormal).mean(dim=(1, 2))    # (C,)
        else:
            raise Exception("type_metrics is invalid")
                
        return result_ss, loss.detach().to("cpu"), unnormalized_metrics.detach().to("cpu")

# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    from models.config_Mike1 import config
    from modules.params_Mike1 import Params, flatten_pm_single
    pm = Params(config)    
    flatten_pm = flatten_pm_single(pm, 0)
        
    model = Zeus(flatten_pm, device=device)
    model.to(device)

    Snapshot.device = device
    Snapshot.export_factor10 = model.factor10
    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None, device=device)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo=None, meteo_unnormal=None, device=device)
    
    s = time.time()
    x_out = model(ss1, ss2, 1)
    e = time.time()
    
    print(x_out.shape, "time: ", e-s)
    export_size = Snapshot.get_export_size()
    assert x_out.shape == (Snapshot.meteo_dims, export_size[0], export_size[1])
    
