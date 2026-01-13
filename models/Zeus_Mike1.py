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
        self.pm = pm

        self.factor10 = pm.factor10

        factor10 = pm.factor10
        num_heads = pm.nHeads
        patch_kernel = pm.patch_kernel
        patch_stride = pm.patch_stride
        swin_dim = pm.swin_dim
        sWin = pm.sWin
        num_encoder_decoder = pm.num_swin_block
        nLevels = pm.nLevels
        increasing = pm.increasing
        type_pos_bias = pm.type_pos_bias
        depth_list = pm.depth_list
        head_list = pm.head_list
        qkv_bias = pm.qkv_bias
        skippable = pm.skippable
        skip_learnable = pm.skip_learnable
        
        nRawStepsPerBatch = pm.nRawStepsPerBatch
        choose_best_loss_model = pm.choose_best_loss_model
        
        
        pair_bias_dim = pm.pair_bias_dim
        

        H, W = int(720/10 * factor10), int(1440/10 * factor10)
    
        print("Zeus model being created..............................................................")
        print(f"factor10 {factor10}, num_heads {num_heads}, patch_kernel {patch_kernel}, patch_stride {patch_stride}, swin_dim {swin_dim}")
        print(f"sWin {sWin}, num_encoder_decoder {num_encoder_decoder}, nLevels {nLevels}, increasing {increasing}, type_pos_bias {type_pos_bias}")
        print(f"depth_list {depth_list}, head_list {head_list}, qkv_bias {qkv_bias}, skippable {skippable}, skip_learnable {skip_learnable}")
        print(f"nRawStepsPerBatch {nRawStepsPerBatch}, choose_best_loss_model {choose_best_loss_model}")
        print(".......................................................................................")

        self.device = device
    
        #---------------- Patch extractor -----------------
        
        Snapshot.set_original_settings(factor10=factor10, increasing=increasing, packed_levels=nLevels - 1, device=self.device)  # Setup that will be persistent.
        
        now = datetime.now(timezone.utc)
        ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
        
        if pm.pair_not_single_snapshot:
            then = now + timedelta(hours=6)
            ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo=None, meteo_unnormal=None)
        else:
            ss2 = ss1

        levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2, device=device)
        
        self.patch_kernel=(patch_kernel,patch_kernel)
        self.patch_stride=(patch_stride,patch_stride)
        
        self.patch_extractor = PatchFeatureExtractor(
                levels, surface, 
                c_out=swin_dim, 
                patch_kernel=self.patch_kernel, 
                patch_stride=self.patch_stride
            )
        
        #---------------- Swin blocks ----------------------
        
        i_reso = (self.patch_extractor.nTotalLevels, H//patch_stride, W//patch_stride)
        
        use_checkpoint = self.pm.use_checkpoint #------------------------------- True for enhanced basiclayer on A100
        
        self.encoder_decoder = Encoder_Decoder(
                i_reso = i_reso, embed_dim=swin_dim, depths=depth_list, num_heads=head_list,    #--------- num_heads to revise for deployment.
                sWin=sWin, mlp_ratio=4., qkv_bias=qkv_bias,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True,
                use_checkpoint=use_checkpoint, pretrained_sWins=[0, 0, 0, 0], skippable=skippable, skip_learnable=skip_learnable)
                
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
        device = next(self.patch_extractor.parameters()).device
        levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2, device=device)
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
    
    # def loss(self, target, pred, batch_diff, zeus_loss_weights_on_target_vars):
    def loss(self, target, pred, batch_diff, zeus_loss_weights_on_target_vars, vars_to_mask_on_loss=None, area_weights=None):
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
    
    def forward_backward_no_optimizer_step(self, train_dict, ss0, ss1, ssY, cascade_step, batch_diff, trainable, scaler, predicted_snapshot_required):
        
        air_level_pressure_indices = train_dict['air_level_pressure_indices']
        zeus_loss_weights_on_target_vars = train_dict['zeus_loss_weights_on_target_vars']
        fn_unnormalize_metro = train_dict['fn_unnormalize_metro']
        type_metrics = train_dict['type_metrics']
        area_weights = train_dict['area_weights']

        pred_normal = self(ss0, ss1, cascade_step)
        
        air_levels_mask = torch.ones(pred_normal.shape[0], dtype=torch.float32, device='cpu')
        air_levels_mask[air_level_pressure_indices] = 0.0
        # air_levels_mask[-len(self.pm.targetVariables):] = self.pm.lossXtargets  # Weight amplifiled -------------------------
        
        lossAndMetrics = self.loss(ssY.meteo, pred_normal, batch_diff, zeus_loss_weights_on_target_vars, vars_to_mask_on_loss=air_levels_mask, area_weights=area_weights)
        normalized_loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics   #---- [0] picks loss

        if trainable:
            scaler.scale(normalized_loss).backward(retain_graph=True)

        pred_normal = pred_normal.detach()
        pred_normal[air_level_pressure_indices] = ssY.meteo[air_level_pressure_indices].to(pred_normal.device).to(pred_normal.dtype)

        if predicted_snapshot_required:
            result_ss = Snapshot(ssY.year, ssY.month, ssY.day, ssY.hour, meteo=pred_normal.to(ssY.meteo.device), meteo_unnormal=None)
        else:
            result_ss = None
            
        #------------------------------ pred_normal is no longer user below -------------------
        pred_unnormal = pred_normal
        fn_unnormalize_metro(pred_normal, pred_unnormal)
        
        if type_metrics == "rmse":
            lossAndMetrics = self.loss(ssY.meteo_unnormal, pred_unnormal, batch_diff, zeus_loss_weights_on_target_vars=None, vars_to_mask_on_loss=air_levels_mask, area_weights=area_weights)
            unnormalized_metrics = lossAndMetrics[1] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics   #---- [1] picks rmse
        elif type_metrics == "mae":
            unnormalized_metrics = torch.abs(ssY.meteo_unnormal - pred_unnormal).mean(dim=(1, 2))    # (C,)
        else:
            raise Exception("type_metrics is invalid")
                
        return result_ss, normalized_loss.detach().to("cpu"), unnormalized_metrics.detach().to("cpu")


    def enhance_model_insert_codec_bloks(self, model, layers_to_enhance_in_enc_dec=[2], verbose=True):
        """
        Simple: Enhance bottleneck layers, return modified model.
        """
        if verbose:
            print("Enhancing bottleneck layers...")
        
        basiclayer_counter = 0
        
        for name, module in model.named_modules():
            if module.__class__.__name__ == "BasicLayer":
                print(f"  BasicLayer name: {name}, basiclayer_counter {basiclayer_counter}")
                
                # Enhance every 3rd BasicLayer (bottleneck)
                # NOTE: this dirty code assumes:
                # - both encoder and decoder have 3 basic layers, 
                # - we want to ease symmetric layers.
                
                if basiclayer_counter % 3 in layers_to_enhance_in_enc_dec:
                    if verbose:
                        print(f"  Enhancing bottleneck #{basiclayer_counter}: {name}")
                    
                    module.enhance_layer_irreversible(module, nEnhancementPairs=self.pm.nEnhancementPairs[layers_to_enhance_in_enc_dec.index(basiclayer_counter % 3)])
        
                basiclayer_counter += 1
        
        if verbose:
            print(f"Enhanced bottleneck layers. 2 of {basiclayer_counter}")
        
        return model  # That's it!


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
    
