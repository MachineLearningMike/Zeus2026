import torch.nn as nn
from SphericalCodec import SphericalEncoder, SphericalDecoder

# modules.ResNet_custom_1 is 0.131:0.090x faster, giving the same accuracy.
# from modules.ResNet50_Zeus_GroupNorm_365K import ResNetCylindrical
# from modules.ResNet50_Zeus_GroupNorm_24M import ResNetCylindrical
# from modules.ResNet50_Zeus_noStage_GroupNorm_xM import ResNetCylindrical
# from modules.ResNet50_Zeus_noStage_GroupNorm_365K import ResNetCylindrical
# from modules.ResNet50_Zeus_GroupNorm import ResNetCylindrical
from ResNet_CircularBottleneckStack import CircularBottleneckStack

# -----------------------------
# ZeusCNN model
# -----------------------------
class Zeus(nn.Module):
    def __init__(self, pm, device='cpu'):
        """
        Past n frames are flattened into channels (n*C) and fed once to encoder + CNN.
        """       
        super().__init__()
        self.pm = pm

        C, H, W, self.n_frames = len(pm.variables), pm.height, pm.width, pm.nFramesIn
        self.device = device

        # Encoder/Decoder always work per-frame
        self.encoder = SphericalEncoder(C, H, W, pm.ObsorbHDistortion, near_pole_h_lines=self.pm.near_pole_h_lines).to(device)
        self.decoder = SphericalDecoder(C, C, H, W, pm.ObsorbHDistortion, near_pole_h_lines=self.pm.near_pole_h_lines).to(device)

        in_channels = self.n_frames * C    # stacked frames
        # self.cnn = Convolutions(in_channels, cnn_hidden_channels, kernel_size=kernel_size).to(device)
        print("ZeusResNet being created...")
        # self.cnn = ResNetCylindrical(nVariables=C, nBlocks=36, mid_channels=in_channels, in_channels=in_channels).to(device)    # 3753K params, from ResNet50_Zeus_GroupNorm_3753K
        # self.cnn = ResNetCylindrical(in_channels=pm.T1*C).to(device)  # 365K params, from ResNet50_Zeus_GroupNorm_365K
        # self.cnn = ResNetCylindrical(self.pm).to(device)
        self.cnn = CircularBottleneckStack(channels=pm.T1 * pm.nVariables, num_blocks=pm.nCNNBlocks, bottlenectRate=pm.bottleneckRate, dropout=pm.bottleneck_dropout).to(device)

        num_params = sum(p.numel() for p in self.cnn.parameters())
        print(f"Total parameters: {num_params:,}")

    def forward(self, x, m_frames, y=None):  # y_seq != None signals training
        B, n_frames, C, H, W = x.shape
        assert n_frames == self.n_frames, f"Expected {self.n_frames} input frames, got {n_frames}"

        x = x.to(self.device)
        if y is not None:
            assert m_frames == y.shape[1]
            y = y.to(self.device)

        x_enc = self.encoder(x.view(B * n_frames, C, H, W)).view(B, n_frames, C, H, W)

        preds = []

        # Initial window of encoded frames
        x_window = x_enc.reshape(B, n_frames*C, H, W)

        # This block is replaced by its following line.
        # for t in range(m_frames):
        #     y_pred = self.cnn(x_window)     # From (B, n_frames * C, H, W) to (B, C, H, W)  =========== # This is why the last of cnn_hidden_channels should C.
        #     y_pred = self.decoder(y_pred)
        #     preds.append(y_pred)
        #     if t < m_frames - 1:
        #         if y is not None and random.random() > self.sampling_prob:  # self.sampling_prob rises as steps go, giving less chance to y.
        #             y_next = self.encoder(y[:, t])
        #         else:
        #             y_next = y_pred      
        #         x_window = torch.cat([x_window[:, C:], y_next], dim=1)        
        # preds = torch.stack(preds, dim=1)   # (B, m_frames, C, H, W)
        
        # This line replaces the above commented-out block. Faster. Larger loss reduction per hour, though smaller loss reduction per epoch.
        pred = self.cnn(x_window).view(B, n_frames, C, H, W)   # self.cnn: (B, n_frames*C, H, W) -> (B, n_frames*C, H, W)
        
        preds = self.decoder(pred.view(B * m_frames, C, H, W)).view(B, m_frames, C, H, W)

        return preds

# Demonstration
if __name__ == "__main__":
    
    import torch
    from config_copy import config
    from params_copy import Params, flatten_pm_single
    
    pm = Params(config)
    model_id = 0
    pm_with_model = flatten_pm_single(pm, model_id)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = Zeus(pm_with_model, device)
    
    B = pm_with_model.batch_size
    C = pm_with_model.nVariables
    H, W = pm_with_model.height, pm.width
    n_frames = pm_with_model.nFrames
    m_frames = pm_with_model.mFrames

    # Fake input: (B, n_frames, C, H, W)
    raw_x = torch.randn(B, n_frames, C, H, W)
    normal_x = raw_x # normalize(raw_x)
    
    if pm.near_pole_h_lines > 0:
        trimmed_x  = raw_x[:, :, :, pm.near_pole_h_lines:-pm.near_pole_h_lines, :]
        
    print("Input shape:", trimmed_x.shape)

    print("\nRunning forward pass...")
    y_pred = model(trimmed_x, m_frames)
    
    unnormal_y_pred = y_pred # unnormalize(y_pred)

    print("Output shape:", y_pred.shape)
    assert trimmed_x.shape == y_pred.shape, "Output shape differs from input shape"
    
    
    # checkpoint_path = "d:/downloads/best_model_0.pth"
    checkpoint_path = "/path/to/best_model_0.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    # print("Model loaded with saved state: \n_missing_keys: {}, \n_unexpected_keys: {}".format(len(missing_keys), len(unexpected_keys)))
    # assert len(missing_keys) == 0 and len(unexpected_keys) == 0, "parameter mismatch"
                                          
    # y_pred = model(trimmed_x, m_frames)
    # print("Output shape:", y_pred.shape)
    
    