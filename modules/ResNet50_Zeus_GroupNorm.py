import torch
import torch.nn as nn

# ---------- Helper ----------
def get_valid_groups(num_channels, max_groups=32):
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1


# ---------- Bottleneck Block ----------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, num_groups=32, circular=False, downsample=None):
        super().__init__()
        print("in, mid", in_channels, mid_channels)
        
        padding_mode = 'circular' if circular else 'zeros'

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.gn1 = nn.GroupNorm(get_valid_groups(mid_channels, num_groups), mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1,
                               bias=False, padding_mode=padding_mode)
        self.gn2 = nn.GroupNorm(get_valid_groups(mid_channels, num_groups), mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, 1, bias=False)
        self.gn3 = nn.GroupNorm(get_valid_groups(mid_channels * self.expansion, num_groups),
                                mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------- ResNet-50 Core ----------
class ResNetCylindrical(nn.Module):
    def __init__(self, pm):
        super().__init__()
        blocksPerState = pm.blocksPerState
        num_groups = pm.nVariables
        in_channels = pm.T1 * pm.nVariables
        
        self.inplanes = in_channels # This is the in_channels for the ResNet50's initial stage. It changes in stage.
        self.circular = True    # This is for Zeus application.

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False, padding_mode='circular' if self.circular else 'zeros')
        self.gn1 = nn.GroupNorm(get_valid_groups(self.inplanes, num_groups), self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Stages
        self.stages = [self._make_stage(Bottleneck, mid_channels = 2 ** (stage + 6), nBlocks = blocksPerState[stage], stride=1, num_groups=num_groups) for stage in range(len(blocksPerState))]
        self.stages = nn.Sequential(*self.stages)

        self.head = nn.Conv2d(2 ** (len(blocksPerState) - 1 + 6) * Bottleneck.expansion, in_channels, kernel_size=1, bias=True)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_stage(self, block, mid_channels, nBlocks, stride, num_groups):
        downsample = None
        if stride != 1 or self.inplanes != mid_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, mid_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(get_valid_groups(mid_channels * block.expansion, num_groups),
                             mid_channels * block.expansion),
            )
        blocks = []
        blocks.append(block(self.inplanes, mid_channels, stride, num_groups,
                            circular=self.circular, downsample=downsample))
        self.inplanes = mid_channels * block.expansion
        for _ in range(1, nBlocks):
            blocks.append(block(self.inplanes, mid_channels, num_groups=num_groups,
                                circular=self.circular))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)
        for stages in self.stages:
            x = stages(x)        
        x = self.head(x)
        return x

# ---------- Demo ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nVariables = 4
    in_channels = 24 * nVariables  # 96
    x = torch.randn(2, in_channels, 72*3, 144*3, device=device)
    
    class Params:
        def __init__(self):
            self.blocksPerState=[3, 20]
            self.T1 = 24
            self.nVariables = 4
    pm = Params()
    model = ResNetCylindrical(pm).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {num_params:,}")

    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(model, x)
    # print("✅ FLOPs:", flops.total())
    # print("✅ FLOPs (formatted):", flops.total()/1e9, "GFLOPs")

    import time
    loops = 2
    start = time.time()
    with torch.no_grad():
        for _ in range(loops):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
                y = model(x)
    end = time.time()

    print("✅ Input shape:", x.shape)
    print("✅ Output shape:", y.shape)
    print("Average forward time per step: {:.3f} s".format((end-start)/loops))
    
    