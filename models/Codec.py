# -----------------------------
# Encoder & Decoder
# -----------------------------

class CnnPair(nn.Module):
    def __init__(self, h_index, channels, W):  # W: input width
        super().__init__()
        alpha = h_index / 720 * 180 - 90
        alpha = math.cos(alpha * math.pi / 180) + 1e-9

        self.floor = min(W, max(1, math.floor(alpha)))
        self.ceil  = min(W, max(1, math.ceil(alpha)))
        self.floorW = self.ceil - alpha
        self.ceilW  = alpha - self.floor

        self.floorCnn = nn.Conv1d(channels, channels, self.floor, stride=1, padding=0, bias=True)
        self.ceilCnn  = nn.Conv1d(channels, channels, self.ceil, stride=1, padding=0, bias=True)

    def forward(self, x):   # (B, C, W)
        pad = self.floor // 2
        xFloor = F.pad(x, (pad, pad), mode="circular")
        xFloor = self.floorCnn(xFloor)

        pad = self.ceil // 2
        xCeil = F.pad(x, (pad, pad), mode="circular")
        xCeil = self.ceilCnn(xCeil)

        x = self.floorW * xFloor + self.ceilW * xCeil
        return x
        
class Encoder(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.cnnPairs = nn.ModuleList([CnnPair(h_index, C, W) for h_index in range(H)])

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        slices = [cnnPair(x[:, :, h_index, :]) for h_index, cnnPair in enumerate(self.cnnPairs)]    # [(B, C, W)]
        x = torch.stack(slices, dim=2)  # (B, C, H, W)
        return x

class Decoder(Encoder):
    def __init__(self, in_C, C, H, W):
        assert in_C == C
        super().__init__(C, H, W)

# -----------------------------
# Convolutions (circular width)
# -----------------------------
class Convolutions(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, kernel_size=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.kernel_sizes = []
        prev_channels = in_channels
        for out_channels in hidden_channels_list:
            self.convs.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=kernel_size, padding=0)
            )
            self.kernel_sizes.append(kernel_size)
            prev_channels = out_channels

    def forward(self, x):
        for conv, k in zip(self.convs, self.kernel_sizes):
            pad_h = k // 2
            pad_w = k // 2

            # circular pad along width (dim=3)
            if pad_w > 0:
                x = torch.cat([x[:, :, :, -pad_w:], x, x[:, :, :, :pad_w]], dim=3)

            # zero pad along height (dim=2)
            if pad_h > 0:
                x = F.pad(x, (0, 0, pad_h, pad_h), mode='constant', value=0)

            x = conv(x)
            x = F.relu(x, inplace=True)
        return x