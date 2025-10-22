import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Bottleneck Block ----------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------- ResNet-50 ----------
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, mid_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != mid_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, mid_channels * Bottleneck.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, mid_channels, stride, downsample))
        self.inplanes = mid_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------- Demo ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet50(num_classes=1000).to(device)

    x = torch.randn(2, 3, 224, 224, device=device)
    y = model(x)

    print("✅ Input shape:", x.shape)
    print("✅ Output shape:", y.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {num_params:,}")

    import time
    loops = 5
    start = time.time()
    with torch.no_grad():
        for _ in range(loops):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
                y = model(x)
    end = time.time()

    print("✅ Input shape:", x.shape)
    print("✅ Output shape:", y.shape)
    print("Average forward time per step: {:.3f} s".format((end-start)/loops))