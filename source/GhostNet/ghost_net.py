import torch
import torch.nn as nn

class GhostNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),  # Change the input channel to 1 (grayscale)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(16, 16, 1, stride=1)
        self.stage2 = self._make_stage(16, 24, 2, stride=2)
        self.stage3 = self._make_stage(24, 40, 3, stride=2, use_se=True)  # Using SE layer 
        self.stage4 = self._make_stage(40, 80, 3, stride=2, use_se=True)
        self.stage5 = self._make_stage(80, 96, 2, stride=1, use_se=True)
        self.stage6 = self._make_stage(96, 192, 4, stride=2, use_se=True)
        self.stage7 = self._make_stage(192, 320, 1, stride=1, use_se=True)

        self.conv9 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

    def _make_stage(self, input_channels, output_channels, num_blocks, stride, use_se=False):  # Thêm tham số use_se
        layers = []
        layers.append(GhostBottleneck(input_channels, input_channels // 2, output_channels, 3, stride, use_se))  # Sử dụng SE trong GhostBottleneck
        for _ in range(1, num_blocks):
            layers.append(GhostBottleneck(output_channels, output_channels // 2, output_channels, 3, 1, use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.conv9(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
