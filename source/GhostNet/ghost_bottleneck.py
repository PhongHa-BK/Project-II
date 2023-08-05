import torch
import torch.nn as nn

class GhostBottleneck(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size, stride, use_se=True):
        super(GhostBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.ghost = GhostModule(hidden_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, output_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.use_se = use_se

        if self.use_se:
            self.se = SELayer(output_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.ghost(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.stride == 1 and identity.shape[1] == out.shape[1]:
            identity = nn.functional.interpolate(identity, size=out.shape[2:], mode='nearest')
            out += identity

        return out