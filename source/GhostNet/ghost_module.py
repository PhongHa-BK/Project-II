import torch
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, ratio=2):
        super(GhostModule, self).__init__()
        self.primary_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels // ratio, kernel_size, bias=False),
            nn.BatchNorm2d(output_channels // ratio),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(output_channels // ratio, output_channels, kernel_size, groups=output_channels // ratio, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)  # The first GhostModule block
        x2 = self.cheap_operation(x1)  # The second GhostModule block
        out = torch.cat([x1, x2], dim=1)
        return out[:, :x.shape[1], :, :]