import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=8, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=8, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Module):
    """ Conv2d -> BN -> ReLU ->Dropout"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """ Conv1d -> BN -> PReLU ->Dropout"""

    def __init__(self, in_channels, out_channels, mid_channels=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.up_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(mid_channels),
            nn.PReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x1, x2):
        """
        param x1: 上一层的特征
        param x2: 跳跃连接的特征
        """
        x1 = self.up(x1)

        diff = x2.size(-1) - x1.size(-1)
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x2 = F.pad(x2, [-diff // 2, -diff - (-diff) // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.up_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
