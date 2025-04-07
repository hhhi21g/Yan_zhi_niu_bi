import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import iSTFT
import torch.nn.functional as F

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SpectrogramUNet(nn.Module):
    def __init__(self):
        super(SpectrogramUNet, self).__init__()

        # 6个编码层
        self.encoder = nn.ModuleList([
            self.encoder_block(1, 8, 8),
            self.encoder_block(8, 16, 8),
            self.encoder_block(16, 32, 2),
            self.encoder_block(32, 64, 2),
            self.encoder_block(64, 64, 1),
            self.encoder_block(64, 64, 1)
        ])

        self.decoder = nn.ModuleList([
            self.decoder_block(64, 64, 1),
            self.decoder_block(128, 64, 1),
            self.decoder_block(128, 32, 2),
            self.decoder_block(64, 16, 2),
            self.decoder_block(32, 8, 8),
            self.decoder_block(16, 1, 8)
        ])

        self.final_conv = nn.Conv2d(1, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    # 二维卷积, IN代替BN, ReLU, dropout,stride=2
    def encoder_block(self, in_channels, out_channels, kernel_size):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    # 二维转置卷积, IN, ReLU, dropout,stride=2
    def decoder_block(self, in_channels, out_channels, kernel_size):
        padding = (kernel_size - 1) // 2
        # output_padding = 1 if kernel_size % 2 == 0 else 0
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding,
                               output_padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    # 跳跃连接
    def forward(self, x):
        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        for i, layer in enumerate(self.decoder):
            x = layer(x)

            if i < len(self.decoder) - 1:
                skip_idx = len(skip_connections) - 2 - i  # 对应的encoder层
                skip = skip_connections[skip_idx]

                # 尺寸对齐，应对可能的边界效应
                if x.shape[-2:] != skip.shape[-2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='nearest')
                x = torch.cat([x, skip], dim=1)

        x = self.final_conv(x)
        return self.sigmoid(x)




model = SpectrogramUNet()

dummy_input = torch.randn(2, 1, 512, 4096)  # 典型的心音频谱尺寸
output = model(dummy_input)

target = torch.rand_like(output)
loss = nn.L1Loss()(output, target)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print(f"输入尺寸: {dummy_input.shape}")
print(f"输出尺寸: {output.shape}")
print(f"L1损失计算验证: {loss.item():.4f}")

