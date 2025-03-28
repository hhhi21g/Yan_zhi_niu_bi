import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import iSTFT
import torch.nn.functional as F


class WaveformNet(nn.Module):
    def __init__(self):
        super(WaveformNet, self).__init__()

        self.encoder = nn.ModuleList([
            self.encoder_block_1d(1, 16, 128),
            self.encoder_block_1d(16, 32, 128),
            self.encoder_block_1d(32, 64, 128),
            self.encoder_block_1d(64, 64, 128),
            self.encoder_block_1d(64, 128, 128),
            self.encoder_block_1d(128, 128, 128)
        ])

        self.decoder = nn.ModuleList([
            self.decoder_block_1d(128, 128, 128),
            self.decoder_block_1d(256, 64, 128),
            self.decoder_block_1d(128, 64, 128),
            self.decoder_block_1d(128, 32, 128),
            self.decoder_block_1d(64, 16, 128),
            self.decoder_block_1d(32, 1, 128)
        ])

        self.final_conv = nn.Conv1d(1, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def encoder_block_1d(self, in_channels, out_channels, kernel_size):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.PReLU(),
            nn.Dropout(0.5)
        )

    def decoder_block_1d(self, in_channels, out_channels, kernel_size):
        padding = (kernel_size - 1) // 2
        # output_padding = 1 if kernel_size % 2 == 0 else 0
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=padding,
                               output_padding=0),
            nn.InstanceNorm1d(out_channels),
            nn.PReLU(),
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
                if x.shape[-1:] != skip.shape[-1:]:
                    x = nn.functional.interpolate(x, size=skip.shape[-1:], mode='nearest')
                x = torch.cat([x, skip], dim=1)

        x = self.final_conv(x)
        return self.sigmoid(x)


model = WaveformNet()

dummy_input = torch.randn(16, 1, 512)  # 典型的心音频谱尺寸
output = model(dummy_input)

target = torch.rand_like(output)
loss = nn.L1Loss()(output, target)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print(f"输入尺寸: {dummy_input.shape}")
print(f"输出尺寸: {output.shape}")
print(f"L1损失计算验证: {loss.item():.4f}")
