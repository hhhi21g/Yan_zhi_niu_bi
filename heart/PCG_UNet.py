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
            self.encoder_block(16, 32, 6),
            self.encoder_block(32, 64, 6),
            self.encoder_block(64, 64, 4),
            self.encoder_block(64, 64, 4)
        ])

        self.decoder = nn.ModuleList([
            self.decoder_block(64, 64, 4),
            self.decoder_block(128, 64, 4),
            self.decoder_block(128, 32, 6),
            self.decoder_block(64, 16, 6),
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
                               output_padding=1),
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
            self.decoder_block_1d(128, 64, 128),
            self.decoder_block_1d(64, 64, 128),
            self.decoder_block_1d(64, 32, 128),
            self.decoder_block_1d(32, 16, 128),
            self.decoder_block_1d(16, 1, 128)
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
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 跳跃连接

    def forward_1d(self, x):
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
                    x = nn.functional.interpolate(x, size=skip.shape[-1:], mode='nearest')
                x = torch.cat([x, skip], dim=1)

        x = self.final_conv(x)
        return self.sigmoid(x)


class TwoStageModel(nn.Module):
    def __init__(self, n_fft=2048, hop_length=1024):
        super().__init__()
        self.stft = iSTFT.STFT(filter_length=n_fft, hop_length=hop_length)
        self.spec_unet = SpectrogramUNet()
        self.wave_unet = WaveformNet()

    def forward(self, waveform):
        # Stage 1: 频谱恢复
        mag, phase = self.stft.transform(waveform)  # mag形状: (B, 513, T)
        mag_input = mag.unsqueeze(1)  # (B, 1, 513, T)
        restored_mag = self.spec_unet(mag_input)  # 输出形状: (B, 1, 513, T)

        # 逆STFT生成初步时域信号
        restored_mag = restored_mag.squeeze(1)  # (B, 513, T)
        phase = phase.detach()  # 使用原始相位
        waveform_stage1 = self.stft.inverse(restored_mag, phase)  # (B, N)

        # Stage 2: 波形精炼
        waveform_input = waveform_stage1.unsqueeze(1)  # (B, 1, N)
        waveform_output = self.wave_unet(waveform_input)  # (B, 1, N)
        return waveform_output.squeeze(1), restored_mag


class PCGLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_length=512, alpha=10.0, beta=1.0):
        super().__init__()
        self.stft = iSTFT.STFT(filter_length=n_fft, hop_length=hop_length)
        self.alpha = alpha
        self.beta = beta

        # 初始化500Hz低通滤波器（二阶Butterworth）
        self.lowpass = self._init_butterworth(cutoff=500, fs=48000, order=2)

    def _init_butterworth(self, cutoff, fs, order):
        from scipy.signal import butter, sosfilt
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, btype='low', output='sos')
        return lambda x: torch.from_numpy(sosfilt(sos, x.numpy()))

    def forward(self, pred_wave, target_wave):
        """
        Args:
            pred_wave:   预测波形 [B, T]
            target_wave: 目标波形 [B, T]
        Returns:
            total_loss: 加权总损失
            loss_dict:  各分项损失详情
        """
        # 第一阶段：频谱损失计算
        pred_mag, _ = self.stft.transform(pred_wave)  # [B, F, T]
        target_mag, _ = self.stft.transform(target_wave)
        loss_spec = F.l1_loss(pred_mag, target_mag)

        # 第二阶段：时域损失 + 频谱一致性
        loss_time = F.l1_loss(pred_wave, target_wave)

        # 对预测波形应用低通滤波后重新计算频谱
        filtered_wave = self.lowpass(pred_wave.detach().cpu()).to(pred_wave.device)
        filtered_mag, _ = self.stft.transform(filtered_wave)
        loss_spec_consistency = F.l1_loss(filtered_mag, pred_mag.detach())

        # 加权总损失（α=10β）
        total_loss = self.alpha * loss_spec + loss_time + self.beta * loss_spec_consistency

        return {
            'total_loss': total_loss,
            'loss_spec': loss_spec,
            'loss_time': loss_time,
            'loss_spec_consistency': loss_spec_consistency,
            'filtered_wave': filtered_wave  # 返回滤波后的最终输出
        }


model = SpectrogramUNet()

dummy_input = torch.randn(2, 1, 512, 4096)  # 典型的心音频谱尺寸
output = model(dummy_input)

target = torch.rand_like(output)
loss = nn.L1Loss()(output, target)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print(f"输入尺寸: {dummy_input.shape}")
print(f"输出尺寸: {output.shape}")
print(f"L1损失计算验证: {loss.item():.4f}")


class HeartSoundDataset(Dataset):
    def __init__(self, waveform_list, transform=None):
        """
        waveform_list: 一个包含心音波形的列表，每个元素是一个 NumPy 数组
        transform: 可选的数据转换操作（如数据增强）
        """
        self.waveform_list = waveform_list
        self.transform = transform

    def __len__(self):
        return len(self.waveform_list)

    def __getitem__(self, idx):
        waveform = self.waveform_list[idx]
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for waveform in train_loader:
        waveform = waveform.to(device)

        optimizer.zero_grad()

        # 前向传播
        pred_wave, _ = model(waveform)

        # 计算损失
        loss_dict = criterion(pred_wave, waveform)
        loss = loss_dict['total_loss']

        # 反向传播 & 优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for waveform in val_loader:
            waveform = waveform.to(device)

            # 前向传播
            pred_wave, _ = model(waveform)

            # 计算损失
            loss_dict = criterion(pred_wave, waveform)
            loss = loss_dict['total_loss']

            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    # 超参数
    batch_size = 4
    epochs = 10
    lr = 1e-4
    n_fft = 2048
    hop_length = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据（假设你已经准备好心音数据）
    # 这里用随机噪声模拟数据，实际应用中应替换为真实的心音信号
    train_data = [np.random.randn(48000) for _ in range(100)]  # 100 个 1s 波形
    val_data = [np.random.randn(48000) for _ in range(20)]  # 20 个 1s 波形

    # 创建数据加载器
    train_dataset = HeartSoundDataset(train_data)
    val_dataset = HeartSoundDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型 & 损失函数 & 优化器
    model = TwoStageModel(n_fft=n_fft, hop_length=hop_length).to(device)
    criterion = PCGLoss(n_fft=n_fft, hop_length=hop_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 训练 & 评估
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{epochs}] - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "two_stage_model.pth")
    print("模型已保存！")


if __name__ == "__main__":
    main()
