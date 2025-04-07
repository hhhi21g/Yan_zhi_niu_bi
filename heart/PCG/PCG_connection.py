import torch
import torch.nn as nn
import torch.optim as optim
import iSTFT
import PCG_stage1
import PCG_stage2
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F


class HeartSoundDataset(Dataset):
    def __init__(self, waveforms, sr=48000, duration=1.0):
        """
        waveforms: List[np.array] 心音波形列表
        sr: 采样率
        duration: 每个样本的秒数
        """
        self.samples = []
        target_length = int(sr * duration)

        for wave in waveforms:
            # 标准化长度
            if len(wave) > target_length:
                wave = wave[:target_length]
            else:
                wave = np.pad(wave, (0, target_length - len(wave)))

            # 标准化幅度
            wave = wave / (np.max(np.abs(wave)) + 1e-8)
            self.samples.append(torch.FloatTensor(wave))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TwoStageModel(nn.Module):
    def __init__(self, n_fft=2048, hop_length=1024):
        super().__init__()
        self.stft = iSTFT.STFT(filter_length=n_fft, hop_length=hop_length)
        self.spec_unet = PCG_stage1.SpectrogramUNet()
        self.wave_unet = PCG_stage2.WaveformNet()

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
    def __init__(self, n_fft=2048, hop_length=1024, alpha=10.0, beta=1.0, sr=48000):
        super().__init__()
        self.stft = iSTFT.STFT(filter_length=n_fft, hop_length=hop_length)
        self.alpha = alpha
        self.beta = beta
        self.sr = sr

    def forward(self, pred_wave, target_wave):
        # 第一阶段：频谱损失计算
        pred_mag, _ = self.stft.transform(pred_wave)[0]  # [B, F, T]
        target_mag, _ = self.stft.transform(target_wave)[0]
        loss_spec = F.l1_loss(pred_mag, target_mag)

        # 第二阶段：时域损失 + 频谱一致性
        loss_time = F.l1_loss(pred_wave, target_wave)

        # 加权总损失
        total_loss = self.alpha * loss_spec + loss_time + self.beta * loss_spec
        return total_loss


def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据准备 (示例，您可以替换为实际的数据)
    # 使用模拟的心音波形数据作为示例，实际应用时替换为实际的心音数据
    train_waves = [np.random.randn(48000) for _ in range(100)]  # 100个训练样本
    val_waves = [np.random.randn(48000) for _ in range(20)]  # 20个验证样本

    train_set = HeartSoundDataset(train_waves)
    val_set = HeartSoundDataset(val_waves)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    # 模型初始化
    model = TwoStageModel().to(device)
    criterion = PCGLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 训练循环
    for epoch in range(10):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            pred, _ = model(batch)
            loss = criterion(pred, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                pred, _ = model(batch)
                val_loss += criterion(pred, batch).item()

        print(f"Epoch {epoch + 1} | Val Loss: {val_loss / len(val_loader):.4f}")


if __name__ == "__main__":
    main()
