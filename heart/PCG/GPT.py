import torch
import torchaudio
import os

from PCG_stage1 import SpectrogramUNet  # STFT + UNet1
from PCG_stage2 import WaveformNet  # UNet2 + iSTFT


def pcg_signal_recovery(input_wav_path, output_wav_path, device='cuda'):
    # 加载音频
    waveform, sr = torchaudio.load(input_wav_path)

    # 模型实例化
    stage1 = SpectrogramUNet()
    stage2 = WaveformNet()

    # STFT + UNet1 处理（得到幅度谱 和 相位）
    mag1, phase1 = stage1.process(waveform)

    # 保存中间结果（可选，方便调试）
    os.makedirs('intermediate_results', exist_ok=True)
    torch.save(mag1, 'intermediate_results/mag1.pt')
    torch.save(phase1, 'intermediate_results/phase1.pt')

    # UNet2 处理 + iSTFT 波形恢复
    recovered_waveform = stage2.process(mag1, phase1)

    # 保存结果
    torchaudio.save(output_wav_path, recovered_waveform.cpu(), sample_rate=sr)


if __name__ == '__main__':
    # 输入输出路径自己定义
    input_wav_path = '..\\data\\h0.wav'
    output_wav_path = '..\\output\\recovered.wav'

    pcg_signal_recovery(input_wav_path, output_wav_path)
