import matplotlib
import numpy as np
import pywt
import librosa
import torch
from matplotlib import pyplot as plt
import soundfile as sf
import torch.nn as nn

matplotlib.use('TkAgg')

def normalize_signal(signal_data):
    return signal_data / np.max(np.abs(signal_data))


def wavelet_denoise(signal, wavelet='db8', level=5):
    # 使用五层小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 计算阈值，硬阈值
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 估计噪声标准差
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # 硬阈值计算

    # 硬阈值去噪
    coeffs_thresholded = [coeffs[0]] + [pywt.threshold(np.array(c), threshold, mode='hard') for c in coeffs[1:]]

    # 小波重构
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    # 截断
    return denoised_signal[:len(signal)]


def extract_mfsc(signal, sr=8000, n_fft=256, hop_length=64, n_mels=40):
    # 计算STFT
    stft_result = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window='hamming')
    power_spectrogram = np.abs(stft_result) ** 2  # 计算功率谱

    # 计算梅尔能量
    mel_filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spectrogram = np.dot(mel_filterbank, power_spectrogram)

    log_mel_spectrogram = np.log10(mel_spectrogram + 1e-10)
    return log_mel_spectrogram

class SpatialAttention(nn.Module):
    def __init__(self,in_channels):
        super(SpatialAttention,self).__init__()
        self.in_channels = in_channels

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels,1,kernel_size=1)

    def forward(self,x):
        # 均值和最大值
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)

        # 均值和最大值合并，进行通道操作
        x_out = torch.cat([avg_out,max_out],dim=1)

        # 卷积层生成空间注意力权重
        attention = self.conv1(x_out)

        # 激活函数都
        attention = torch.sigmoid(attention)

        # 使用注意力权重对输入特征图加权
        out = x * attention

        return out


if __name__ == '__main__':
    filename = 'data/a0001.wav'
    signal,sr = librosa.load(filename,sr=8000)

    norm_signal = normalize_signal(signal)
    denoise_signal = wavelet_denoise(norm_signal)
    mfsc_features = extract_mfsc(denoise_signal)

    mfsc_tensor = torch.tensor(mfsc_features,dtype=torch.float32).unsqueeze(0).unsqueeze(0).float()
    mfsc_tensor = torch.cat([mfsc_tensor,mfsc_tensor],dim=1)
    spatial_attention = SpatialAttention(in_channels=2)

    enhanced_mfsc = spatial_attention(mfsc_tensor)

    enhanced_mfsc = enhanced_mfsc.squeeze(0).squeeze(0).detach().numpy()

    sf.write('denosed_heart_sound.wav',denoise_signal,sr)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfsc_features, sr=sr, hop_length=64, x_axis='time', y_axis='mel')
    plt.colorbar(label="Log Mel Energy")
    plt.title("MFSC Features of Heart Sound")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency")
    plt.show()
