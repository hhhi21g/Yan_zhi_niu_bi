import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
import soundfile as sf

filename = "data/earPhone_10s.wav"
signal, fs = librosa.load(filename, sr=11025, mono=False)  # 保持原始采样率

cutoff = 20
order = 4
b, a = butter(order, cutoff / (fs / 2), btype='low')

if signal.ndim == 1:
    filtered_signal = filtfilt(b, a, signal)
else:
    filtered_signal = np.asarray([filtfilt(b, a, channel) for channel in signal])

sf.write("earPhone.wav", filtered_signal.T, fs, subtype="FLOAT")

# 绘制时域信号
num_samples = signal.shape[1] if signal.ndim > 1 else len(signal)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, num_samples / fs, num_samples),
         signal[0] if signal.ndim > 1 else signal, label="original signal", alpha=0.7)
plt.title("original signal")
plt.xlabel("time (s)")
plt.ylabel("A")
plt.legend()

num_samples = filtered_signal.shape[1] if filtered_signal.ndim > 1 else len(filtered_signal)

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, num_samples / fs, num_samples),
         filtered_signal[0] if filtered_signal.ndim > 1 else filtered_signal, label="filtered signal", color='r',
         alpha=0.7)
plt.title("filtered signal")
plt.xlabel("time (s)")
plt.ylabel("A")
plt.legend()
plt.tight_layout()
plt.show()


# 计算傅里叶变换
def plot_spectrum(signal, fs, title):
    n = len(signal)
    yf = fft(signal)  # 快速傅里叶变换
    xf = fftfreq(n, 1 / fs)  # 频率轴

    plt.figure(figsize=(10, 4))
    plt.plot(xf[:n // 2], np.abs(yf[:n // 2]) / n, label=title)  # 只绘制正频率部分
    plt.xscale("log")
    plt.xlabel("freq (Hz)")
    plt.ylabel("A")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


if signal.ndim == 1:
    plot_spectrum(signal, fs, "original signal")
    plot_spectrum(filtered_signal, fs, "filtered signal")
else:
    plot_spectrum(signal[0], fs, "original signal(1)")
    plot_spectrum(filtered_signal[0], fs, "filtered signal(1)")
