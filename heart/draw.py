import librosa
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
filename = 'sub_and_filt.wav'  # 替换为你的音频文件路径
signal, fs = librosa.load(filename, sr=None)  # sr=None 保持原始采样率

# 创建时间轴
time = np.linspace(0, len(signal) / fs, num=len(signal))

# 绘制波形
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label="Waveform")
plt.title("Waveform of the Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
