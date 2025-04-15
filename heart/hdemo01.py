import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa.feature
from scipy.spatial.distance import euclidean
import soundfile as sf

matplotlib.use('TkAgg')
y, sr = librosa.load("data/lhb/generated_audio5.wav", sr=11025)
time = np.arange(len(y)) / sr  # 时间轴（秒）


# 2. 低通滤波
def lowpass_filter(data, sr, cutoff=220):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(10, norm_cutoff, btype='low')
    return signal.filtfilt(b, a, data)


filtered = lowpass_filter(y, sr)


def noise_suppression(data):
    D = librosa.stft(data)
    S, phase = librosa.magphase(D)
    S_denoised = librosa.decompose.nn_filter(S, aggregate=np.median)
    return librosa.istft(S_denoised * phase)


filtered_denoised = noise_suppression(filtered)

# ===== 2. S1/S2检测 =====
# 计算短时能量
frame_length = int(0.14 * sr)
energy = np.array([
    np.sum(filtered[i:i + frame_length] ** 2)
    for i in range(0, len(filtered_denoised), frame_length)
])
energy_time = np.arange(len(energy)) * (frame_length / sr)  # 能量时间轴

# 设定能量阈值，去除低能量部分（非心音）
energy_threshold = np.median(energy) * 0.5  # 设定为能量中值的一半
filtered_energy = energy > energy_threshold  # 仅保留能量高于阈值的部分

filtered_energy_interp = np.interp(np.arange(len(filtered_denoised)),
                                   np.arange(0, len(filtered_denoised), frame_length), filtered_energy)
# 仅保留高能量部分的信号
filtered_denoised_high_energy = np.array(
    [filtered_denoised[i] if filtered_energy_interp[i] else 0 for i in range(len(filtered_denoised))])

# 检测能量峰值（S1/S2位置）
peaks, _ = signal.find_peaks(
    energy,
    distance=int(0.3 * sr / frame_length),  # 最小间隔0.3秒
    prominence=np.median(energy)
)

segment_length = int(0.14 * sr)  # 140ms窗口
s1_s2_segments = []
for i, peak in enumerate(peaks):
    start = peak * frame_length / sr  # 转换为时间（秒）
    end = start + 0.14
    s1_s2_segments.append((start, end))

mfcc_features = []
for start, end in s1_s2_segments:
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = filtered[start_sample:end_sample]

    # 提取13维MFCC特征（每帧10ms）
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, hop_length=int(0.01 * sr))
    mfcc_features.append(np.mean(mfcc, axis=1))  # 求每个MFCC特征的平均值

# ===== 5. 计算FSR特征 =====
fsr_features = []
for i, (start, end) in enumerate(s1_s2_segments):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = filtered[start_sample:end_sample]

    # 假设S1是前半部分，S2是后半部分（根据实际情况可以调整）
    half_point = len(segment) // 2
    energy_s1 = np.sum(segment[:half_point] ** 2)
    energy_s2 = np.sum(segment[half_point:] ** 2)
    print(energy_s1)
    print(energy_s2)
    # 计算FSR（S1能量与S2能量的比值）
    fsr = energy_s1 / energy_s2
    fsr_features.append(fsr)

# ===== 6. 模板匹配与身份验证 =====
# 假设我们有一个模板MFCC（已知的心音特征）
# 这里我们将假设模板来自前几段信号（例如第一个S1/S2对）
# template_mfcc = mfcc_features[0]

# 计算测试特征与模板之间的欧氏距离
# distances = [euclidean(feature, template_mfcc) for feature in mfcc_features]

# 基于FSR的加权距离（根据文章的描述）
# fsr_distances = [distance * fsr for distance, fsr in zip(distances, fsr_features)]

# ===== 7. 性能评估 =====
# 在此步骤中，可以根据欧氏距离和FSR加权距离来判断是否属于同一身份
# 假设你有一个阈值，判断是否为匹配身份
threshold = 1.5  # 设定一个阈值

# matches = [dist < threshold for dist in fsr_distances]

# ===== 8. 绘图 =====
plt.figure(figsize=(15, 8))

# 原始信号（灰色半透明）
plt.plot(time, y, color='gray', alpha=0.3, label='Raw Signal')

# 滤波后信号（蓝色）
plt.plot(time, filtered, color='blue', linewidth=0.8, label='Filtered Signal')

# 标注S1/S2段落
for i, (start, end) in enumerate(s1_s2_segments):
    start_time = start
    end_time = end
    if i % 2 == 0:
        plt.axvspan(start_time, end_time, color='red', alpha=0.2, label='S1' if i == 0 else "")
    else:
        plt.axvspan(start_time, end_time, color='green', alpha=0.2, label='S2' if i == 1 else "")

# 短时能量曲线（橙色，次坐标轴）
ax2 = plt.gca().twinx()
ax2.plot(energy_time, energy, color='orange', linewidth=1, label='Energy')
ax2.set_ylabel('Energy', color='orange')

# 标注设置
plt.title('Processed Heart Sound with S1/S2 Segmentation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 输出结果
# print("FSR加权距离：", fsr_distances)
# print("匹配结果（True为匹配，False为不匹配）：", matches)

sf.write("sub_and_filt.wav", filtered_denoised, sr)
