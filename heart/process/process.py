import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import ScalarFormatter
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import soundfile as sf
from scipy.spatial.distance import euclidean
import os
import math


def read_m4a_and_clip(file_path, start_time, end_time):
    # 使用 librosa 加载音频文件
    data, sample_rate = librosa.load(file_path, sr=None)  # sr=None 保持原始采样率
    if data.ndim == 2:  # 如果是立体声，提取左声道
        data = data[:, 0]
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    clipped_data = data[start_sample:end_sample]
    return sample_rate, clipped_data


def generate_reference_signal(sample_rate, frequency, num_samples):
    t = np.arange(num_samples) / sample_rate
    return np.cos(2 * np.pi * frequency * t)


def mix_signal(received_signal, reference_signal):
    return received_signal * reference_signal


def low_pass_filter(signal, cutoff_frequency, sample_rate, order=4):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def extract_heart_signal(received_signal, reference_frequency, sample_rate, cutoff_frequency, start_time, end_time):
    num_samples = received_signal.shape[0]
    reference_signal = generate_reference_signal(sample_rate, reference_frequency, num_samples)
    mixed_signal = mix_signal(received_signal, reference_signal)
    filtered_signal = low_pass_filter(mixed_signal, cutoff_frequency, sample_rate)
    return filtered_signal


def echo(file_path, reference_frequency, cutoff_frequency, plot_title):
    start_time = 3  # 起始时间，单位秒
    end_time = 10  # 结束时间，单位秒

    sample_rate, received_signal = read_m4a_and_clip(file_path, start_time, end_time)
    '''
    #low_filter = low_pass_filter(received_signal,cutoff_frequency,sample_rate)
    x = np.arange(start_time, end_time, 1 / sample_rate)
    plt.plot(x,received_signal)
    '''
    heart_signal = extract_heart_signal(received_signal, reference_frequency, sample_rate, cutoff_frequency, start_time,
                                        end_time)

    # 可视化结果
    x = np.arange(len(received_signal)) / sample_rate + start_time

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, received_signal)
    plt.title(plot_title + ' Received Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(x, heart_signal)
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-3, -3))  # 设置科学计数法的指数范围
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title(plot_title + ' processed signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    return heart_signal, x, sample_rate


def ica_proc(reference_frequency, cutoff_frequency, file_path):
    plot_title = f'reference_frequency = {reference_frequency}, cutoff_frequency = {cutoff_frequency}'
    heart_signal, x, sample_rate = echo(file_path, reference_frequency, cutoff_frequency, plot_title)
    numOfIC = 0
    ica = FastICA(n_components=None, algorithm='deflation',
                  whiten='unit-variance', fun='cube', fun_args=None, max_iter=200,
                  tol=0.0001, w_init=None, random_state=None)
    heart_signal = heart_signal.reshape(-1, 1)
    S_estimated = ica.fit_transform(heart_signal)  # 估计的独立成分
    A_estimated = ica.mixing_  # 估计的混合矩阵
    print("估计的独立成分形状:", S_estimated.shape)
    print("估计的混合矩阵:\n", A_estimated)
    return sample_rate, S_estimated


def main():

    reference_frequencies = 21000
    file_path = "..\\dataSet_original\\lsr\\record2.m4a"

    save_dir = "..\\dataSet_pic_1epoch\\lsr"
    os.makedirs(save_dir, exist_ok=True)

    # 利用5Hz图像找出最低处作为分割点
    sample_rate, S_estimated_5Hz = ica_proc(reference_frequencies, cutoff_frequency=5, file_path=file_path)

    window = 0.55

    for i in range(S_estimated_5Hz.shape[1]):
        split_point = []
        audio = S_estimated_5Hz[:, i]
        print(sample_rate)
        for j in range(math.floor(len(audio) / sample_rate / window) - 1):
            sample_audio = torch.tensor(
                audio[(math.floor(j * window * sample_rate)): (math.floor((j + 1) * window * sample_rate))])
            d = torch.argmin(sample_audio)
            split_point.append(math.floor(j * window * sample_rate) + d)
    print(split_point)

    n_component = len(split_point) - 2
    fig, axes = plt.subplots(n_component, 1, figsize=(12, 2 * (n_component)))
    # 提取25Hz内的周期
    sample_rate, S_estimated_25Hz = ica_proc(reference_frequencies, cutoff_frequency=25, file_path=file_path)

    audio = S_estimated_25Hz[:, 0]

    # 计算需要显示的分段数量（split_point中 j >=2 的有效分段数）
    n_segments = len(split_point) - 2  # 例如 split_point有7个点，则有效分段数为5
    fig, axes = plt.subplots(n_segments, 1, figsize=(12, 2 * n_segments))

    for j in range(2, len(split_point)-1):  # 从 j=2 开始处理有效分段
        # 当前子图索引
        ax_idx = j - 2
        ax = axes[ax_idx]

        # 提取分段区间
        start = split_point[j - 1].item()
        end = split_point[j+1].item()

        # 提取分段数据
        sample_audio = audio[start:end]  # 直接使用张量切片

        # 生成时间轴（单位：秒）
        time_axis = np.linspace(start / sample_rate, end / sample_rate, len(sample_audio))

        if (end / sample_rate - start / sample_rate) < window:
            continue

            # 创建单独的图像
        plt.figure(figsize=(10, 2.5))
        plt.plot(time_axis, sample_audio, color='blue', linewidth=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Segment {j - 1}: {start / sample_rate:.2f}s to {end / sample_rate:.2f}s")
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, f"segment_{j - 1}.png")
        plt.savefig(save_path)
        plt.close()  # 关闭当前图，避免内存累积
        print(f"✅ 保存: {save_path}")
    #     # 绘制波形
    #     ax.plot(time_axis, sample_audio, color='blue', linewidth=0.8)
    #
    #     # 设置标签和标题
    #     ax.set_xlabel("Time (s)")
    #     ax.set_ylabel("Amplitude")
    #     ax.set_title(f"Segment {ax_idx + 1}: {start / sample_rate:.2f}s to {end / sample_rate:.2f}s")
    #
    # plt.tight_layout()
    # plt.show()

    """x = np.arange(split_point[0], split_point[-1])
    ax[-1].plot(x, S_estimated_25Hz)

    plt.tight_layout()
    plt.show()"""


if __name__ == "__main__":
    main()
