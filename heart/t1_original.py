import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import soundfile as sf
import pyaudio


def read_m4a_and_clip(file_path, start_time, end_time):
    # 使用 librosa 加载音频文件
    data, sample_rate = librosa.load(file_path, sr=None)  # sr=None 保持原始采样率
    if data.ndim == 2:  # 如果是立体声，提取左声道
        data = data[:, 0]
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    clipped_data = data[start_sample:end_sample]
    return sample_rate, clipped_data


def generate_reference_signal(sample_rate, frequency, duration):
    t = np.arange(0, duration, 1 / sample_rate)
    return np.cos(2 * np.pi * frequency * t)


def mix_signal(received_signal, reference_signal):
    return received_signal * reference_signal


def low_pass_filter(signal, cutoff_frequency, sample_rate, order=4):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def extract_heart_signal(received_signal, reference_frequency, sample_rate, cutoff_frequency, start_time, end_time):
    duration = end_time - start_time
    reference_signal = generate_reference_signal(sample_rate, reference_frequency, duration)
    mixed_signal = mix_signal(received_signal, reference_signal)
    filtered_signal = low_pass_filter(mixed_signal, cutoff_frequency, sample_rate)
    return filtered_signal


def echo(file_path, reference_frequency, cutoff_frequency, plot_title):
    start_time = 4  # 起始时间，单位秒
    end_time = 9  # 结束时间，单位秒

    sample_rate, received_signal = read_m4a_and_clip(file_path, start_time, end_time)
    '''
    #low_filter = low_pass_filter(received_signal,cutoff_frequency,sample_rate)
    x = np.arange(start_time, end_time, 1 / sample_rate)
    plt.plot(x,received_signal)
    '''
    heart_signal = extract_heart_signal(received_signal, reference_frequency, sample_rate, cutoff_frequency, start_time,
                                        end_time)

    # 可视化结果
    x = np.arange(start_time, end_time, 1 / sample_rate)
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


def main():
    reference_frequencies = {21000}
    cutoff_frequencies = {25, 15, 5}

    for reference_frequency in reference_frequencies:
        file_path = "dataSet\\a0001.wav"
        for cutoff_frequency in cutoff_frequencies:
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
            plt.figure(figsize=(10, 6))

            # 假设分离出的独立成分有多个，绘制每个成分
            for i in range(S_estimated.shape[1]):
                plt.subplot(S_estimated.shape[1], 1, i + 1)
                plt.plot(x, S_estimated[:, i])
                plt.title(f"Independent Component {i + 1}")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                sf.write(f'a0001_{cutoff_frequency}.wav', S_estimated[:, i], sample_rate)

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
