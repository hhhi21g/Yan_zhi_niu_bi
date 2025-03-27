import librosa
import numpy as np
import soundfile as sf
from scipy.fft import fft, ifft

earPhone_s = "earPhone.wav"
testfilt_s = "data/test_10s.wav"

signal_e, fs_e = librosa.load(earPhone_s, sr=11025)
signal_t, fs_t = librosa.load(testfilt_s, sr=11025)

minLen = min(len(signal_e), len(signal_t))
signal_e = signal_e[:minLen]
signal_t = signal_t[:minLen]

fft_e = fft(signal_e)
fft_t = fft(signal_t)

fft_result = fft_t - fft_e

res_signal = np.real(ifft(fft_result))

sf.write("data/test1_use.wav", res_signal, fs_e)
