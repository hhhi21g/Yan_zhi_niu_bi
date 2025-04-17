import numpy as np
np.complex = complex
np.float = float

import librosa
import scipy.fftpack
import joblib
import os

# ========== 配置 ==========
FRAME_LEN = 1024
FRAME_SHIFT = 1024
N_CEPS = 60  # 与训练时一致即可
MODEL_PATH = "gmm_heart_sound_model_fiveonly.pkl"

# ========== 特征提取函数 ==========
def extract_lfbc_features(signal, sr, n_ceps=N_CEPS):
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frames = librosa.util.frame(emphasized, frame_length=FRAME_LEN, hop_length=FRAME_SHIFT).T
    if frames.shape[0] < 2:
        return np.array([])
    window = np.hanning(FRAME_LEN)
    windowed_frames = frames * window
    mag_frames = np.abs(np.fft.rfft(windowed_frames, axis=1))
    freqs = np.fft.rfftfreq(FRAME_LEN, d=1.0/sr)
    valid_bins = np.where((freqs >= 20) & (freqs <= 150))[0]
    filtered = mag_frames[:, valid_bins]
    log_spec = np.log(filtered + 1e-10)
    lfbc = scipy.fftpack.dct(log_spec, type=2, axis=1, norm='ortho')[:, :n_ceps]
    energies = np.sum(filtered ** 2, axis=1)
    db_energy = 10 * np.log10(energies + 1e-10)
    threshold = np.min(db_energy) + 6
    lfbc = lfbc[db_energy < threshold]
    lfbc -= np.mean(lfbc, axis=0)
    return lfbc

# ========== 待识别的测试音频路径 ==========
test_files = [
    "..\\data\\xyt\\audio25.wav",
    "..\\data\\lshenr\\audio25.wav",
    "..\\data\\lsr\\audio25.wav"
]

# ========== 加载模型 ==========
print(f"📦 正在加载模型：{MODEL_PATH}")
gmm_model = joblib.load(MODEL_PATH)

# ========== 提取特征并评分 ==========
print("\n🎧 开始识别测试音频：")
min_ceps = gmm_model.means_.shape[1]  # 使用训练时的维度

for path in test_files:
    if not os.path.exists(path):
        print(f"🚫 未找到文件: {path}")
        continue

    y, sr = librosa.load(path, sr=None)
    feats = extract_lfbc_features(y, sr, n_ceps=min_ceps)

    if feats.shape[0] < 2:
        print(f"⚠️  {path} 特征太少，跳过")
        continue

    # 如果提取的维度多于模型期望维度 → 截断
    if feats.shape[1] > min_ceps:
        feats = feats[:, :min_ceps]
    elif feats.shape[1] < min_ceps:
        print(f"⚠️  {path} 提取特征维度不足，跳过")
        continue

    score = gmm_model.score(feats)
    print(f"{path}: GMM 对数似然得分 = {score:.4f}")