from sklearn.decomposition import PCA
import librosa
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler

# ========== 数据增强函数 ==========
def add_noise(y, noise_factor=0.001):
    noise = np.random.randn(len(y))
    return (y + noise_factor * noise).astype(np.float32)

def time_stretch(y, rate=1.05):
    return librosa.effects.time_stretch(y=y, rate=rate)

def extract_features_from_array(y, sr, n_mfcc=16, hop_length=480, n_fft=2048, energy_thresh=4):
    energy = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.vstack([mfcc, delta, delta2])
    threshold = np.percentile(energy, energy_thresh)
    valid_idx = energy > threshold
    return mfcc_combined[:, valid_idx].T

# ========== 加载音频路径 ==========
def load_wav_files_from_dataset(root_folder):
    wav_files_list = []
    user_ids = []
    for user_folder in sorted(os.listdir(root_folder)):
        user_path = os.path.join(root_folder, user_folder)
        if os.path.isdir(user_path):
            wav_files = [
                os.path.join(user_path, f) for f in os.listdir(user_path) if f.endswith('.wav')
            ]
            if wav_files:
                wav_files_list.append(sorted(wav_files))
                user_ids.append(user_folder)
    return wav_files_list, user_ids

# ========== 构建训练集 + 数据增强 ==========
dataset_path = "../dataSet_wav_1epoch"
wav_files_list, user_ids = load_wav_files_from_dataset(dataset_path)

X_train, y_train = [], []
all_features = []

for user_idx, user_files in enumerate(wav_files_list):
    for file in user_files:
        y, sr = librosa.load(file, sr=None)
        for y_aug in [y, add_noise(y), time_stretch(y)]:
            try:
                feat = extract_features_from_array(y_aug, sr)
                all_features.append(feat)
            except:
                continue

biometric_matrix = np.vstack(all_features)
mean = np.mean(biometric_matrix, axis=0)
biometric_matrix_centered = biometric_matrix - mean
U, sigma, VT = np.linalg.svd(biometric_matrix_centered, full_matrices=False)

normalized_variances = (sigma ** 2) / (sigma ** 2).sum()
sum_rest = 1.0 - normalized_variances[:2].sum()
required_sum = sum_rest * 0.9
current_sum, selected_indices = 0.0, []
for idx in range(2, len(normalized_variances)):
    current_sum += normalized_variances[idx]
    selected_indices.append(idx)
    if current_sum >= required_sum:
        break

# ========== 重新提取训练特征（含增强） ==========
X_train, y_train = [], []
for user_idx, user_files in enumerate(wav_files_list):
    for file in user_files:
        y, sr = librosa.load(file, sr=None)
        for y_aug in [y, add_noise(y), time_stretch(y)]:
            try:
                feat = extract_features_from_array(y_aug, sr)
                transformed = (feat - mean) @ VT.T[:, selected_indices]
                for frame in transformed:
                    X_train.append(frame)
                    y_train.append(user_ids[user_idx])
            except:
                continue

scaler = StandardScaler()
X_train = scaler.fit_transform(np.array(X_train))
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

# ========== 测试阶段 ==========
new_list, new_user_id = load_wav_files_from_dataset("../testSet_wav_1epoch")
current = 0
for i, user_files in enumerate(new_list):
    user_name = new_user_id[i]
    for new_file in user_files:
        y_test, sr_test = librosa.load(new_file, sr=None)
        feat_test = extract_features_from_array(y_test, sr_test)
        transformed_test = (feat_test - mean) @ VT.T[:, selected_indices]
        transformed_test = scaler.transform(transformed_test)
        pred_frames = knn.predict(transformed_test)
        pred = Counter(pred_frames).most_common(1)[0][0]
        print(f"样本文件：{os.path.basename(new_file)}，所属用户：{user_name} → 预测为：{pred}")
        if pred == user_name:
            current += 1
print(current)