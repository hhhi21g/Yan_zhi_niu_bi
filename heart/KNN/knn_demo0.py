from sklearn.decomposition import PCA
import librosa
import numpy as np
from scipy.io import wavfile
from sympy import false
import os
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler


def extract_features_with_mfcc(wav_file, n_mfcc=13, hop_length=512):
    y, sr = librosa.load(wav_file, sr=None)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # 返回MFCC特征矩阵，即音频信号的一个低纬度特征向量
    return mfcc.T  # (时间帧数, 特征维度)


def load_wav_files_from_dataset(root_folder):
    wav_files_list = []
    user_ids = []

    for user_folder in sorted(os.listdir(root_folder)):
        user_path = os.path.join(root_folder, user_folder)
        if os.path.isdir(user_path):
            wav_files = [
                os.path.join(user_path, f)
                for f in os.listdir(user_path)
                if f.endswith('.wav')
            ]
            if wav_files:  # 确保不是空文件夹
                wav_files_list.append(sorted(wav_files))  # 可选：按文件名排序
                user_ids.append(user_folder)

    return wav_files_list, user_ids


def SVD(user_list, flag):
    all_features = []
    for user_files in user_list:
        user_feats = []
        for file in user_files:
            feat = extract_features_with_mfcc(file)
            # feat = cut_wav_file(feat)
            user_feats.append(feat)
        # 合并用户所有音频的特征矩阵
        all_features.append(np.vstack(user_feats))
    # 将多个用户的特征矩阵合并为一个生物特征矩阵
    biometric_matrix = np.vstack(all_features)
    # 数据中心化
    mean = np.mean(biometric_matrix, axis=0)
    biometric_matrix_centered = biometric_matrix - mean

    # 对生物特征矩阵进行奇异值分解(svd)
    U, sigma, VT = np.linalg.svd(biometric_matrix_centered, full_matrices=False)

    # 主成分选择
    normalized_variances = (sigma ** 2) / (sigma ** 2).sum()
    sum_first_two = normalized_variances[:2].sum()
    sum_rest = 1.0 - sum_first_two
    required_sum = sum_rest * 0.8  # 累计剩余方差的90%，选择这些主成分

    # 舍弃前两个主成分，从第三个开始选择
    current_sum = 0.0
    selected_indices = []
    start = 2
    # if flag == false:
    #     start = 3
    # else:
    #     start = 0
    for idx in range(start, len(normalized_variances)):
        current_sum += normalized_variances[idx]
        selected_indices.append(idx)
        if current_sum >= required_sum:
            break

    # 构建用户档案
    user_profiles = []
    for user_files in user_list:
        user_feats = []
        for file in user_files:
            feat = extract_features_with_mfcc(file)
            # feat = cut_wav_file(feat)
            user_feats.append(feat)
        user_feats_matrix = np.vstack(user_feats)
        # 中心化并投影
        transformed = (user_feats_matrix - mean) @ VT.T[:, selected_indices]
        user_profiles.append(np.mean(transformed, axis=0))
    return mean, VT, selected_indices, user_profiles


# 加载所有用户数据构建生物特征矩阵，暂时建立三个档案
all_features = []

dataset_path = "../dataSet_wav_1epoch"
wav_files_list, user_ids = load_wav_files_from_dataset(dataset_path)

for user_files in wav_files_list:
    user_feats = []
    for file in user_files:
        feat = extract_features_with_mfcc(file)
        # feat = cut_wav_file(feat)
        user_feats.append(feat)
    # 合并用户所有音频的特征矩阵
    all_features.append(np.vstack(user_feats))

# 将多个用户的特征矩阵合并为一个生物特征矩阵
biometric_matrix = np.vstack(all_features)

# 数据中心化
mean = np.mean(biometric_matrix, axis=0)
biometric_matrix_centered = biometric_matrix - mean

# 对生物特征矩阵进行奇异值分解(svd)
U, sigma, VT = np.linalg.svd(biometric_matrix_centered, full_matrices=False)

# 主成分选择
normalized_variances = (sigma ** 2) / (sigma ** 2).sum()
sum_first_two = normalized_variances[:2].sum()
sum_rest = 1.0 - sum_first_two
required_sum = sum_rest * 0.9  # 累计剩余方差的90%，选择这些主成分

# 舍弃前两个主成分，从第三个开始选择
current_sum = 0.0
selected_indices = []
for idx in range(2, len(normalized_variances)):
    current_sum += normalized_variances[idx]
    selected_indices.append(idx)
    if current_sum >= required_sum:
        break

# 构建用户档案
user_profiles = []
X_train = []
y_train = []

for user_idx, user_files in enumerate(wav_files_list):
    for file in user_files:
        feat = extract_features_with_mfcc(file)
        transformed = (feat - mean) @ VT.T[:, selected_indices]
        profile = np.mean(transformed, axis=0)
        X_train.append(profile)
        y_train.append(user_ids[user_idx])

scaler = StandardScaler()
X_train = np.array(X_train)
y_train = np.array(y_train)

# X_train = scaler.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

new_list, new_user_id = load_wav_files_from_dataset("../testSet_wav_1epoch")
flag = False

current = 0
for i, user_files in enumerate(new_list):
    user_name = new_user_id[i]
    for new_file in user_files:
        new_feat = extract_features_with_mfcc(new_file)
        new_transformed = (new_feat - mean) @ VT.T[:, selected_indices]
        new_profile = np.mean(new_transformed, axis=0).reshape(1, -1)
        # new_profile = scaler.transform(new_profile)

        pred = knn.predict(new_profile)[0]

        print(f"样本文件：{os.path.basename(new_file)}，所属用户：{user_name} → 预测为：{pred}")
        if (new_user_id[i] == pred):
            current = current + 1
print(current)  # 36
