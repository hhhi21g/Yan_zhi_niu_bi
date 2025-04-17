from sklearn.decomposition import PCA
import librosa
import numpy as np
from scipy.io import wavfile

max_frames = 15000
min_frames = 5000


def extract_features_with_mfcc(wav_file, n_mfcc=15, hop_length=64):
    y, sr = librosa.load(wav_file, sr=None)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # 返回MFCC特征矩阵，即音频信号的一个低纬度特征向量
    return mfcc.T  # (时间帧数, 特征维度)


# 剪切音频时间帧
def cut_wav_file(extracted_matrix):
    if extracted_matrix.shape[0] > max_frames + min_frames:
        extracted_matrix = extracted_matrix[min_frames:min_frames + max_frames]
    return extracted_matrix


# 加载所有用户数据构建生物特征矩阵，暂时建立三个档案

user_gender = [0, 1, 0, 1]  # 存储用户性别,0女1男

wav_files_list = [
    ['..\\data\\xyt\\audio5.wav', '..\\data\\xyt\\audio15.wav', '..\\data\\xyt\\audio25.wav'],
    ['..\\data\\lshenr\\audio5.wav', '..\\data\\lshenr\\audio15.wav', '..\\data\\lshenr\\audio25.wav'],
    ['..\\data\\lsr\\audio5.wav', '..\\data\\lsr\\audio15.wav', '..\\data\\lsr\\audio25.wav'],
    ['..\\data\\lhb\\audio5.wav', '..\\data\\lhb\\audio10.wav', '..\\data\\lhb\\audio15.wav']
]

female_list = [wav_files_list[0], wav_files_list[2]]
male_list = [wav_files_list[1], wav_files_list[3]]


def SVD(user_list):
    all_features = []
    for user_files in user_list:
        user_feats = []
        for file in user_files:
            feat = extract_features_with_mfcc(file)
            feat = cut_wav_file(feat)
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
    for idx in range(2, len(normalized_variances)):
        current_sum += normalized_variances[idx]
        selected_indices.append(idx)
        if current_sum >= required_sum:
            break

    # 构建用户档案
    user_profiles = []
    for user_files in wav_files_list:
        user_feats = []
        for file in user_files:
            feat = extract_features_with_mfcc(file)
            feat = cut_wav_file(feat)
            user_feats.append(feat)
        user_feats_matrix = np.vstack(user_feats)
        # 中心化并投影
        transformed = (user_feats_matrix - mean) @ VT.T[:, selected_indices]
        user_profiles.append(np.mean(transformed, axis=0))
    return mean, VT, selected_indices, user_profiles


user_ids = ['xyt', 'lshenr', 'lsr', 'lhb']
female_ids = ['xyt', 'lsr']
male_ids = ['lshenr', 'lhb']

# 处理新用户
new_feat = extract_features_with_mfcc('..\\data\\xyt\\audio25.wav')
new_feat = cut_wav_file(new_feat)
mean, VT, selected_indices, user_profiles = SVD(wav_files_list)
new_transformed = (new_feat - mean) @ VT.T[:, selected_indices]
new_profile = np.mean(new_transformed, axis=0)

# 计算欧几里得距离
distances = [np.linalg.norm(up - new_profile) for up in user_profiles]
threshold = 1.78  # 需要根据实际数据校准

# 找到最小距离和对应的用户索引
min_distance = min(distances)
min_index = distances.index(min_distance)

print(distances)
print(min_distance)
print(min_index)
print(user_gender[min_index])
print("************************************************************************")

if user_gender[min_index] == 0:  # 女
    mean_0, VT_0, selected_indices_0, user_profiles_0 = SVD(female_list)
    new_female_transformed = (new_feat - mean_0) @ VT_0.T[:, selected_indices_0]
    new_female_profile = np.mean(new_female_transformed, axis=0)
    distances_0 = [np.linalg.norm(up - new_female_profile) for up in user_profiles_0]
    new_min_distance = min(distances_0)
    print(distances_0)
    new_min_index = distances_0.index(new_min_distance)

if user_gender[min_index] == 1:  # 男
    mean_1, VT_1, selected_indices_1, user_profiles_1 = SVD(male_list)
    new_male_transformed = (new_feat - mean_1) @ VT_1.T[:, selected_indices_1]
    new_male_profile = np.mean(new_male_transformed, axis=0)
    distances_1 = [np.linalg.norm(up - new_male_profile) for up in user_profiles_1]
    print(distances_1)
    new_min_distance = min(distances_1)
    new_min_index = distances_1.index(new_min_distance)

print(new_min_distance)
print(new_min_index)

if new_min_index <= threshold:
    print("身份验证通过")
    if user_gender[min_index] == 0:
        print(female_ids[new_min_index])
    else:
        print(male_ids[new_min_index])
else:
    print("身份验证失败")
