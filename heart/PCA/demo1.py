from sklearn.decomposition import PCA
import librosa
import numpy as np
from scipy.io import wavfile
from sympy import false

from PCA.demo2 import SVD

max_frames = 20000
min_frames = 5000


def extract_features_with_mfcc(wav_file, n_mfcc=13, hop_length=512):
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
all_features = []
user_gender = [0, 1, 0, 1]  # 存储用户性别,0女1男

wav_files_list = [
    [
        '..\\data\\3\\record_generate1_25.wav', '..\\data\\3\\record_generate2_25.wav',
        '..\\data\\3\\record_generate3_25.wav', '..\\data\\3\\record_generate4_25.wav',
        '..\\data\\3\\record_generate5_25.wav', '..\\data\\3\\record_generate6_25.wav',
        '..\\data\\3\\record_generate7_25.wav', '..\\data\\3\\record_generate8_25.wav'
    ],
    [
        '..\\data\\1\\record_generate1_25.wav', '..\\data\\1\\record_generate2_25.wav',
        '..\\data\\1\\record_generate3_25.wav', '..\\data\\1\\record_generate4_25.wav',
        '..\\data\\1\\record_generate5_25.wav', '..\\data\\1\\record_generate6_25.wav',
        '..\\data\\1\\record_generate7_25.wav'
    ],
    [
        '..\\data\\2\\audio25(1).wav', '..\\data\\2\\audio25(2).wav', '..\\data\\2\\audio25(3).wav',
        '..\\data\\2\\audio25(4).wav', '..\\data\\2\\audio25(5).wav', '..\\data\\2\\audio25(6).wav'
    ],
    [
        '..\\data\\0\\record_generate1_25.wav', '..\\data\\0\\record_generate2_25.wav',
        '..\\data\\0\\record_generate3_25.wav', '..\\data\\0\\record_generate4_25.wav',
        '..\\data\\0\\record_generate5_25.wav', '..\\data\\0\\record_generate6_25.wav',
        '..\\data\\0\\record_generate7_25.wav', '..\\data\\0\\record_generate8_25.wav',
    ]
]

female_list = [wav_files_list[0], wav_files_list[2]]
male_list = [wav_files_list[1], wav_files_list[3]]

for user_files in wav_files_list:
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

user_ids = ['3', '1', '2']

new_list = ['..\\data\\survey1\\record_generate2_25.wav', '..\\data\\survey2\\record_generate2_25.wav',
            '..\\data\\survey3\\record_generate2_25.wav', '..\\data\\survey4\\record_generate2_25.wav',
            '..\\data\\survey5\\record_generate2_25.wav']
flag = false
# 处理新用户
for new_file in new_list:
    new_feat = extract_features_with_mfcc(new_file)
    new_feat = cut_wav_file(new_feat)
    mean, VT, selected_indices, user_profiles = SVD(wav_files_list, flag)
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

if min(distances) <= threshold:
    print("身份验证通过")
    print(user_ids[min_index])
else:
    print("身份验证失败")
