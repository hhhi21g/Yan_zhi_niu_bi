from sklearn.decomposition import PCA
import librosa
import numpy as np

max_frames = 20000
min_frames = 5000


def extract_features_with_mfcc(wav_file, n_mfcc=13, hop_length=512):
    y, sr = librosa.load(wav_file, sr=None)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # 返回MFCC特征矩阵,即音频信号的一个低维度特征向量
    return mfcc.T  # 每一列为一个时间帧的特征


def cut_wav_file(extracted_matrix):
    if (extracted_matrix.shape[0] > max_frames + min_frames):
        extracted_matrix = extracted_matrix[min_frames:min_frames + max_frames]
    return extracted_matrix


feature_matrices = []

wav_files_list = [
    ['..\\data\\xyt\\audio5.wav', '..\\data\\xyt\\audio15.wav', '..\\data\\xyt\\audio25.wav'],
    ['..\\data\\lshenr\\audio5.wav', '..\\data\\lshenr\\audio15.wav', '..\\data\\lshenr\\audio25.wav'],
    ['..\\data\\lsr\\audio5.wav', '..\\data\\lsr\\audio15.wav', '..\\data\\lsr\\audio25.wav']
]

new_wav_file = '..\\data\\lsr\\audio5.wav'
new_user_features = extract_features_with_mfcc(new_wav_file)
new_user_features = cut_wav_file(new_user_features)

for wav_files in wav_files_list:
    user_features = []
    for wav_file in wav_files:
        feature_matrix = extract_features_with_mfcc(wav_file)
        feature_matrix = cut_wav_file(feature_matrix)
        user_features.append(feature_matrix)

    # 合并所有用户音频特征矩阵
    user_features_matrix = np.vstack(user_features)
    feature_matrices.append(user_features_matrix)

# 叠在一起，形成三位数组(音频数，时间帧数，特征维度)
# feature_stack = np.stack(feature_matrices, axis=0)

# 将多个用户的特征矩阵合并为一个生物特征矩阵
biometric_matrix = np.vstack(feature_matrices)

# 数据中心化
mean = np.mean(biometric_matrix,axis=0)
biometric_matrix_centered = biometric_matrix - mean

# 对生物特征矩阵进行奇异值分解(svd)
U, sigma, VT = np.linalg.svd(biometric_matrix_centered, full_matrices=False)
print(f"U shape: {U.shape}, Sigma: {sigma.shape}, VT shape: {VT.shape}")

# 计算归一化方差
variances = sigma ** 2
total_variance = np.sum(variances)
normalized_variances = variances / total_variance
print("Normalized Variance:", normalized_variances)

# 从第三个主成分开始进行主成分选择过程
threshold = 0.9
select_indices = np.where(normalized_variances < 0.9)[0]

k = 5
select_indices = select_indices[select_indices >= 2][:k]

select_U = U[:, select_indices]
select_sigma = sigma[select_indices]
select_VT = VT[select_indices, :]
print(user_features_matrix.shape)
print(select_U.shape)
# 将原始心音特征矩阵与选定的主成分矩阵相乘得到心脏摘要
transformed_features = np.dot(user_features_matrix, select_U)

# 用户档案
user_profiles = transformed_features


# 计算欧几里得距离
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


new_captured_summary = np.dot(new_user_features, select_U)

distances = []
for user_profile in user_profiles:
    dist = euclidean_distance(user_profile, new_captured_summary)
    distances.append(dist)

print(distances)
threshold = 0.9
min_distance = min(distances)
if min_distance <= threshold:
    print("身份验证通过")
else:
    print("身份验证失败")
