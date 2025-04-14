from sklearn.decomposition import PCA
import librosa
import numpy as np


def extract_features_with_mfcc(wav_file, n_mfcc=13, hop_length=512):
    y, sr = librosa.load(wav_file, sr=None)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # 返回MFCC特征矩阵
    return mfcc.T  # 每一列为一个时间帧的特征


feature_matrices = []

wav_files = ['..\\data\\xyt\\audio5.wav', '..\\data\\xyt\\audio15.wav', '..\\data\\xyt\\audio25.wav']

for wav_file in wav_files:
    feature_matrix = extract_features_with_mfcc(wav_file)
    feature_matrices.append(feature_matrix)

# 叠在一起，形成三位数组(音频数，时间帧数，特征维度)
feature_stack = np.stack(feature_matrices, axis=0)

# 按列计算中位数
aggregated_features = np.median(feature_stack, axis=0)

print(aggregated_features.shape)
print(aggregated_features)

pca = PCA(n_components=10)  # 降到10个主成分
transformed_data = pca.fit_transform(aggregated_features)

# 使用PCA降维后的结果
print(transformed_data.shape)
print(transformed_data)

U, sigma, VT = np.linalg.svd(transformed_data, full_matrices=False)

# 计算每个主成分的方差
variances = sigma ** 2
total_variance = np.sum(variances)
normalized_variance = variances/total_variance

print(normalized_variance)

