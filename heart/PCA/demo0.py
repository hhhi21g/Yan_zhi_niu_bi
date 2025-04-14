from sklearn.decomposition import PCA
import librosa
import numpy as np

def extract_features_with_mfcc(wav_file,n_mfcc=13,hop_length=512):
    y,sr = librosa.load(wav_file,sr=None)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc,hop_length=hop_length)

    # 返回MFCC特征矩阵
    return mfcc.T  # 每一列为一个时间帧的特征


feature_matrices = []

wav_files

