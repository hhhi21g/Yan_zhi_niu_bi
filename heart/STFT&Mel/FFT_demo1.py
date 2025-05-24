import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.feature
import os
from sympy import false

# ========================== 参数设置 =============================
LOW_FREQ = 20
HIGH_FREQ = 200
TARGET_SR = 2000
THRESHOLD = 1.78
N_MFCC = 13
HOP_LENGTH = 256
MAX_FRAMES = 35 * TARGET_SR
MIN_FRAMES = 5 * TARGET_SR
N_FFT = 512

# ======================= 样本处理函数 ===========================
def cut_wav_file(extracted_matrix):
    if extracted_matrix.shape[0] > MAX_FRAMES + MIN_FRAMES:
        extracted_matrix = extracted_matrix[MIN_FRAMES:MIN_FRAMES + MAX_FRAMES]
    return extracted_matrix

def extract_features_with_mfcc(y, sr, n_fft = N_FFT,n_mfcc=N_MFCC, hop_length=HOP_LENGTH):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft = n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
    print(f"✅ MFCC shape: {mfcc.shape}, max={np.max(mfcc):.4f}, min={np.min(mfcc):.4f}")

    return mfcc.T

# ======================= 构建训练集 ============================
def build_train_test_split(root_dir):
    train_list = []
    test_list = []
    user_ids = []
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            user_files = sorted([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path) if f.endswith('.wav')
            ])
            if len(user_files) >= 2:
                train_list.append(user_files[:-1])
                test_list.append(user_files[-1])
                user_ids.append(folder_name)
    return train_list, test_list, user_ids

# ======================= SVD 建立用户档案 =====================
def SVD(user_list, flag):
    all_features = []
    for user_idx, user_files in enumerate(user_list):
        user_feats = []
        for file in user_files:

            y, sr = librosa.load(file, sr=None)
            y = y[int(5 * sr):]
            print(f"🎧 加载文件: {file}, 原始长度: {len(y)}")

            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            y = bandpass_filter(y, TARGET_SR)
            feat = extract_features_with_mfcc(y, TARGET_SR)
            feat = cut_wav_file(feat)
            if feat.shape[0] > 0:
                user_feats.append(feat)
            else:
                print(f"⚠️ 特征为空，跳过: {file}")

        if user_feats:
            all_features.append(np.vstack(user_feats))
        else:
            print(f"❌ 用户 {user_idx} 所有特征都无效，未加入训练。")

    if not all_features:
        raise ValueError("🚨 所有用户都无有效数据，终止训练。")

    # 下面保持不变
    biometric_matrix = np.vstack(all_features)
    mean = np.mean(biometric_matrix, axis=0)
    centered = biometric_matrix - mean
    U, sigma, VT = np.linalg.svd(centered, full_matrices=False)
    norm_var = (sigma ** 2) / (sigma ** 2).sum()
    sum_first_two = norm_var[:2].sum()
    required_sum = (1.0 - sum_first_two) * 0.8
    current_sum = 0.0
    selected_indices = []
    start = 3 if flag == false else 0
    for idx in range(start, len(norm_var)):
        current_sum += norm_var[idx]
        selected_indices.append(idx)
        if current_sum >= required_sum:
            break

    user_profiles = []
    for user_idx, user_files in enumerate(user_list):
        user_feats = []
        for file in user_files:
            try:
                y, sr = librosa.load(file, sr=None)
                y = y[int(5 * sr):]
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                y = bandpass_filter(y, TARGET_SR)
                feat = extract_features_with_mfcc(y, TARGET_SR)
                feat = cut_wav_file(feat)
                if feat.shape[0] > 0:
                    user_feats.append(feat)
            except:
                continue
        if user_feats:
            user_feats_matrix = np.vstack(user_feats)
            transformed = (user_feats_matrix - mean) @ VT.T[:, selected_indices]
            user_profiles.append(np.mean(transformed, axis=0))
        else:
            print(f"⚠️ 用户 {user_idx} 无有效特征，跳过档案构建。")

    return mean, VT, selected_indices, user_profiles


# ====================== 带通滤波函数 =========================
def bandpass_filter(y, sr):
    N = len(y)
    T = 1 / sr
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, d=T)
    mask = (np.abs(freqs) >= LOW_FREQ) & (np.abs(freqs) <= HIGH_FREQ)
    Y_filtered = Y * mask
    y_filtered = np.fft.ifft(Y_filtered).real
    return y_filtered

# ======================== 主程序入口 ========================
if __name__ == '__main__':
    root_data_dir = '..\\dataSet_original'
    train_list, test_list, user_ids = build_train_test_split(root_data_dir)
    mean, VT, selected_indices, user_profiles = SVD(train_list, flag=false)

    for test_path, true_id in zip(test_list, user_ids):
        print(f"\n📄 测试文件: {test_path} (真实身份: {true_id})")
        y, sr = librosa.load(test_path, sr=None)
        y = y[int(5 * sr):]
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        y_filtered = bandpass_filter(y, TARGET_SR)
        y, sr = librosa.load(test_path, sr=None)

        mfcc_feat = extract_features_with_mfcc(y_filtered, TARGET_SR)
        mfcc_feat = cut_wav_file(mfcc_feat)
        new_transformed = (mfcc_feat - mean) @ VT.T[:, selected_indices]
        new_profile = np.mean(new_transformed, axis=0)

        distances = [np.linalg.norm(up - new_profile) for up in user_profiles]
        min_distance = min(distances)
        min_index = distances.index(min_distance)

        print(f"🧬 预测身份索引：{min_index}（预测ID：{user_ids[min_index]}）")
        print(f"✅ 最小距离：{min_distance:.3f}")
        if min_distance <= THRESHOLD:
            print("🎉 身份验证通过")
        else:
            print("❌ 身份验证失败")
