import os
import numpy as np
import librosa
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import random

# 🧠 动态特征提取
def extract_features_from_wave(y, sr, n_mfcc=15, hop_length=128, n_fft=1024):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate([mfcc, delta, delta2], axis=0)
    return combined.T  # (T, 45)

# 🎧 数据增强函数
def augment_wave(y, sr):
    augments = []
    augments.append(y + 0.005 * np.random.randn(len(y)))
    augments.append(np.roll(y, int(0.05 * sr)))
    augments.append(y * np.random.uniform(0.7, 1.3))
    for noise_level in [0.003, 0.007]:
        augments.append(y + noise_level * np.random.randn(len(y)))
    for shift_sec in [0.03, 0.07]:
        augments.append(np.roll(y, int(sr * shift_sec)))
    for stretch_rate in [0.95, 1.05]:
        try:
            aug = librosa.effects.time_stretch(y, rate=stretch_rate)
            augments.append(aug)
        except:
            pass
    for pitch_step in [-2, 2]:
        try:
            aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_step)
            augments.append(aug)
        except:
            pass
    try:
        stacked = librosa.effects.time_stretch(y, rate=1.05)
        stacked = stacked + 0.004 * np.random.randn(len(stacked))
        augments.append(stacked)
    except:
        pass
    return random.sample(augments, k=min(len(augments), 10))

# 📂 加载音频路径和标签
def collect_all_wavs(root_folder):
    wav_paths = []
    labels = []
    for user_id in sorted(os.listdir(root_folder)):
        user_path = os.path.join(root_folder, user_id)
        if not os.path.isdir(user_path):
            continue
        for file in sorted(os.listdir(user_path)):
            if file.endswith('.wav'):
                wav_paths.append(os.path.join(user_path, file))
                labels.append(user_id)
    return wav_paths, labels

# 🧪 主流程：K-Fold 验证
dataset_path = "../dataSet_wav_1epoch"
wav_files, labels = collect_all_wavs(dataset_path)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

print("\n开始进行 K-Fold 验证...")
for fold, (train_idx, test_idx) in enumerate(kf.split(wav_files)):
    X_train, y_train, X_test, y_test = [], [], [], []

    print(f"\nFold {fold+1}: 正在增强训练数据...")
    for i in tqdm(train_idx, desc=f"Fold {fold+1} 增强中"):
        path = wav_files[i]
        label = labels[i]
        y_raw, sr = librosa.load(path, sr=None)
        feats = [extract_features_from_wave(y_raw, sr)]
        for _ in range(5):
            for aug in augment_wave(y_raw, sr):
                feats.append(extract_features_from_wave(aug, sr))
        for f in feats:
            mean = np.mean(f, axis=0)
            std = np.std(f, axis=0)
            q75, q25 = np.percentile(f, [75, 25], axis=0)
            iqr = q75 - q25
            stats = np.concatenate([mean, std, iqr])
            X_train.append(stats)
            y_train.append(label)

    for i in test_idx:
        path = wav_files[i]
        label = labels[i]
        y_raw, sr = librosa.load(path, sr=None)
        feat = extract_features_from_wave(y_raw, sr)
        mean = np.mean(feat, axis=0)
        std = np.std(feat, axis=0)
        q75, q25 = np.percentile(feat, [75, 25], axis=0)
        iqr = q75 - q25
        stats = np.concatenate([mean, std, iqr])
        X_test.append(stats)
        y_test.append(label)

    X_train, y_train = np.array(X_train), le.transform(y_train)
    X_test, y_test = np.array(X_test), le.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=30)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=7, metric='manhattan')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold+1} 准确率: {acc*100:.2f}%")

print(f"\nK-Fold 平均准确率: {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
