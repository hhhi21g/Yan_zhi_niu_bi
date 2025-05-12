import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def extract_features_with_mfcc(wav_file, n_mfcc=15, hop_length=480, n_fft=2048):
    y, sr = librosa.load(wav_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    delta = librosa.feature.delta(mfcc)
    feat = np.vstack([mfcc, delta])  # [30, T]
    mean = np.mean(feat, axis=1)
    std = np.std(feat, axis=1)
    return np.concatenate([mean, std])  # → [60]

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
            if wav_files:
                wav_files_list.append(sorted(wav_files))
                user_ids.append(user_folder)
    return wav_files_list, user_ids

# === 训练集加载 ===
train_path = "../dataSet_wav_1epoch"
train_list, train_ids = load_wav_files_from_dataset(train_path)

X_train, y_train = [], []
for idx, file_list in enumerate(train_list):
    for file in file_list:
        feat = extract_features_with_mfcc(file)
        X_train.append(feat)
        y_train.append(train_ids[idx])

# === 标准化 ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# === KNN 训练 ===
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# === 测试集加载 ===
test_path = "../testSet_wav_1epoch"
test_list, test_ids = load_wav_files_from_dataset(test_path)

X_test, y_test = [], []
for idx, file_list in enumerate(test_list):
    for file in file_list:
        feat = extract_features_with_mfcc(file)
        X_test.append(feat)
        y_test.append(test_ids[idx])

X_test = scaler.transform(X_test)
preds = knn.predict(X_test)

# === 结果展示 ===
correct = sum([p == t for p, t in zip(preds, y_test)])
print(f"✅ 正确预测数：{correct} / {len(y_test)}")
print("\n🎯 分类报告：")
print(classification_report(y_test, preds))
print("\n📊 混淆矩阵：")
print(confusion_matrix(y_test, preds))
