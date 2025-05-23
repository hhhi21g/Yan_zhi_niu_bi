import os
import librosa
import numpy as np
import pywt
import pandas as pd
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns


def polycoherence_0d(data, fs, f1=40, f2=80):
    f, t, spec = spectrogram(data, fs=fs, mode='complex', nperseg=256, noverlap=128)
    ind1 = np.argmin(np.abs(f - f1))
    ind2 = np.argmin(np.abs(f - f2))
    ind_sum = np.argmin(np.abs(f - (f1 + f2)))
    spec = np.transpose(spec, [1, 0])
    p1 = spec[:, ind1] * spec[:, ind2]
    p2 = np.conjugate(spec[:, ind_sum])
    coh = np.mean(p1 * p2)
    return np.abs(coh)


def extract_polycoherence_features(y, sr):
    freqs = [(30, 60), (40, 80), (50, 70)]
    return np.array([polycoherence_0d(y, sr, f1, f2) for f1, f2 in freqs])


def preprocess_audio(filepath, target_sr=8000):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y_resampled, target_sr


def extract_mfcc(y, sr, n_mfcc=19, frame_length=240, hop_length=80):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=frame_length, hop_length=hop_length)
    return np.mean(mfcc.T, axis=0)


def extract_dwt(y, wavelet='db4', level=3):
    coeffs = pywt.wavedec(y, wavelet, level=level)
    feature = []
    for c in coeffs:
        feature.append(np.mean(c))
        feature.append(np.std(c))
    return np.array(feature)[:24]


def extract_all_features(audio_path):
    y, sr = preprocess_audio(audio_path)
    mfcc_feat = extract_mfcc(y, sr)
    dwt_feat = extract_dwt(y)
    poly_feat = extract_polycoherence_features(y, sr)
    return np.concatenate((mfcc_feat, dwt_feat, poly_feat))  # 46维


def run_knn_with_polycoherence(data_dir):
    filepaths, labels = [], []
    classes = sorted(os.listdir(data_dir))
    for label in classes:
        class_path = os.path.join(data_dir, label)
        if os.path.isdir(class_path):
            for fname in sorted(os.listdir(class_path)):
                if fname.lower().endswith('.wav'):
                    filepaths.append(os.path.join(class_path, fname))
                    labels.append(label)

    X = [extract_all_features(f) for f in filepaths]
    X, labels = shuffle(np.array(X), np.array(labels), random_state=42)
    y = LabelEncoder().fit_transform(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_list, f1_list = [], []
    confusion_matrices = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = KNeighborsClassifier(n_neighbors=13, metric='manhattan')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred, average='macro'))

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

        print(f"Fold {i + 1} 标签分布: {np.unique(y_test, return_counts=True)}")
        print(f"Fold {i + 1} 混淆矩阵:{cm}\n")

    result_df = pd.DataFrame({
        "Accuracy": acc_list,
        "F1-Score": f1_list
    })
    result_df.to_csv("knn_results_with_polycoherence_analysis.csv", index=False)

    print("📊 KNN + MFCC + DWT + Polycoherence（分析版）")
    print(f"✅ 平均准确率: {np.mean(acc_list):.4f}")
    print(f"✅ 平均F1-score: {np.mean(f1_list):.4f}")

    # 准确率与F1变化图
    plt.figure()
    plt.plot(acc_list, label="Accuracy")
    plt.plot(f1_list, label="F1 Score")
    plt.title("KFold Accuracy & F1")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("knn_accuracy_f1_analysis.png")
    plt.close()

    # 绘制最后一折混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrices[-1], annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Last Fold)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("knn_confusion_matrix_last_fold.png")
    plt.close()


if __name__ == "__main__":
    run_knn_with_polycoherence("..\\dataSet_wav_1epoch")  # 修改为你的数据路径
