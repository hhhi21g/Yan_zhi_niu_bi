import numpy as np
np.complex = complex  # 兼容 librosa 与 numpy

import os
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from typing import List, Tuple
from tqdm import tqdm
from sklearn.ensemble import VotingClassifier
from collections import Counter

# ======================== 参数配置 ========================
N_MFCC = 15
HOP_LENGTH = 480
N_FFT = 2048
USE_SVD = True
DROP_FIRST_N_COMPONENTS = 0
SVD_VAR_RETAIN = 0.9
TRAIN_PATH = "../dataSet_wav_1epoch"
N_SPLITS = 5
# ==========================================================

def extract_features_mfcc(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return mfcc.T

def load_dataset(path: str) -> Tuple[List[np.ndarray], List[str]]:
    features, labels = [], []
    for user in sorted(os.listdir(path)):
        user_path = os.path.join(path, user)
        if os.path.isdir(user_path):
            for fname in sorted(os.listdir(user_path)):
                if fname.lower().endswith('.wav'):
                    full_path = os.path.join(user_path, fname)
                    feat = extract_features_mfcc(full_path)
                    features.append(feat)
                    labels.append(user)
    return features, labels

def apply_svd(all_frames: List[np.ndarray], drop_n: int, retain_var: float):
    full_matrix = np.vstack(all_frames)
    mean_vec = np.mean(full_matrix, axis=0)
    centered = full_matrix - mean_vec
    U, sigma, VT = np.linalg.svd(centered, full_matrices=False)
    explained = (sigma ** 2) / np.sum(sigma ** 2)

    selected_indices = []
    acc_var = 0.0
    for i in range(drop_n, len(explained)):
        selected_indices.append(i)
        acc_var += explained[i]
        if acc_var >= retain_var:
            break

    return mean_vec, VT[selected_indices, :]

def project_features(features: List[np.ndarray], mean_vec: np.ndarray, VT_sub: np.ndarray) -> np.ndarray:
    projected = []
    for mfcc_seq in features:
        centered = mfcc_seq - mean_vec
        reduced = centered @ VT_sub.T
        projected.append(np.mean(reduced, axis=0))
    return np.array(projected)

def train_svm_with_cv(X: np.ndarray, y: np.ndarray) -> Tuple[SVC, StandardScaler, dict]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {
        'C': [100],
        'kernel': ['rbf'],
        'decision_function_shape': ['ovr']
    }

    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_scaled, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, scaler, best_params

# ======================== K折交叉验证主流程 ========================
if __name__ == "__main__":
    print("📥 正在加载音频特征数据...")
    all_feats, all_labels = load_dataset(TRAIN_PATH)

    if USE_SVD:
        print("🔍 正在进行 SVD 降维处理...")
        mean_vec, VT_sub = apply_svd(all_feats, drop_n=DROP_FIRST_N_COMPONENTS, retain_var=SVD_VAR_RETAIN)
        X_all = project_features(all_feats, mean_vec, VT_sub)
    else:
        X_all = np.array([np.mean(feat, axis=0) for feat in all_feats])

    y_all = np.array(all_labels)

    print(f"\n🔁 开始 {N_SPLITS}-折交叉验证...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accuracies = []

    models = []
    scalers = []
    all_preds = []
    all_gts = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_all), total=N_SPLITS, desc="K-Fold"), 1):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        clf, scaler, best_params = train_svm_with_cv(X_train, y_train)
        models.append(clf)
        scalers.append(scaler)

        # 每个模型在验证集上预测
        val_preds = []
        for model, sc in zip(models, scalers):
            pred = model.predict(sc.transform(X_val))
            val_preds.append(pred)

        # 投票融合
        val_preds = np.array(val_preds)
        final_pred = []
        for i in range(val_preds.shape[1]):
            votes = val_preds[:, i]
            majority = Counter(votes).most_common(1)[0][0]
            final_pred.append(majority)

        acc = accuracy_score(y_val, final_pred)
        print(f"✅ 第 {fold} 折融合投票准确率: {acc:.2%}")

        all_preds.extend(final_pred)
        all_gts.extend(y_val)

    final_acc = accuracy_score(all_gts, all_preds)
    print(f"\n🎯 最终融合预测准确率（{N_SPLITS} 折）: {final_acc:.2%}")
