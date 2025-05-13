import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import os
import sys
sys.path.append("E:/Xinan/XinAnBei/heart")  # 添加KNN的上一级路径

from KNN.knn_demo0 import wav_files_list, user_ids, new_list


# ========== 引入 KNN 提取的音频路径 ==========
# ========== 数据增强与特征提取函数 ==========
def extract_mfcc_image(y, sr, n_mfcc=40, hop_length=512, n_fft=2048):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfcc

def add_noise(y, noise_factor=0.001):
    noise = np.random.randn(len(y))
    return (y + noise_factor * noise).astype(np.float32)

def time_stretch(y, rate=1.05):
    try:
        return librosa.effects.time_stretch(y=y, rate=rate)
    except:
        return y  # 若拉伸失败，则返回原始音频

# ========== 构造训练数据 ==========
X = []
y = []

for user_idx, user_files in enumerate(wav_files_list):
    for file in user_files:
        y_raw, sr = librosa.load(file, sr=None)
        for y_aug in [y_raw, add_noise(y_raw), time_stretch(y_raw)]:
            try:
                mfcc = extract_mfcc_image(y_aug, sr)
                mfcc = librosa.util.fix_length(mfcc, size=100, axis=1)
                X.append(mfcc[np.newaxis, ...])
                y.append(user_idx)
            except:
                continue

X = np.array(X, dtype=np.float32)
y = np.array(y)

# ========== 定义 CNN 模型 ==========
class CNNVoiceNet(nn.Module):
    def __init__(self, num_classes):
        super(CNNVoiceNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        dummy_input = torch.zeros(1, 1, 40, 100)  # shape: (B, C, n_mfcc, time)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 40, 100)  # B, C, H, W
            out = self.pool2(self.conv2(self.pool1(self.conv1(dummy))))
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# ========== 模型训练 ==========
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = CNNVoiceNet(num_classes=len(user_ids))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-5,weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# ========== 测试阶段 ==========
model.eval()
correct = 0

for i, user_files in enumerate(new_list):
    for file in user_files:
        y_test, sr_test = librosa.load(file, sr=None)
        try:
            mfcc_test = extract_mfcc_image(y_test, sr_test)
            mfcc_test = librosa.util.fix_length(mfcc_test, size=100, axis=1)
            input_tensor = torch.tensor(mfcc_test[np.newaxis, np.newaxis, ...], dtype=torch.float32).to(device)
            pred = model(input_tensor).argmax().item()
            if pred == i:
                correct += 1
        except:
            continue

print("识别正确数量：", correct)
