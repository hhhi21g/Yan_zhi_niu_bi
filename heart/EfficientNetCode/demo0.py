import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, KFold
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import EfficientNetForImageClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 处理Mel频谱图
def extract_mel_spectrogram(file_path, sr=96000, n_mels=128, fmax=8000):
    y, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_image_path = file_path.replace('.wav', '.jpg')

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(mel_image_path)
    plt.close()

    return mel_image_path


# 获取所有音频文件的路径和标签
def get_all_data(audio_dir):
    audio_files = []
    category_folders = sorted(os.listdir(audio_dir))

    for label, user_folder in enumerate(category_folders):  # 使用枚举获取标签
        user_folder_path = os.path.join(audio_dir, user_folder)
        if os.path.isdir(user_folder_path):
            wav_files = [f for f in os.listdir(user_folder_path) if f.endswith(".wav")]
            wav_files.sort()  # 确保文件顺序一致
            for file in wav_files:
                audio_files.append((os.path.join(user_folder_path, file), label))

    return audio_files


class HeartSoundDataset(Dataset):
    def __init__(self, audio_files, transform=None, is_test=False):
        self.audio_files = audio_files
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path, label = self.audio_files[idx]
        mel_image_path = extract_mel_spectrogram(audio_path)
        image = Image.open(mel_image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


# 数据转换（确保输入格式适配 EfficientNet）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

audio_dir = '..//data'
audio_files = get_all_data(audio_dir)

train_data, test_data = train_test_split(audio_files, test_size=0.2, random_state=42)

# 创建模型、优化器和损失函数
model = EfficientNetForImageClassification.from_pretrained('E:\\Xinan\\XinAnBei\\heart\\EfficientNet')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# 初始化学习率调度器
# lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


# 早停回调
class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode='min', verbose=False):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.best_model = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif (self.mode == 'min' and val_loss < self.best_score - self.delta) or \
                (self.mode == 'max' and val_loss > self.best_score + self.delta):
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.verbose:
            print(f"Validation loss decreased. Saving model...")
        self.best_model = model.state_dict()


def train(model, train_dataloader, val_dataloader, optimizer, criterion, lr_scheduler, epochs=10, patience=5):
    early_stopping = EarlyStopping(patience=patience, mode='min', verbose=True)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")

        val_loss = validate(model, val_dataloader, criterion)
        early_stopping(val_loss, model)

        # 学习率调整
        lr_scheduler.step(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training...")
            model.load_state_dict(early_stopping.best_model)
            break


def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


# 测试模型：加载保存的最优模型并进行测试
def test_model(test_dataloader, model, device):
    correct = 0
    total = 0

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算，减少内存消耗
        for images, labels in tqdm(test_dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 获取模型输出
            _, predicted = torch.max(outputs.logits, 1)  # 获取预测结果
            total += labels.size(0)  # 累加样本总数
            correct += (predicted == labels).sum().item()  # 计算预测正确的数量

    accuracy = 100 * correct / total
    return accuracy


# 获取测试集
def get_test_data(transform=None):
    # 通过 is_test=True 只加载测试集
    test_dataset = HeartSoundDataset(test_data, transform=transform, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    return test_dataloader


# 加载测试数据集
test_dataloader = get_test_data( transform=transform)


# K折交叉验证
def cross_validation(train_data, test_data, n_splits=5, epochs=70, patience=7):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"Fold {fold + 1}/{n_splits}")

        # 获取当前fold的训练集和验证集
        train_fold = [train_data[i] for i in train_idx]
        val_fold = [train_data[i] for i in val_idx]

        # 这里传入的是训练集和验证集
        train_dataset = HeartSoundDataset(train_fold, transform=transform, is_test=False)
        val_dataset = HeartSoundDataset(val_fold, transform=transform, is_test=False)

        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 初始化模型、优化器等
        model = EfficientNetForImageClassification.from_pretrained('E:\\Xinan\\XinAnBei\\heart\\EfficientNet')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

        lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # 训练模型
        train(model, train_dataloader, val_dataloader, optimizer, criterion, lr_scheduler, epochs, patience)

        # 评估模型
        accuracy = evaluate(model, val_dataloader)
        fold_accuracies.append(accuracy)
        break

    average_accuracy = np.mean(fold_accuracies)
    print(f"Average accuracy across all folds: {average_accuracy:.2f}%")

    # 训练完成后，使用测试集评估模型
    test_dataset = HeartSoundDataset(test_data, transform=transform, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_accuracy = test_model(test_dataloader, model, device)
    print(f"Test accuracy: {test_accuracy:.2f}%")


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# 开始训练和 K 折交叉验证
cross_validation(train_data, test_data, n_splits=5, epochs=70, patience=7)

# 保存模型
torch.save(model.state_dict(), "EfficientNet_model.pth")
