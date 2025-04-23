import os
import torch
import torchaudio
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2ForSequenceClassification
from sklearn.model_selection import KFold

kFold = KFold(n_splits=6, shuffle=True, random_state=42)

model_name = 'E:\\Xinan\\XinAnBei\\heart\\wav2vec2-base-960'

processor = Wav2Vec2Processor.from_pretrained(model_name)


class AudioFolderDataset(Dataset):
    def __init__(self, audio_dir, processor, sample_rate=16000, start_frame=2500, end_frame=12500):
        """
        初始化数据集类，使用最大长度截断音频

        :param audio_dir: 存放音频文件夹的目录，包含多个子文件夹，每个子文件夹是一个类别
        :param processor: Wav2Vec2Processor，用于处理音频文件
        :param sample_rate: 采样率，默认为16000
        :param start_frame: 截取的起始帧
        :param end_frame: 截取的结束帧
        """
        self.audio_dir = audio_dir
        self.processor = processor
        self.sample_rate = sample_rate
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.classes = os.listdir(audio_dir)
        self.file_paths = []
        self.labels = []

        for label_idx, label in enumerate(self.classes):
            label_folder = os.path.join(audio_dir, label)
            if os.path.isdir(label_folder):
                for file_name in os.listdir(label_folder):
                    if file_name.endswith(".wav"):
                        file_path = os.path.join(label_folder, file_name)
                        self.file_paths.append(file_path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 获取音频文件路径和标签
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # 加载音频文件
        waveform, original_sample_rate = torchaudio.load(audio_path)

        # 如果采样率不匹配，则进行重采样
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)(
                waveform)

        # 截取音频帧范围，从 start_frame 到 end_frame
        waveform = waveform[:, self.start_frame:self.end_frame]

        # 使用processor处理音频数据
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, padding=False, return_tensors="pt")

        return inputs, label


# 设置数据集和数据加载器
audio_dir = "..\\data"
dataset = AudioFolderDataset(audio_dir, processor)

batch_size = 4
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 记录每个折叠的准确率
fold_accuracies = []

# K折交叉验证
for fold, (train_idx, val_idx) in enumerate(kFold.split(dataset)):
    print(f"Training fold {fold + 1}/{kFold.get_n_splits()}...")

    # 创建训练和验证数据集
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    # 创建 DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs, labels = batch
            input_values = inputs.input_values.squeeze().to(device)  # 获取音频特征
            labels = labels.to(device)  # 标签

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(input_values=input_values, labels=labels)

            # 计算损失
            loss = outputs.loss
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # 模型评估
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    # 遍历验证集
    for batch in val_loader:
        inputs, labels = batch
        input_values = inputs.input_values.squeeze().to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_values=input_values)

        predicted_labels = torch.argmax(outputs.logits, dim=-1)

        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy for fold {fold + 1}: {accuracy:.4f}")
    fold_accuracies.append(accuracy)

# 输出所有折叠的准确率及其平均值
print(f"Average accuracy: {sum(fold_accuracies) / len(fold_accuracies):.4f}")
