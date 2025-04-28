import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EfficientNetForImageClassification
from torchvision import transforms

from EfficientNetCode.demo0 import HeartSoundDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_dir = '..//data'

# 数据转换（确保输入格式适配 EfficientNet）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit EfficientNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizing based on ImageNet stats
])


def get_test_data(audio_dir, transform=None):
    # 通过 is_test_set=True 只加载测试集
    test_dataset = HeartSoundDataset(audio_dir, transform=transform, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    return test_dataloader


# 加载测试数据集
test_dataset = get_test_data(audio_dir=audio_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 加载测试图像并进行推理
model = EfficientNetForImageClassification.from_pretrained('E:\\Xinan\\XinAnBei\\heart\\EfficientNet')
model.load_state_dict(torch.load("EfficientNet_model.pth"))
model.to(device)
model.eval()

# test_audio_path = "..\\data\\survey1\\record_generate4_25.wav"
# test_image_path = extract_mel_spectrogram(test_audio_path)
# test_image = Image.open(test_image_path)
# test_image = transform(test_image).unsqueeze(0)  # Add batch dimension

# test_image = test_image.to(device)

correct = 0
total = 0

# 加载测试数据集
test_dataset = get_test_data(audio_dir=audio_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 加载测试图像并进行推理
model = EfficientNetForImageClassification.from_pretrained('E:\\Xinan\\XinAnBei\\heart\\EfficientNet')
model.load_state_dict(torch.load("EfficientNet_model.pth"))
model.to(device)
model.eval()

# 遍历测试集进行预测
with torch.no_grad():  # 关闭梯度计算，减少内存消耗
    for images, labels in tqdm(test_dataloader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # 获取模型输出
        _, predicted = torch.max(outputs.logits, 1)  # 获取预测结果
        total += labels.size(0)  # 累加样本总数
        correct += (predicted == labels).sum().item()  # 计算预测正确的数量

# 计算准确率
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
