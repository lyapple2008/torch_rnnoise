import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

# 自定义损失函数
def my_crossentropy(y_true, y_pred):
    return torch.mean(2 * torch.abs(y_true - 0.5) * nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none'), dim=-1)

def mymask(y_true):
    return torch.min(y_true + 1., torch.tensor(1.))

def msse(y_true, y_pred):
    return torch.mean(mymask(y_true) * torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true)), dim=-1)

def mycost(y_true, y_pred):
    return torch.mean(mymask(y_true) * (10 * torch.square(torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true))) 
                                        + torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true)) 
                                        + 0.01 * nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')), dim=-1)

# 定义模型
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.input_dense = nn.Linear(42, 24)
        self.vad_gru = nn.GRU(24, 24, batch_first=True)
        self.vad_output = nn.Linear(24, 1)
        self.noise_gru = nn.GRU(42 + 24 + 24, 48, batch_first=True)
        self.denoise_gru = nn.GRU(42 + 24 + 48, 96, batch_first=True)
        self.denoise_output = nn.Linear(96, 22)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        tmp = self.tanh(self.input_dense(x))
        vad_gru_output, _ = self.vad_gru(tmp)
        vad_output = self.sigmoid(self.vad_output(vad_gru_output))
        noise_input = torch.cat([tmp, vad_gru_output, x], dim=-1)
        noise_gru_output, _ = self.noise_gru(noise_input)
        denoise_input = torch.cat([vad_gru_output, noise_gru_output, x], dim=-1)
        denoise_gru_output, _ = self.denoise_gru(denoise_input)
        denoise_output = self.sigmoid(self.denoise_output(denoise_gru_output))
        return denoise_output, vad_output
class FeatureDataset(Dataset):
    def __init__(self, file_path, nb_features, nb_bands, window_size):
        """
        初始化数据集类。

        :param file_path: 存储特征数据的二进制文件路径
        :param nb_features: 特征的数量
        :param nb_bands: 频段的数量
        :param window_size: 窗口大小
        """
        self.file_path = file_path
        self.nb_features = nb_features
        self.nb_bands = nb_bands
        self.window_size = window_size

        # 读取二进制文件
        self.data = np.fromfile(file_path, dtype=np.float32)

        # 计算样本数量
        total_features = nb_features + 2 * nb_bands + 1
        self.nb_sequences = len(self.data) // (total_features * window_size)

        # 重塑数据
        self.data = self.data[:self.nb_sequences * total_features * window_size]
        self.data = self.data.reshape(self.nb_sequences, window_size, total_features)

    def __len__(self):
        """
        返回数据集的样本数量。

        :return: 样本数量
        """
        return self.nb_sequences

    def __getitem__(self, idx):
        """
        根据索引获取单个样本。

        :param idx: 样本索引
        :return: 输入特征、目标数据和 VAD 数据的张量
        """
        sample = self.data[idx]
        x = torch.tensor(sample[:, :self.nb_features], dtype=torch.float32)
        y = torch.tensor(sample[:, self.nb_features:self.nb_features + self.nb_bands], dtype=torch.float32)
        vad = torch.tensor(sample[:, -1:], dtype=torch.float32)
        return x, y, vad

# 初始化模型、优化器和损失函数
model = RNNModel()
optimizer = optim.Adam(model.parameters())
loss_weights = [10, 0.5]

# 加载数据
nb_features = 42
nb_bands = 22
window_size = 2000
file_path = '/Volumes/tiger/Workspace/side-projects/2501-ains/torch_rnnoise/03_rnnoise/src/training_10000.f32'  # 替换为实际的特征文件路径
dataset = FeatureDataset(file_path, nb_features, nb_bands, window_size)

# 划分训练集和验证集
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
epochs = 120
print('Train...')
for epoch in range(epochs):
    model.train()
    for x, y, vad in train_loader:
        optimizer.zero_grad()
        denoise_output, vad_output = model(x)
        loss_denoise = mycost(y, denoise_output)
        loss_vad = my_crossentropy(vad, vad_output)
        total_loss = loss_weights[0] * torch.mean(loss_denoise) + loss_weights[1] * torch.mean(loss_vad)
        total_loss.backward()
        optimizer.step()

    # 验证集上的损失
    model.eval()
    total_loss_val = 0
    with torch.no_grad():
        for x, y, vad in val_loader:
            denoise_output_val, vad_output_val = model(x)
            loss_denoise_val = mycost(y, denoise_output_val)
            loss_vad_val = my_crossentropy(vad, vad_output_val)
            total_loss_val += loss_weights[0] * torch.mean(loss_denoise_val) + loss_weights[1] * torch.mean(loss_vad_val)
    total_loss_val /= len(val_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss.item()}, Val Loss: {total_loss_val.item()}')

# 保存模型
torch.save(model.state_dict(), "weights.pth")