import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 扩散模型参数
T = 1000
beta_start = 1e-4
beta_end = 0.02
device = torch.device('cuda'  if torch.cuda.is_available()  else 'cpu')
beta_schedule = torch.linspace(beta_start,  beta_end, T, dtype=torch.float32).to(device)
alpha_schedule = 1. - beta_schedule
alpha_bar_schedule = torch.cumprod(alpha_schedule, dim=0)


class weighted_mse_loss(nn.Module):
    def __init__(self, feature_weights_dict):  # 只保留特征权重
        super().__init__()
        # 确保权重顺序与数据特征严格对应
        self.feature_order = ['magnitude', 'latitude', 'longitude', 'delta_time']
        self.weights = torch.tensor(
            [feature_weights_dict[k] for k in self.feature_order],
            dtype=torch.float32
        )

    def forward(self, pred, target):
        # 特征维度加权计算
        loss = (pred[:,0:4] - target[:,0:4]) ** 2
        loss = loss * self.weights.to(loss.device)
        return loss.mean()

class TransformerConditionEncoder(nn.Module):
    """基于Transformer的条件编码器"""

    def __init__(self, input_dim, num_layers=4, nhead=8, dim_feedforward=256):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, input_dim))  # 可学习位置编码

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,

        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

    def forward(self, x):
        # 修正维度顺序调整
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 聚合时序维度
        return self.projection(x)


class TransformerDiffusionModel(nn.Module):
    def __init__(self, input_dim, cond_steps, cond_dim, device):  # 新增device参数
        super().__init__()
        self.device = device  # 存储设备信息

        # 所有层自动初始化为指定设备
        self.cond_encoder = TransformerConditionEncoder(
            input_dim=cond_dim,  # 例如 cond_dim=4
            num_layers=4,
            nhead=4,
            dim_feedforward=256
        ).to(device)
        self.time_embed = nn.Sequential(
            nn.Embedding(T, 64).to(device),
            nn.Linear(64, 64).to(device),
            nn.SiLU()
        )


        # 噪声预测网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim + 64 + 64, 256),  # 输入+条件+时间
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, cond_input, y_input, t_input):
        # 显式设备同步
        cond_input = cond_input.to(self.device)
        y_input = y_input.to(self.device)
        t_input = t_input.to(self.device)

        # 条件编码
        cond_vec = self.cond_encoder(cond_input)

        # 时间嵌入
        t_emb = self.time_embed(t_input)

        # 合并输入
        combined = torch.cat([y_input, cond_vec, t_emb], dim=1)
        return self.noise_predictor(combined)


class EarthquakeDataset(Dataset):
    def __init__(self, data, input_features, output_features, time_steps=12):
        self.data = data
        self.time_steps = time_steps
        self.input_features = input_features
        self.output_features = output_features

        # 初始化归一化器
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])

        # 生成样本索引
        self.indices = np.arange(len(self.scaled_X) - time_steps)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        X = torch.tensor(self.scaled_X[i:i + self.time_steps], dtype=torch.float32)
        y = torch.tensor(self.scaled_y[i + self.time_steps], dtype=torch.float32)
        return X, y


def train_diffusion_model(model, dataloader, feature_weights_dict, epochs=100, device='cuda'):
    """训练扩散模型的核心函数（已修复特征权重问题）"""

    # 初始化组件
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = weighted_mse_loss(feature_weights_dict).to(device)  # 关键修改点

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            # 数据加载到设备
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = X_batch.size(0)

            # 随机时间步生成
            t = torch.randint(0, T, (batch_size,), device=device)

            # 噪声生成（与y_batch同设备）
            noise = torch.randn_like(y_batch, device=device)

            # 前向扩散计算
            alpha_bar_t = alpha_bar_schedule[t].view(-1, 1)
            y_noisy = torch.sqrt(alpha_bar_t) * y_batch + torch.sqrt(1 - alpha_bar_t) * noise

            # 梯度清零
            optimizer.zero_grad()

            # 前向预测
            pred_noise = model(X_batch, y_noisy, t)

            # 计算损失（仅特征加权）
            loss = loss_fn(pred_noise, noise)  # 简化调用方式

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 添加梯度裁剪
            optimizer.step()

            # 累计损失
            total_loss += loss.item() * batch_size

            # 调度器更新
        scheduler.step()

        # 打印日志
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    return model


@torch.no_grad()
def generate_prediction(model, cond_input, steps=200, device='cuda'):
    model.eval()
    cond_input = cond_input.to(device)
    y_t = torch.randn(1, cond_input.size(-1), device=device)

    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t], device=device)

        # 预测噪声
        pred_noise = model(cond_input, y_t, t_tensor)

        # 更新参数
        beta_t = beta_schedule[t]
        alpha_t = alpha_schedule[t]
        alpha_bar_t = alpha_bar_schedule[t]

        # 反向扩散步骤
        y_t = (y_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(y_t)
            y_t += torch.sqrt(beta_t) * noise

    return y_t.cpu()


# 数据加载函数（保持与原始代码相同）
def load_data_from_folder(folder_path):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    full_data = []
    prev_last_time = None

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            df = df.dropna()
            # 时间解析增强
            if 'time' not in df.columns:
                print(f"文件 {file_name} 缺少'time'列，跳过处理")
                continue

            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time')

            # 跨文件时间差处理
            if prev_last_time is not None:
                time_diff = (df.iloc[0]['time'] - prev_last_time).total_seconds() / 3600
                df.loc[df.index[0], 'delta_time'] = time_diff

            # 文件内时间差计算
            df['delta_time'] = df['time'].diff().dt.total_seconds().fillna(0) / 3600
            prev_last_time = df.iloc[-1]['time']

            full_data.append(df)
        except Exception as e:
            print(f"加载 {file_path} 失败: {str(e)}")

    if not full_data:
        return None
    return pd.concat(full_data).sort_values('time')

def prepare_dataset(data, input_features, output_features, time_steps=12):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        scaled_X = scaler_X.fit_transform(data[input_features]).astype(np.float32)
        scaled_y = scaler_y.fit_transform(data[output_features]).astype(np.float32)

        X, y = [], []
        for i in range(len(scaled_X) - time_steps):
            X.append(scaled_X[i:i + time_steps])
            y.append(scaled_y[i + time_steps])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler_X, scaler_y



if __name__ == "__main__":
    # 配置参数
    DATA_FOLDER = r"C:\Users\86150\Desktop\dataset\25 years\train\4"
    TIME_STEPS = 96
    EPOCHS = 200
    BATCH_SIZE = 64
    TARGET_DATE = pd.to_datetime('2022-03-01')
    input_features = ['mag', 'latitude', 'longitude', 'delta_time',
                     ]
    output_features = ['mag', 'latitude', 'longitude', 'delta_time',
                    ]

    # 加载数据
    processed_data = load_data_from_folder(DATA_FOLDER)

    # 准备数据集
    dataset = EarthquakeDataset(
        processed_data,
        input_features,
        output_features,
        TIME_STEPS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerDiffusionModel(
        input_dim=len(output_features),
        cond_steps=TIME_STEPS,
        cond_dim=len(input_features),
        device=device  # 传入设备参数
    ).to(device)  # 二次确认迁移
    # 训练模型
    feature_weights = {
        'magnitude': 1.0,  # 震级
        'latitude': 3.0,  # 纬度
        'longitude': 3.0,  # 经度
        'delta_time': 1.0  # 时间差
    }

    train_diffusion_model(
        model,
        dataloader,
        feature_weights_dict=feature_weights,  # 传入字典
        epochs=EPOCHS,
        device=device
    )

    # 预测初始化
    last_sequence = torch.tensor(
        dataset.scaled_X[-TIME_STEPS:],
        dtype=torch.float32
    ).unsqueeze(0).to(device)  # (1, seq_len, features)
    last_timestamp = processed_data['time'].iloc[-1]
    results = []

    # 预测循环
    while True:
        y_pred_norm = generate_prediction(model, last_sequence, device=device)
        y_pred = dataset.scaler_y.inverse_transform(y_pred_norm.numpy())[0]

        delta_hours = y_pred[3]
        new_time = last_timestamp + pd.Timedelta(hours=delta_hours)

        if new_time > TARGET_DATE:
            break

        results.append({
            'time': new_time.strftime('%Y-%m-%d %H:%M:%S'),
            'magnitude': y_pred[0],
            'latitude': y_pred[1],
            'longitude': y_pred[2],
            'delta_time': delta_hours
        })

        # 生成新输入
        new_input = dataset.scaler_X.transform([y_pred])
        new_tensor = torch.tensor(new_input, dtype=torch.float32).to(device)

        # 更新序列
        last_sequence = torch.cat(
            [last_sequence[:, 1:, :],
             new_tensor.unsqueeze(1)],
            dim=1
        )
        last_timestamp = new_time

    # 保存结果
    result_df = pd.DataFrame(results)
    output_path = os.path.join(DATA_FOLDER, 'torch_diffusion_predictions.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存至：{output_path}")