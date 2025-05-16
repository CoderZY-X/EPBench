import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# 自定义加权MSE损失函数
def weighted_mse_loss(y_pred, y_true, feature_weights, sample_weights):
    squared_error = (y_true - y_pred) ** 2
    weighted_error = squared_error * feature_weights
    per_sample_loss = weighted_error.mean(dim=1)
    total_loss = (per_sample_loss * sample_weights).mean()
    return total_loss


# 自定义数据集类
class EarthquakeDataset(Dataset):
    def __init__(self, data, input_features, output_features, time_steps=12):
        self.data = data
        self.input_features = input_features
        self.output_features = output_features
        self.time_steps = time_steps

        # 初始化归一化器
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])

        # 生成样本索引和权重
        self.sample_indices = np.arange(len(self.scaled_X) - time_steps)
        self.sample_weights = np.array([
            10.0 if data.iloc[i + time_steps]['magnitude'] >= 5 else 1.0
            for i in self.sample_indices
        ])

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        i = self.sample_indices[idx]
        X = self.scaled_X[i:i + self.time_steps]  # (time_steps, features)
        y = self.scaled_y[i + self.time_steps]  # (features,)
        weight = self.sample_weights[idx]

        # 调整维度为PyTorch格式 (features, time_steps)
        X = torch.tensor(X.T, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        weight = torch.tensor(weight, dtype=torch.float32)

        return X, y, weight


# 定义CNN模型
class EarthquakeCNN(nn.Module):
    def __init__(self, input_features, output_features, time_steps):
        super(EarthquakeCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_features, 512, 7, padding=15, dilation=5),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv1d(512, 256, 5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 3
            nn.Conv1d(256, 128, 5, padding=6, dilation=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 4
            nn.Conv1d(128, 128, 3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 5
            nn.Conv1d(128, 128, 3, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

        # 计算最终特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, input_features, time_steps)
            dummy = self.features(dummy)
            flattened_size = dummy.view(1, -1).shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_features)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# 数据加载函数（保持不变）
def load_data_from_folder(folder_path):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    full_data = []
    prev_last_time = None

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()

            # 时间处理（保持原有增强处理）
            df['time'] = df['time'].str.replace(r'\s+', ' ', regex=True)
            df['time'] = df['time'].str.replace(r'\.\d+$', '', regex=True)
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time')
            # 修复1：先计算文件内时间差
            df['delta_time'] = df['time'].diff().dt.total_seconds() / 3600

            # 修复2：处理跨文件时间差（覆盖第一条记录）
            if prev_last_time is not None:
                time_diff = (df.iloc[0]['time'] - prev_last_time).total_seconds() / 3600
                df.at[df.index[0], 'delta_time'] = time_diff
            # 修复3：保留有效数据
            valid_df = df.dropna(subset=['delta_time'])
            # print(df['delta_time'])
            full_data.append(valid_df)
            prev_last_time = df['time'].iloc[-1]  # 使用原始df的最后时间
        except Exception as e:
            print(f"加载 {file_path} 失败: {str(e)}")

    # 修复4：保持原始文件顺序，不重新排序
    combined_df = pd.concat(full_data)
    return combined_df


def prepare_dataset(data, input_features, output_features, time_steps=12):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_X = scaler_X.fit_transform(data[input_features])
    scaled_y = scaler_y.fit_transform(data[output_features])

    X, y = [], []
    for i in range(len(scaled_X) - time_steps):
        X.append(scaled_X[i:i + time_steps])
        y.append(scaled_y[i + time_steps])

    return np.array(X), np.array(y), scaler_X, scaler_y



# ...（与原始代码相同）...

if __name__ == "__main__":
    # 参数设置
    DATA_FOLDER = r"/root/autodl-tmp/EP/train1/1"
    TIME_STEPS = 96
    EPOCHS = 400
    BATCH_SIZE = 32
    TARGET_DATE = pd.to_datetime('1995-04-01')

    input_features = ['magnitude', 'latitude', 'longitude', 'delta_time']
    output_features = ['magnitude', 'latitude', 'longitude', 'delta_time']

    # 加载数据
    processed_data = load_data_from_folder(DATA_FOLDER)

    # 创建数据集和数据加载器
    dataset = EarthquakeDataset(processed_data, input_features, output_features, TIME_STEPS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EarthquakeCNN(
        input_features=len(input_features),
        output_features=len(output_features),
        time_steps=TIME_STEPS
    ).to(device)

    # 定义优化器和特征权重
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    feature_weights = torch.tensor([1.0, 3.0, 3.0, 1.0], device=device)

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch, weights_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            weights_batch = weights_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = weighted_mse_loss(outputs, y_batch, feature_weights, weights_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataset):.4f}')

    # 预测初始化
    model.eval()
    last_sequence = torch.tensor(dataset.scaled_X[-TIME_STEPS:].T,  # (features, time_steps)
                                 dtype=torch.float32).unsqueeze(0).to(device)
    last_timestamp = processed_data['time'].iloc[-1]
    results = []

    # 预测循环
    with torch.no_grad():
        while True:
            pred = model(last_sequence)
            pred_np = pred.cpu().numpy()
            pred_denorm = dataset.scaler_y.inverse_transform(pred_np)[0]

            delta_hours = pred_denorm[3]
            new_time = last_timestamp + pd.Timedelta(hours=delta_hours)

            if new_time > TARGET_DATE:
                break

            # 记录结果
            results.append({
                'time': new_time.strftime('%Y-%m-%d %H:%M:%S'),
                'magnitude': pred_denorm[0],
                'latitude': pred_denorm[1],
                'longitude': pred_denorm[2],
                'delta_time': delta_hours
            })

            # 生成新输入
            new_input = dataset.scaler_X.transform([pred_denorm])
            new_tensor = torch.tensor(new_input.T,  # (features, 1)
                                      dtype=torch.float32).unsqueeze(0).to(device)

            # 更新序列
            last_sequence = torch.cat([last_sequence[:, :, 1:], new_tensor], dim=2)
            last_timestamp = new_time

    # 保存结果
    result_df = pd.DataFrame(results)
    output_path = os.path.join(r"/root/autodl-tmp/EP", 'torch_predictions-4.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存至：{output_path}")