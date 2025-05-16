import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# 增强版加权MSE损失函数（支持双重加权）
def weighted_mse_loss(y_pred, y_true, feature_weights, sample_weights):
    """
    y_pred: (batch_size, output_features)
    y_true: (batch_size, output_features)
    feature_weights: (output_features,)
    sample_weights: (batch_size,)
    """
    squared_error = (y_true[:,0:4] - y_pred[:,0:4]) ** 2  # (batch_size, features)
    feature_weighted = squared_error * feature_weights  # 特征维度加权
    sample_weighted = feature_weighted.mean(dim=1) * sample_weights  # 样本维度加权
    return sample_weighted.mean()


# 改进数据集类（修复索引错误）
class EarthquakeDataset(Dataset):
    def __init__(self, data, input_features, output_features, time_steps=96):
        self.data = data
        self.time_steps = time_steps
        self.input_features = input_features
        self.output_features = output_features

        # 增强归一化处理
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])

        # 修复索引生成逻辑
        self.indices = np.arange(len(self.scaled_X) - time_steps)

        # 增强权重计算
        self.sample_weights = np.array([
            10.0 if data.iloc[i + time_steps]['magnitude'] >= 5 else 1.0
            for i in self.indices
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        X = torch.tensor(self.scaled_X[i:i + self.time_steps], dtype=torch.float32)
        y = torch.tensor(self.scaled_y[i + self.time_steps], dtype=torch.float32)
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return X, y, weight

    # 重构后的LSTM模型（完全对齐Keras结构）


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, output_size, sequence_length=24):  # 添加参数
        super().__init__()
        self.sequence_length = sequence_length

        # LSTM层
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(256)

        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(128)

        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.dropout3 = nn.Dropout(0.1)
        self.bn3 = nn.BatchNorm1d(64)

        self.lstm4 = nn.LSTM(64, 64, batch_first=True)
        self.dropout4 = nn.Dropout(0.1)
        self.bn4 = nn.BatchNorm1d(64)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * self.sequence_length, 64),  # 根据序列长度调整输入
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # 第一层LSTM，输出序列
        x, _ = self.lstm1(x)
        # 保留序列，用于下一层
        x = self.dropout1(self.bn1(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)

        # 第二层LSTM
        x, _ = self.lstm2(x)
        x = self.dropout2(self.bn2(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)

        # 第三层LSTM
        x, _ = self.lstm3(x)
        x = self.dropout3(self.bn3(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)

        # 第四层LSTM
        x, _ = self.lstm4(x)
        x = self.dropout4(self.bn4(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)

        # 展开整个序列作为输入到全连接层
        x = x.reshape(x.shape[0], -1)
        output = self.fc(x)
        return output


def load_data_from_folder(folder_path):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    all_dfs = []

    # 第一步：加载所有文件并预处理
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()

            # 检查必要列
            required_cols = ['time', 'mag', 'latitude', 'longitude']
            if not all(col in df.columns for col in required_cols):
                print(f"文件 {file_path} 缺少必要列，跳过处理")
                continue

            if df.empty:
                print(f"文件 {file_path} 为空，跳过处理")
                continue

            # 时间处理
            df['time'] = df['time'].str.replace(r'\s+', ' ', regex=True)
            df['time'] = df['time'].str.replace(r'\.\d+$', '', regex=True)
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time')
            df= df.dropna()
            all_dfs.append(df)

        except Exception as e:
            print(f"加载 {file_path} 失败: {str(e)}")
            continue

    if not all_dfs:
        print("警告: 没有加载到任何有效数据!")
        return pd.DataFrame(columns=['time', 'mag', 'latitude', 'longitude', 'delta_time'])

    # 第二步：合并所有数据并按时间排序
    combined_df = pd.concat(all_dfs).sort_values('time')

    # 第三步：计算全局连续的时间差
    combined_df['delta_time'] = combined_df['time'].diff().dt.total_seconds() / 3600

    # 删除第一条没有时间差的记录
    combined_df = combined_df.iloc[1:].reset_index(drop=True)

    return combined_df

65
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


class EarthquakeDataset(Dataset):
    def __init__(self, data, input_features, output_features, time_steps=96):
        self.data = data
        self.time_steps = time_steps
        self.input_features = input_features
        self.output_features = output_features

        # 初始化归一化器
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])

        # 生成样本索引和权重
        self.indices = np.arange(len(self.scaled_X) - time_steps)
        self.sample_weights = np.array([
            10.0 if data.iloc[i + time_steps]['mag'] >= 5 else 1.0
            for i in self.indices
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        # 输入形状：(time_steps, input_features)
        X = torch.tensor(self.scaled_X[i:i + self.time_steps], dtype=torch.float32)
        # 输出形状：(output_features,)
        y = torch.tensor(self.scaled_y[i + self.time_steps], dtype=torch.float32)
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return X, y, weight




if __name__ == "__main__":
    # 配置参数（新增模型保存路径）
    DATA_FOLDER = r"C:\Users\86150\Desktop\dataset\25 years\train\4"
    SAVE_PATH = r"C:\Users\86150\Desktop\dataset\25 years\train/lstm_model.pth"
    TIME_STEPS = 24
    EPOCHS = 200
    BATCH_SIZE = 32
    TARGET_DATE = pd.to_datetime('2022-04-01')
    input_features = ['mag', 'latitude', 'longitude', 'delta_time',
                      'np1_strike','np1_dip','np1_rake','np2_strike','np2_dip','np2_rake',
                      't_value','t_plunge','t_azimuth',
                      'n_value','n_plunge','n_azimuth'
                      ,'p_value','p_plunge','p_azimuth']
    output_features = ['mag', 'latitude', 'longitude', 'delta_time','np1_strike','np1_dip','np1_rake','np2_strike','np2_dip','np2_rake',
                      't_value','t_plunge','t_azimuth',
                      'n_value','n_plunge','n_azimuth'
                      ,'p_value','p_plunge','p_azimuth']

    # 数据加载与数据集创建
    processed_data = load_data_from_folder(DATA_FOLDER)
    dataset = EarthquakeDataset(processed_data, input_features, output_features, TIME_STEPS)

    # 改进的数据加载器（添加persistent_workers）
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型和优化器
    model = LSTMRegressor(
        input_size=len(input_features),
        output_size=len(output_features)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # 训练循环（添加梯度裁剪）
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for X_batch, y_batch, w_batch in dataloader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)

            # 使用与Keras相同的特征权重
            feature_weights = torch.tensor([1.0, 3.0, 3.0, 1.0], device=device)
            loss = weighted_mse_loss(preds, y_batch, feature_weights, w_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()

            # 保存中间模型
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Model saved at epoch {epoch + 1}")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(dataloader):.4f}")

    # 预测阶段（添加温度缩放）
    model.eval()
    last_sequence = torch.tensor(dataset.scaled_X[-TIME_STEPS:],
                                 dtype=torch.float32).unsqueeze(0).to(device)
    last_timestamp = processed_data['time'].iloc[-1]
    results = []

    with torch.no_grad():
        while True:
            pred = model(last_sequence)
            pred_np = pred.cpu().numpy()
            pred_denorm = dataset.scaler_y.inverse_transform(pred_np)[0]

            delta_hours = max(pred_denorm[3], 0.1)  # 防止负时间差
            new_time = last_timestamp + pd.Timedelta(hours=delta_hours)

            if new_time > TARGET_DATE:
                break

                # 记录结果（添加异常值过滤）
            if 0 < pred_denorm[0] < 10 and -90 <= pred_denorm[1] <= 90 and -180 <= pred_denorm[2] <= 180:
                results.append({
                    'time': new_time.strftime('%Y-%m-%d  %H:%M:%S'),
                    'magnitude': pred_denorm[0],
                    'latitude': pred_denorm[1],
                    'longitude': pred_denorm[2],
                    'delta_time': delta_hours
                })

            # 改进的序列更新（添加平滑处理）
            new_input = dataset.scaler_X.transform([pred_denorm])
            new_tensor = torch.tensor(new_input, dtype=torch.float32).to(device)

            # 保持序列长度恒定
            last_sequence = torch.cat([
                last_sequence[:, 1:, :],
                new_tensor.unsqueeze(0)
            ], dim=1)

            last_timestamp = new_time

            # 保存最终结果（添加异常处理）
    try:
        result_df = pd.DataFrame(results)
        output_path = os.path.join(r"C:\Users\86150\Desktop\dataset\results", 'lstm_predictions-2020-4-realmt.csv')
        result_df.to_csv(output_path, index=False)
        print(f"成功保存预测结果至：{output_path}")
    except Exception as e:
        print(f"保存失败：{str(e)}")