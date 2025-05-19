import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# Custom weighted MSE loss function
def weighted_mse_loss(y_pred, y_true, feature_weights, sample_weights):
    """
       y_pred: (batch_size, output_features)
       y_true: (batch_size, output_features)
       feature_weights: (output_features,)
       sample_weights: (batch_size,)
    """
    squared_error = (y_true - y_pred) ** 2
    feature_weighted = squared_error * feature_weights
    sample_weighted = feature_weighted.mean(dim=1) * sample_weights
    return sample_weighted.mean()



class EarthquakeDataset(Dataset):
    def __init__(self, data, input_features, output_features, time_steps=24):
        self.data = data
        self.time_steps = time_steps
        self.input_features = input_features
        self.output_features = output_features


        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])


        self.indices = np.arange(len(self.scaled_X) - time_steps)


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




class EPLSTM(nn.Module):
    def __init__(self, input_size, output_size, sequence_length=96):
        super().__init__()
        self.sequence_length = sequence_length


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

        x, _ = self.lstm1(x)
        x = self.dropout1(self.bn1(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)


        x, _ = self.lstm2(x)
        x = self.dropout2(self.bn2(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)


        x, _ = self.lstm3(x)
        x = self.dropout3(self.bn3(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)


        x, _ = self.lstm4(x)
        x = self.dropout4(self.bn4(x.contiguous().view(-1, x.shape[-1]))).view(x.shape)


        x = x.reshape(x.shape[0], -1)
        output = self.fc(x)
        return output

def load_data_from_folder(folder_path):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    full_data = []
    prev_last_time = None

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()


            df['time'] = df['time'].str.replace(r'\s+', ' ', regex=True)
            df['time'] = df['time'].str.replace(r'\.\d+$', '', regex=True)
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df = df.dropna()
            df = df.dropna(subset=['time']).sort_values('time')

            df['delta_time'] = df['time'].diff().dt.total_seconds() / 3600


            if prev_last_time is not None:
                time_diff = (df.iloc[0]['time'] - prev_last_time).total_seconds() / 3600
                df.at[df.index[0], 'delta_time'] = time_diff

            valid_df = df.dropna(subset=['delta_time'])
            full_data.append(valid_df)
            prev_last_time = df['time'].iloc[-1]
        except Exception as e:
            print(f"Failed to load {file_path}: {str(e)}")


    combined_df = pd.concat(full_data)
    return combined_df




class EarthquakeDataset(Dataset):
    def __init__(self, data, input_features, output_features, time_steps=96):
        self.data = data
        self.time_steps = time_steps
        self.input_features = input_features
        self.output_features = output_features


        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])


        self.indices = np.arange(len(self.scaled_X) - time_steps)
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




if __name__ == "__main__":

    DATA_FOLDER = r".../train/6"
    SAVE_PATH = ".../lstm_model.pth"
    TIME_STEPS = 96
    EPOCHS = 400
    BATCH_SIZE = 32
    TARGET_DATE = pd.to_datetime('2020-04-01') # prediction cutoff
    input_features = ['magnitude', 'latitude', 'longitude', 'delta_time']
    output_features = ['magnitude', 'latitude', 'longitude', 'delta_time']


    processed_data = load_data_from_folder(DATA_FOLDER)
    dataset = EarthquakeDataset(processed_data, input_features, output_features, TIME_STEPS)


    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = EPLSTM(
        input_size=len(input_features),
        output_size=len(output_features)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for X_batch, y_batch, w_batch in dataloader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)


            feature_weights = torch.tensor([1.0, 3.0, 3.0, 1.0], device=device)
            loss = weighted_mse_loss(preds, y_batch, feature_weights, w_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()


        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Model saved at epoch {epoch + 1}")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(dataloader):.4f}")


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


            if 0 < pred_denorm[0] < 10 and -90 <= pred_denorm[1] <= 90 and -180 <= pred_denorm[2] <= 180:
                results.append({
                    'time': new_time.strftime('%Y-%m-%d  %H:%M:%S'),
                    'magnitude': pred_denorm[0],
                    'latitude': pred_denorm[1],
                    'longitude': pred_denorm[2],
                    'delta_time': delta_hours
                })


            new_input = dataset.scaler_X.transform([pred_denorm])
            new_tensor = torch.tensor(new_input, dtype=torch.float32).to(device)


            last_sequence = torch.cat([
                last_sequence[:, 1:, :],
                new_tensor.unsqueeze(0)
            ], dim=1)

            last_timestamp = new_time


    try:
        result_df = pd.DataFrame(results)
        output_path = os.path.join(".../output", 'lstm-prediction.csv')
        result_df.to_csv(output_path, index=False)
        print(f"prediction saved to：{output_path}")
    except Exception as e:
        print(f"save failure：{str(e)}")
