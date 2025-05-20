import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler



class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, max_len, d_model))  

    def forward(self, x):
        return x + self.position[:, :x.size(1), :]



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() 

    def forward(self, src):
     
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

  
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class EnhancedEarthquakeTransformer(nn.Module):
    def __init__(self, input_dim=4, output_dim=10, seq_len=96):
        super().__init__()
  
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256)  
        )

        self.pos_encoder = LearnablePositionalEncoding(seq_len, 256)

 
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ) for _ in range(6)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
       
        x = self.feature_transform(x)  
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  

        # 逐层编码
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(1, 0, 2)  
    
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # (batch, 256)
      
        output = self.fc(x)
        return output



class WeightedEarthquakeDataset(Dataset):
    def __init__(self, data, input_features, output_features, time_steps=12):
        self.data = data
        self.time_steps = time_steps
        self.input_features = input_features
        self.output_features = output_features

 
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])

     
        self.indices = np.arange(len(self.scaled_X) - time_steps)
        self.sample_weights = self._calculate_sample_weights()

    def _calculate_sample_weights(self):

        weights = []
        for i in self.indices:
            target_idx = i + self.time_steps
            mag = self.data.iloc[target_idx]['magnitude'] 
            weights.append(10.0 if mag >= 5 else 1.0)
        return np.array(weights)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        X = torch.tensor(self.scaled_X[i:i + self.time_steps], dtype=torch.float32)
        y = torch.tensor(self.scaled_y[i + self.time_steps], dtype=torch.float32)
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return X, y, weight



class WeightedMSELoss(nn.Module):
    def __init__(self, feature_weights):
        super().__init__()
        self.feature_weights = torch.tensor(feature_weights, dtype=torch.float32)

    def forward(self, pred, target, sample_weights):
        """
        pred: (batch_size, num_features)
        target: (batch_size, num_features)
        sample_weights: (batch_size,)
        """
        squared_error = (pred - target) ** 2
        feature_weighted = squared_error * self.feature_weights.to(pred.device)
        sample_weighted = feature_weighted.mean(dim=1) * sample_weights
        return sample_weighted.mean()



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
            df = df.dropna(subset=['time']).sort_values('time')

       
            if prev_last_time is not None:
                time_diff = (df.iloc[0]['time'] - prev_last_time).total_seconds() / 3600
                df.at[df.index[0], 'delta_time'] = time_diff

      
            df['delta_time'] = df['time'].diff().dt.total_seconds() / 3600
            prev_last_time = df.iloc[-1]['time']

            full_data.append(df)
        except Exception as e:
            print(f"Failed to load {file_path}: {str(e)}")

    combined_df = pd.concat(full_data).sort_values('time')
    combined_df = combined_df.dropna(subset=['delta_time'])  
    return combined_df




if __name__ == "__main__":

    DATA_FOLDER = ".../train/6"
    TIME_STEPS = 96
    EPOCHS = 400
    BATCH_SIZE = 32
    TARGET_DATE = pd.to_datetime('2020-04-01')
    FEATURE_WEIGHTS = [1.0, 3.0, 3.0, 1.0]  
    input_features = ['magnitude', 'latitude', 'longitude', 'delta_time']
    output_features = ['magnitude', 'latitude', 'longitude', 'delta_time']


    processed_data = load_data_from_folder(DATA_FOLDER)


    dataset = WeightedEarthquakeDataset(
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

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedEarthquakeTransformer(
        input_dim=len(input_features),
        output_dim=len(output_features),
        seq_len=TIME_STEPS
    ).to(device)

    criterion = WeightedMSELoss(FEATURE_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for X_batch, y_batch, weights_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            weights_batch = weights_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch, weights_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {total_loss / len(dataset):.4f}")


    model.eval()
    last_sequence = torch.tensor(
        dataset.scaled_X[-TIME_STEPS:],
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    last_timestamp = processed_data['time'].iloc[-1]
    results = []

   
    with torch.no_grad():
        while True:
            pred = model(last_sequence)
            pred_np = pred.cpu().numpy()
            pred_denorm = dataset.scaler_y.inverse_transform(pred_np)[0]

            delta_hours = pred_denorm[3]
            new_time = last_timestamp + pd.Timedelta(hours=delta_hours)

            if new_time > TARGET_DATE:
                break

            results.append({
                'time': new_time.strftime('%Y-%m-%d %H:%M:%S'),
                'magnitude': pred_denorm[0],
                'latitude': pred_denorm[1],
                'longitude': pred_denorm[2],
                'delta_time': delta_hours
            })

      
            new_input = dataset.scaler_X.transform([pred_denorm])
            new_tensor = torch.tensor(new_input, dtype=torch.float32).to(device)
            last_sequence = torch.cat(
                [last_sequence[:, 1:, :], new_tensor.unsqueeze(1)],
                dim=1
            )
            last_timestamp = new_time


    result_df = pd.DataFrame(results)
    output_path = os.path.join('.../output', 'transformer-prediction.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"prediction saved to：{output_path}")
