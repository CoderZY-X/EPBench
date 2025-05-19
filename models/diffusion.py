import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# parameters
T = 1000
beta_start = 1e-4
beta_end = 0.02
device = torch.device('cuda'  if torch.cuda.is_available()  else 'cpu')
beta_schedule = torch.linspace(beta_start,  beta_end, T, dtype=torch.float32).to(device)
alpha_schedule = 1. - beta_schedule
alpha_bar_schedule = torch.cumprod(alpha_schedule, dim=0)


class weighted_mse_loss(nn.Module):
    def __init__(self, feature_weights_dict):
        super().__init__()
        self.feature_order = ['magnitude', 'latitude', 'longitude', 'delta_time']
        self.weights = torch.tensor(
            [feature_weights_dict[k] for k in self.feature_order],
            dtype=torch.float32
        )

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        loss = loss * self.weights.to(loss.device)
        return loss.mean()

class TransformerEncoder(nn.Module):

    def __init__(self, input_dim, num_layers=4, nhead=8, dim_feedforward=256):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, input_dim))

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

        x = x.permute(1, 0, 2)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.projection(x)


class EPDiff(nn.Module):
    def __init__(self, input_dim, cond_steps, cond_dim, device):  # 新增device参数
        super().__init__()
        self.device = device


        self.cond_encoder = TransformerEncoder(
            input_dim=cond_dim,
            num_layers=4,
            nhead=4,
            dim_feedforward=256
        ).to(device)
        self.time_embed = nn.Sequential(
            nn.Embedding(T, 64).to(device),
            nn.Linear(64, 64).to(device),
            nn.SiLU()
        )



        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim + 64 + 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, cond_input, y_input, t_input):

        cond_input = cond_input.to(self.device)
        y_input = y_input.to(self.device)
        t_input = t_input.to(self.device)


        cond_vec = self.cond_encoder(cond_input)


        t_emb = self.time_embed(t_input)


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



    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = weighted_mse_loss(feature_weights_dict).to(device)  # 关键修改点


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = X_batch.size(0)


            t = torch.randint(0, T, (batch_size,), device=device)


            noise = torch.randn_like(y_batch, device=device)


            alpha_bar_t = alpha_bar_schedule[t].view(-1, 1)
            y_noisy = torch.sqrt(alpha_bar_t) * y_batch + torch.sqrt(1 - alpha_bar_t) * noise


            optimizer.zero_grad()


            pred_noise = model(X_batch, y_noisy, t)


            loss = loss_fn(pred_noise, noise)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


            total_loss += loss.item() * batch_size


        scheduler.step()


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


        pred_noise = model(cond_input, y_t, t_tensor)


        beta_t = beta_schedule[t]
        alpha_t = alpha_schedule[t]
        alpha_bar_t = alpha_bar_schedule[t]


        y_t = (y_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(y_t)
            y_t += torch.sqrt(beta_t) * noise

    return y_t.cpu()



def load_data_from_folder(folder_path):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    full_data = []
    prev_last_time = None

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()

            if 'time' not in df.columns:
                print(f"File {file_name} missing 'time' column, skipping")
                continue

            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time')


            if prev_last_time is not None:
                time_diff = (df.iloc[0]['time'] - prev_last_time).total_seconds() / 3600
                df.loc[df.index[0], 'delta_time'] = time_diff


            df['delta_time'] = df['time'].diff().dt.total_seconds().fillna(0) / 3600
            prev_last_time = df.iloc[-1]['time']

            full_data.append(df)
        except Exception as e:
            print(f"Failed to load {file_path}: {str(e)}")

    if not full_data:
        return None
    return pd.concat(full_data).sort_values('time')




if __name__ == "__main__":

    DATA_FOLDER = r".../train/6"
    TIME_STEPS = 96
    EPOCHS = 400
    BATCH_SIZE = 64
    TARGET_DATE = pd.to_datetime('2020-04-01') # prediction cutoff
    input_features = ['magnitude', 'latitude', 'longitude', 'delta_time']
    output_features = ['magnitude', 'latitude', 'longitude', 'delta_time']


    processed_data = load_data_from_folder(DATA_FOLDER)


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


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EPDiff(
        input_dim=len(output_features),
        cond_steps=TIME_STEPS,
        cond_dim=len(input_features),
        device=device
    ).to(device)

    feature_weights = {
        'magnitude': 1.0,
        'latitude': 3.0,
        'longitude': 3.0,
        'delta_time': 1.0  #
    }

    train_diffusion_model(
        model,
        dataloader,
        feature_weights_dict=feature_weights,
        epochs=EPOCHS,
        device=device
    )


    last_sequence = torch.tensor(
        dataset.scaled_X[-TIME_STEPS:],
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    last_timestamp = processed_data['time'].iloc[-1]
    results = []


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


        new_input = dataset.scaler_X.transform([y_pred])
        new_tensor = torch.tensor(new_input, dtype=torch.float32).to(device)


        last_sequence = torch.cat(
            [last_sequence[:, 1:, :],
             new_tensor.unsqueeze(1)],
            dim=1
        )
        last_timestamp = new_time


    result_df = pd.DataFrame(results)
    output_path = os.path.join('.../output', 'diff-prediction.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"prediction saved to：{output_path}")