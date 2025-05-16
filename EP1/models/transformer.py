import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Layer
)
import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class PositionalEncoding(Layer):
    """可学习的位置编码层（带完整输出规范）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.position = self.add_weight(
            name='pos_encoding',
            shape=(1, input_shape[1], input_shape[2]),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position

    def get_config(self):
        return super().get_config()


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    res = attn_output + inputs

    ln_output = LayerNormalization(epsilon=1e-6)(res)
    ff_output = Dense(ff_dim, activation="relu")(ln_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)

    return LayerNormalization(epsilon=1e-6)(ff_output + ln_output)


def build_transformer_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = PositionalEncoding()(inputs)

    x = transformer_encoder(
        x,
        head_size=32,
        num_heads=4,
        ff_dim=128,
        dropout=0.3
    )

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_shape)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def load_data_from_folder(folder_path):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    full_data = []
    prev_last_time = None

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()

            # 时间解析
            df['time'] = df['time'].str.replace(r'\s+', ' ', regex=True)
            df['time'] = df['time'].str.replace(r'\.\d+$', '', regex=True)
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time')

            # 跨文件时间差计算
            if prev_last_time is not None:
                time_diff = (df.iloc[0]['time'] - prev_last_time).total_seconds() / 3600
                df.at[df.index[0], 'delta_time'] = time_diff

            # 文件内时间差计算
            df['delta_time'] = df['time'].diff().dt.total_seconds() / 3600
            prev_last_time = df.iloc[-1]['time']

            full_data.append(df)
        except Exception as e:
            print(f"加载 {file_path} 失败: {str(e)}")

    combined_df = pd.concat(full_data).sort_values('time')
    combined_df = combined_df.dropna(subset=['delta_time'])  # 删除第一条无delta_time的记录
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


if __name__ == "__main__":
    DATA_FOLDER = "C:\\Users\\86150\\Desktop\\csv_files"
    TIME_STEPS = 12
    EPOCHS = 100
    BATCH_SIZE = 8
    TARGET_DATE = pd.to_datetime('2001-02-02')

    # 修改后的输入输出特征
    input_features = ['mag', 'latitude', 'longitude', 'delta_time']
    output_features = ['mag', 'latitude', 'longitude', 'delta_time']

    # 加载并预处理数据
    processed_data = load_data_from_folder(DATA_FOLDER)
    X, y, scaler_X, scaler_y = prepare_dataset(processed_data, input_features, output_features, TIME_STEPS)

    # 构建模型
    model = build_transformer_model(
        input_shape=(TIME_STEPS, len(input_features)),
        output_shape=len(output_features)
    )

    # 训练
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # 预测逻辑
    last_sequence = X[-1]
    last_timestamp = processed_data['time'].iloc[-1]
    results = []

    while True:
        # 预测下一个时间步
        pred = model.predict(last_sequence.reshape(1, TIME_STEPS, -1))
        pred_features = scaler_y.inverse_transform(pred)[0]

        # 解析预测结果
        delta_hours = pred_features[3]
        new_timestamp = last_timestamp + pd.Timedelta(hours=delta_hours)

        # 终止条件
        if new_timestamp > TARGET_DATE:
            break

        # 记录结果
        results.append({
            'time': new_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'mag': pred_features[0],
            'latitude': pred_features[1],
            'longitude': pred_features[2],
            'delta_time': delta_hours
        })

        # 更新输入序列
        new_input = scaler_X.transform([pred_features])
        last_sequence = np.concatenate([last_sequence[1:], new_input])
        last_timestamp = new_timestamp

    # 保存结果
    result_df = pd.DataFrame(results)
    output_path = os.path.join(DATA_FOLDER, 'improved_predictions.csv')
    result_df.to_csv(output_path, index=False)
    print(f"预测结果已保存至：{output_path}")