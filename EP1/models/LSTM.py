import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.utils import Sequence
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 新增自定义损失函数
# def custom_loss(y_true, y_pred):
#     # 提取前4个特征（mag, latitude, longitude, delta_time）
#     y_true_sub = y_true[:, :4]
#     y_pred_sub = y_pred[:, :4]
#     # 使用类实例方式计算损失
#     return tf.keras.losses.MeanSquaredError()(y_true_sub, y_pred_sub)

def weighted_mse_loss(y_true, y_pred):
    # 特征权重分配（对应magnitude, latitude, longitude, delta_time）
    feature_weights = [1.0, 3.0, 3.0, 1.0]  # 经纬度权重设为5倍
    weights = tf.constant(feature_weights, dtype=tf.float32)

    # 计算加权MSE
    squared_error = tf.square(y_true - y_pred)
    weighted_error = squared_error * weights
    return tf.reduce_mean(weighted_error, axis=-1)

def weighted_mse_loss(y_true, y_pred):
    # 特征权重分配（对应magnitude, latitude, longitude, delta_time）
    feature_weights = [1.0, 3.0, 3.0, 1.0]  # 经纬度权重设为5倍
    weights = tf.constant(feature_weights, dtype=tf.float32)

    # 计算加权MSE
    squared_error = tf.square(y_true - y_pred)
    weighted_error = squared_error * weights
    return tf.reduce_mean(weighted_error, axis=-1)

class TimeSeriesGenerator(Sequence):
    def __init__(self, data, input_features, output_features, time_steps=12, batch_size=32):
        self.data = data
        self.input_features = input_features
        self.output_features = output_features
        self.time_steps = time_steps
        self.batch_size = batch_size

        # 初始化并拟合归一化器
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaled_X = self.scaler_X.fit_transform(data[input_features])
        self.scaled_y = self.scaler_y.fit_transform(data[output_features])

        # 生成有效样本索引
        self.sample_indices = np.arange(len(self.scaled_X) - time_steps)

        # 新增：计算样本权重（根据下一时间步的震级）
        self.sample_weights = []
        for i in self.sample_indices:
            target_idx = i + time_steps
            mag = data.iloc[target_idx]['magnitude']  # 使用原始数据获取震级
            # 震级≥5的样本权重设为5，其他为1（可根据需要调整倍数）
            self.sample_weights.append(10.0 if mag >= 5 else 1.0)

        # 新增：每个epoch结束后打乱数据
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.sample_indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.sample_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        weights_batch = []

        for i in batch_indices:
            X = self.scaled_X[i:i + self.time_steps]
            y = self.scaled_y[i + self.time_steps]
            X_batch.append(X)
            y_batch.append(y)
            weights_batch.append(self.sample_weights[i])  # 新增权重

        # 返回三元组（输入，输出，样本权重）
        return np.array(X_batch), np.array(y_batch), np.array(weights_batch)

    def on_epoch_end(self):
        # 每个epoch结束后打乱索引顺序
        np.random.shuffle(self.sample_indices)

    def get_scalers(self):
        return self.scaler_X, self.scaler_y


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


def build_lstm_model(input_shape, output_shape):
    model = Sequential()

    # 第一层 LSTM
    model.add(LSTM(512, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    # 第二层 LSTM
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    # 第三层 LSTM
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    # 全连接层
    model.add(Dense(128, activation='relu'))

    model.add(Dense(output_shape))

    # 修改损失函数
    model.compile(optimizer='adam', loss=weighted_mse_loss)  # 使用自定义损失
    return model


if __name__ == "__main__":
    DATA_FOLDER = r"C:\Users\86150\Desktop\dataset\1970~1995\train1\1"
    TIME_STEPS =96
    EPOCHS = 200
    BATCH_SIZE = 32
    TARGET_DATE = pd.to_datetime('1995-04-01')

    # 修改后的特征列表（新增delta_time）
    # input_features = ['mag', 'latitude', 'longitude', 'delta_time',
    #                   'np1_strike','np1_dip','np1_rake','np2_strike','np2_dip','np2_rake',
    #                   't_value','t_plunge','t_azimuth',
    #                   'n_value','n_plunge','n_azimuth'
    #                   ,'p_value','p_plunge','p_azimuth']
    # output_features = ['mag', 'latitude', 'longitude', 'delta_time','np1_strike','np1_dip','np1_rake','np2_strike','np2_dip','np2_rake',
    #                   't_value','t_plunge','t_azimuth',
    #                   'n_value','n_plunge','n_azimuth'
    #                   ,'p_value','p_plunge','p_azimuth']
    input_features = ['magnitude', 'latitude', 'longitude', 'delta_time'
                     ]
    output_features = ['magnitude', 'latitude', 'longitude', 'delta_time'
                     ]


    # 加载数据
    processed_data = load_data_from_folder(DATA_FOLDER)

    # 准备数据集
    X, y, scaler_X, scaler_y = prepare_dataset(
        processed_data,
        input_features,
        output_features,
        TIME_STEPS
    )

    generator = TimeSeriesGenerator(
        processed_data,
        input_features,
        output_features,
        time_steps=TIME_STEPS,
        batch_size=BATCH_SIZE
    )
    scaler_X, scaler_y = generator.get_scalers()
    # 构建模型
    model = build_lstm_model(
        input_shape=(TIME_STEPS, len(input_features)),
        output_shape=len(output_features)
    )

    # 训练模型
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.0)

    # 预测初始化
    last_sequence = generator.scaled_X[-TIME_STEPS:].reshape(1, TIME_STEPS, len(input_features))
    last_timestamp = processed_data['time'].iloc[-1]
    results = []

    # 递归预测
    while True:
        # 预测下一个时间步
        pred = model.predict(last_sequence.reshape(1, TIME_STEPS, -1))
        pred_features = scaler_y.inverse_transform(pred)[0]

        # 解析预测结果
        delta_hours = pred_features[3]

        new_timestamp = last_timestamp + pd.Timedelta(hours=delta_hours)
        last_timestamp = new_timestamp
        print(new_timestamp)
        # 终止条件
        if new_timestamp > TARGET_DATE:
            break

        # 记录结果
        results.append({
            'time': new_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'magnitude': pred_features[0],
            'latitude': pred_features[1],
            'longitude': pred_features[2],
            'delta_time': delta_hours
        })
        # 更新输入序列
        new_input = scaler_X.transform([pred_features])  # 形状 (1,4)

        # 将新输入调整为三维 (1,1,4)
        new_input_3d = new_input.reshape(1, 1, -1)  # 关键调整

        # 拼接时沿时间步维度操作
        last_sequence = np.concatenate(
            [last_sequence[:, 1:, :],  # 截取后47个时间步 (1,47,4)
             new_input_3d  # 添加新时间步 (1,1,4)
             ], axis=1)  # 沿时间维度拼接，结果形状 (1,48,4)

    # 保存结果
    result_df = pd.DataFrame(results)
    output_path = os.path.join(r"C:\Users\86150\Desktop\dataset\results", 'lstm_predictions-1995-1.csv')
    result_df.to_csv(output_path, index=False)
    print(f"预测结果已保存至：{output_path}")