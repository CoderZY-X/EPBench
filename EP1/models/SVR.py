import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os
import xarray as xr

# 设置 CSV 文件所在的文件夹路径
csv_folder_path = "C:\\Users\\86150\\Desktop\\csv_files"

# 读取并合并所有CSV文件
all_data = []
for f in sorted([os.path.join(csv_folder_path, f) for f in os.listdir(csv_folder_path) if f.endswith('.csv')]):
    df = pd.read_csv(f)

    df['time'] = pd.to_datetime(df['time']).dt.to_period('M')  # 转换为月份周期

    all_data.append(df)
all_data = []
for f in sorted([os.path.join(csv_folder_path, f) for f in os.listdir(csv_folder_path) if f.endswith('.csv')]):
    df = pd.read_csv(f)
    df['num_entries'] = len(df)
    # 先转换为datetime并提取实际日期
    dt_series = pd.to_datetime(df['time'])
    df['day_of_month'] = dt_series.dt.day  # 先提取实际天数

    # 再转换为月份周期（用于后续分组）
    df['time_month'] = dt_series.dt.to_period('M')  # 新增月份周期列

    all_data.append(df)
data = pd.concat(all_data, ignore_index=True)

# 创建月份序号映射（基于time_month列）
unique_months = sorted(data['time_month'].unique())
month_seq_map = {month: i + 1 for i, month in enumerate(unique_months)}
data['month_seq'] = data['time_month'].map(month_seq_map)

# 新增日期序号（按原始时间排序）
data.sort_values(by=['time_month', 'day_of_month'], inplace=True)  # 按月份和实际日期排序
data['day_seq'] = data.groupby('month_seq').cumcount() + 1

# ...（后续代码保持原逻辑，但所有涉及月份的操作均使用month_seq列）

# 按月份聚合时使用month_seq
monthly_counts = data.groupby('month_seq').size().reset_index(name='num_entries')

# 训练事件数量预测模型（使用月份序号）
X_count = monthly_counts[['month_seq']].values
y_count = monthly_counts['num_entries'].values


scaler_count = StandardScaler()
X_count_scaled = scaler_count.fit_transform(X_count)


model_count = SVR(kernel='rbf').fit(X_count_scaled, y_count)

# 预测第13个月的事件数（假设前12个月是训练数据）
target_month_seq = len(unique_months) + 1
pred_count = int(round(model_count.predict(scaler_count.transform([[target_month_seq]]))[0]))
print(f"预测月份序号 {target_month_seq} 的事件数量: {pred_count}")

# 准备事件特征训练数据（使用原始所有事件）
feature_columns = ['depth', 'mag', 'latitude', 'longitude']  # 移除 day_of_month，单独处理
X_features = data[['month_seq', 'day_seq']].values  # 输入特征为二维
y_day_of_month = data['day_of_month'].values  # 单独处理日期预测

# 特征标准化（注意二维输入）
scaler_features = StandardScaler()
X_features_scaled = scaler_features.fit_transform(X_features)

# 训练特征预测模型（包括日期预测）
feature_models = {}
# 1. 训练日期预测模型
model_day = SVR(kernel='rbf')
model_day.fit(X_features_scaled, y_day_of_month)
feature_models['day_of_month'] = model_day

# 2. 训练其他特征模型
for col in feature_columns:
    model = SVR(kernel='rbf')
    model.fit(X_features_scaled, data[col])
    feature_models[col] = model

# 生成预测特征（包含日期序号）
X_pred = np.zeros((pred_count, 2))
X_pred[:, 0] = target_month_seq  # 月份序号
X_pred[:, 1] = np.arange(1, pred_count+1)  # 日期序号从1到预测事件数
X_pred_scaled = scaler_features.transform(X_pred)

# 预测日期（使用日期序号）
predictions = {
    'month_seq': X_pred[:, 0],
    'day_seq': X_pred[:, 1],
    'day_of_month': feature_models['day_of_month'].predict(X_pred_scaled)  # 关键修改
}

# 预测其他特征
for col in feature_columns:
    predictions[col] = feature_models[col].predict(X_pred_scaled)

# 处理日期生成（确保日期有效）
pred_df = pd.DataFrame(predictions)
pred_df['day_of_month'] = pred_df['day_of_month'].round().astype(int)
# 修正日期范围（考虑不同月份天数，假设预测月份是1月，最多31天）
pred_df['day_of_month'] = pred_df['day_of_month'].clip(1, 31)

# 生成实际日期
pred_df['date'] = pd.to_datetime({
    'year': 2001,  # 假设预测的是2001年
    'month': 1,    # 假设预测的是1月
    'day': pred_df['day_of_month']
})

# 整理最终结果（添加时区转换）
final_df = pd.DataFrame({
    'time': pred_df['date'].dt.date,  # 转换为日期格式
    'depth': pred_df['depth'],
    'mag': pred_df['mag'],
    'latitude': pred_df['latitude'],
    'longitude': pred_df['longitude'],
    'num_entries': pred_count
})


# 保存结果
output_path = os.path.join(csv_folder_path, "2001-01_predictions.csv")
final_df.to_csv(output_path, index=False)

print(f"预测结果已保存至: {output_path}")
print("示例数据:")
print(final_df.head())