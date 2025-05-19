import pandas as pd
import numpy as np
import os

# 定义输入和输出文件路径
input_file = r"C:\Users\86150\Desktop\dataset\1995~2020\train\6\1995~2020.csv" # 替换为你的CSV文件路径
output_folder = r"C:\Users\86150\Desktop\dataset\1995~2020\train-2\6" # 输出文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 使用 pandas 读取 CSV 文件
data = pd.read_csv(input_file)

# 打印所有列名
print("所有列名:", data.columns.tolist())  # 打印所有列名

# 将时间列转换为 datetime 对象，并确保不带时区
data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)
time_array = np.array(data['time'])
unique_months = np.unique(time_array.astype('datetime64[M]'))  # 获取唯一的月份

# 定义一个函数来计算角度的 cos 值
def convert_to_cos(value):
    if isinstance(value, str):
        value = value.replace('°', '')  # 去掉度符号并转换为 float
        angle = float(value)  # 转换为 float
        return np.cos(np.radians(angle))  # 计算 cos 值，确保已转换为弧度
    return value

# 循环处理每个月的数据
for month in unique_months:
    # 筛选出当前月份的数据
    indices = np.where(time_array.astype('datetime64[M]') == month)[0]

    # 确保 indices 是一个有效的数组
    if indices.size == 0:
        print(f"警告: 没有找到月份 {month} 的数据！")
        continue

    # 包含时间列
    monthly_data = {name: np.array(data[name].iloc[indices]) for name in data.columns}

    # 创建一个 DataFrame 来存储当前月份的数据
    monthly_df = pd.DataFrame()

    # 处理其他变量
    variable_columns = [
        'NP1_Strike', 'NP1_Dip', 'NP1_Rake',
        'NP2_Strike', 'NP2_Dip', 'NP2_Rake',
        'T_Value', 'T_Plunge', 'T_Azimuth',
        'N_Value', 'N_Plunge', 'N_Azimuth',
        'P_Value', 'P_Plunge', 'P_Azimuth',
        'depth', 'magnitude', 'latitude', 'longitude'  # 添加 depth, mag, latitude 和 longitude
    ]

    for col in variable_columns:
        if col in monthly_data:
            if col in ['depth', 'magnitude', 'latitude', 'longitude']:
                # 直接存储 depth, mag, latitude 和 longitude 的值
                monthly_df[col] = monthly_data[col]
            else:
                # 转换角度并处理数据
                monthly_data[col] = np.array([convert_to_cos(value) for value in monthly_data[col]])
                monthly_df[col] = monthly_data[col]

    # 添加时间列，保留所有数据的时间
    monthly_df['time'] = monthly_data['time']

    # 保存为 CSV 文件，文件名以月份命名
    output_file = os.path.join(output_folder, f"{month}.csv")
    monthly_df.to_csv(output_file, index=False)

    print(f"已成功生成文件: {output_file}")

print("所有月份的 CSV 文件已生成。")