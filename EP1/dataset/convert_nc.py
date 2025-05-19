import pandas as pd
import numpy as np
import xarray as xr
import os

# 定义输入和输出文件路径
input_file = "C:\\Users\\86150\\Desktop\\2000-merge.csv"  # 替换为你的CSV文件路径
output_folder = "C:\\Users\\86150\\Desktop\\nc_files"  # 输出文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 使用 pandas 读取 CSV 文件
data = pd.read_csv(input_file)

# 打印所有列名
print("所有列名:", data.columns.tolist())  # 打印所有列名

# 将数据转换为字典
data_dict = data.to_dict(orient='list')

# 将时间列转换为 datetime 对象，并确保不带时区
data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)
time_array = np.array(data['time'])
unique_months = np.unique(time_array.astype('datetime64[M]'))

# 定义经纬度范围
latitudes = np.round(np.arange(-90, 90.1, 0.1), 1)  # 纬度从-90到90，保留一位小数
longitudes = np.round(np.arange(-180, 180.1, 0.1), 1)  # 经度从-180到180，保留一位小数

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

    monthly_data = {name: np.array(data[name].iloc[indices]) for name in data.columns if name != 'time'}

    # 创建一个 xarray Dataset
    ds = xr.Dataset()

    # 设置纬度和经度坐标
    ds['latitude'] = (['lat'], latitudes)
    ds['longitude'] = (['lon'], longitudes)

    # 处理其他变量
    variable_columns = [
        'NP1_Strike', 'NP1_Dip', 'NP1_Rake',
        'NP2_Strike', 'NP2_Dip', 'NP2_Rake',
        'T_Value', 'T_Plunge', 'T_Azimuth',
        'N_Value', 'N_Plunge', 'N_Azimuth',
        'P_Value', 'P_Plunge', 'P_Azimuth',
        'depth', 'mag'  # 添加 depth 和 mag
    ]

    for col in variable_columns:
        if col in monthly_data:
            if col in ['depth', 'mag']:
                # 直接存储 depth 和 mag 的值
                ds[col] = (['lat', 'lon'], np.full((len(latitudes), len(longitudes)), np.nan))  # 初始化为 NaN
                for lat, lon, value in zip(monthly_data['latitude'], monthly_data['longitude'], monthly_data[col]):
                    lat_index = np.argmin(np.abs(latitudes - lat))
                    lon_index = np.argmin(np.abs(longitudes - lon))
                    ds[col][lat_index, lon_index] = value  # 直接填充 depth 和 mag 的值
            else:
                # 转换角度并处理数据
                monthly_data[col] = np.array([convert_to_cos(value) for value in monthly_data[col]])

                # 检查数据是否有效
                valid_lat_indices = ~np.isnan(monthly_data['latitude'])
                valid_lon_indices = ~np.isnan(monthly_data['longitude'])

                if valid_lat_indices.any() and valid_lon_indices.any():
                    # 创建填充数据的二维数组，初始化为 NaN
                    pivoted_data = np.full((len(latitudes), len(longitudes)), np.nan)

                    # 填充数据
                    for lat, lon, value in zip(monthly_data['latitude'], monthly_data['longitude'], monthly_data[col]):
                        lat_index = np.argmin(np.abs(latitudes - lat))
                        lon_index = np.argmin(np.abs(longitudes - lon))
                        pivoted_data[lat_index, lon_index] = value

                    # 将数据添加到 xarray Dataset 中
                    ds[col] = (['lat', 'lon'], pivoted_data)

                    # 打印每个网格存放的数据的维度和部分值
                    non_empty_values = np.count_nonzero(~np.isnan(pivoted_data))  # 计算非空值的数量
                    if non_empty_values > 0:
                        print(f"月份: {month}, 列: {col}, 非空值数量: {non_empty_values}")
                    else:
                        print(f"警告: 列 {col} 中没有非空值！")

    # 添加时间坐标
    ds['time'] = (['time'], [month.astype('datetime64[M]').astype(str)])
    print(f"保存的时间: {ds['time'].values}")

    # 保存为 NetCDF 文件
    output_file = os.path.join(output_folder, f"{month}.nc")
    ds.to_netcdf(output_file, encoding={var: {'zlib': True, 'complevel': 5} for var in ds.data_vars})

    # 检查文件是否为空
    if not ds.data_vars:
        print(f"警告: 文件 {output_file} 是空的！")
    else:
        print(f"已成功生成文件: {output_file}")

print("所有月份的 NetCDF 文件已生成。")