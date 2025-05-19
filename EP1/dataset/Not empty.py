import xarray as xr

# 指定要检查的 NetCDF 文件路径（以2000-06文件为例）
nc_file_path = "C:\\Users\\86150\\Desktop\\nc_files\\2000-01.nc"  # 替换为你要检查的文件路径

# 读取 NetCDF 文件
ds = xr.open_dataset(nc_file_path)

# 打印数据集的基本信息
# print("数据集基本信息:")
# print(ds)

# 检查每个变量的非空值数量，最大值和最小值
for var in ds.data_vars:
    non_empty_count = ds[var].notnull().sum().item()  # 计算非空值的数量
    max_value = ds[var].max().item()  # 计算最大值
    min_value = ds[var].min().item()  # 计算最小值
    print(f"{var} 变量的非空值数量: {non_empty_count}, 最大值: {max_value}, 最小值: {min_value}")

# 关闭数据集
ds.close()