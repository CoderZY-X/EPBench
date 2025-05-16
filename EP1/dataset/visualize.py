import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# 指定你的NetCDF文件路径
nc_file = 'C:\\Users\\86150\\Desktop\\nc_files\\2000-06.nc'  # 示例文件路径
ds = xr.open_dataset(nc_file)

# 查看数据集的信息
print("数据集信息:")
print(ds)

# 选择你想要可视化的变量，变量名为 'mag'
var_name = 'mag'  # 数据变量名
data = ds[var_name]

# 打印经纬度
print("经度和纬度:")
print(ds['longitude'])  # 打印经度
print(ds['latitude'])   # 打印纬度

# 可视化配置
plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())  # 使用PlateCarree投影

# 画出海岸线和网格线
ax.coastlines()
ax.gridlines(draw_labels=True)

# 设定阈值，找到有值的位置
threshold = 0  # 例如，只显示大于 0 的值
masked_data = data.where(data > threshold)

# 这里检查纬度和经度的维度以创建网格
lons = ds['longitude'].values  # 获取经度值
lats = ds['latitude'].values    # 获取纬度值
LON, LAT = np.meshgrid(lons, lats)  # 创建经纬度网格

# 绘制红色圆圈
points_with_values = masked_data.notnull()  # 找到有值的位置
for i in range(points_with_values.shape[0]):  # 遍历每一行
    for j in range(points_with_values.shape[1]):  # 遍历每一列
        if points_with_values[i, j]:  # 如果这个点有值
            ax.plot(LON[i, j], LAT[i, j], 'ro', markersize=4)  # 绘制红色圆圈

# 添加标题
plt.title(f'Visualization of {var_name} for {nc_file}')

# 展示图像
plt.show()