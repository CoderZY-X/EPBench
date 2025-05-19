import numpy as np

# 定义全球的纬度和经度边界
coordinates = np.array([
    [-89.9, -179.9],  # 左下角
    [89.99, -179.9],   # 左上角
    [89.99, 179.99],    # 右上角
    [-89.9, 179.99],   # 右下角
    [-89.9, -179.9]   # 再次到左下角以关闭多边形
])

# 指定要保存的路径
output_path = r'C:\code\pythonProject\lmizrahi-etas-9152bb2\input_data/global_coordinates.npy'  # 请替换为您实际的路径

# 保存数组到 .npy 文件
np.save(output_path, coordinates)

print(f"全球范围的坐标已保存到 {output_path}")