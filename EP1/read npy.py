import numpy as np

# 指定 .npy 文件的路径
file_path = 'C:\code\pythonProject\lmizrahi-etas-9152bb2\input_data\ch_rect.npy'

# 加载 .npy 文件
data = np.load(file_path)

# 打印数据
print(data)