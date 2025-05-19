import pandas as pd
import numpy as np

# 读取CSV文件
A = pd.read_csv("C:\\Users\\86150\\Desktop\\1.csv")
B = pd.read_csv("C:\\Users\\86150\\Desktop\\csv_files\\2001-01_predictions.csv")

# 处理时间，转换为日期格式
A['time'] = pd.to_datetime(A['time'], errors='coerce')
B['time'] = pd.to_datetime(B['time'], format='%Y-%m-%d')

# 根据小时调整A的时间
A['date_only'] = A['time'].apply(lambda x: x.date() + pd.Timedelta(days=1) if x.hour > 12 else x.date())

# 地球半径（单位：千米）
R = 6371.0

# 哈弗辛函数
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # 返回距离（千米）

# 遍历每个B的数据
for index_b, row_b in B.iterrows():
    found_events = []

    for index_a, row_a in A.iterrows():
        # 计算时间差
        time_diff = abs((row_a['date_only'] - row_b['time'].date()).days)
        # 计算距离差
        distance_diff = haversine(row_a['longitude'], row_a['latitude'], row_b['longitude'], row_b['latitude'])

        # 检查条件，时间不超过2天，距离不超过10km
        if time_diff <= 20 and distance_diff <= 10000 :
            found_events.append((row_a, time_diff, distance_diff))

    if not found_events:
        # 若没有找到符合条件的事件
        print(f"B中事件{index_b + 1}（时间：{row_b['time']}, 经纬度：{row_b['latitude']}, {row_b['longitude']}): 没找到")
    else:
        # 找到符合条件的事件，计算loss并找到最小值
        min_loss = float('inf')
        best_match = None

        for (event_a, time_diff, distance_diff) in found_events:
            # 计算损失
            loss = (time_diff / 2) + (distance_diff / 10)

            if loss < min_loss:
                min_loss = loss
                best_match = event_a

        # 打印找到的结果
        print(f"B中事件{index_b + 1}（时间：{row_b['time']}, 经纬度：{row_b['latitude']}, {row_b['longitude']},震级:{row_b['mag']}): "  
              f"A中事件{index_a + 1}（时间：{best_match['time']}, 经纬度：{best_match['latitude']}, {best_match['longitude']},震级:{best_match['mag']}），最小损失：{min_loss}")