import pandas as pd
import numpy as np

# 读取CSV文件
A = pd.read_csv(r"C:\Users\86150\Desktop\dataset\1970~1995\test\2\test2.csv")
B = pd.read_csv(r"C:\Users\86150\Desktop\dataset\results\etas\1970~1995\simulations_ch-2.csv")

# 处理时间，转换为日期时间格式，确保精确到秒
A['time'] = pd.to_datetime(A['time'], errors='coerce').dt.tz_localize(None)
# B['time'] = pd.to_datetime(B['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
if 'mag' not in B.columns:
    B['mag'] = B['magnitude']
B['time'] = pd.to_datetime(
    B['time'],
    format='%Y-%m-%d %H:%M:%S.%f',  # 明确包含微秒格式
    errors='coerce'
).dt.floor('s')  # 将时间精度统一到秒级

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


# 初始化全局统计变量
success_count = 0  # 成功匹配计数
total_events = 0  # 总事件计数
total_losses = []  # 新增：存储所有成功匹配的最小损失值
total_mag = []  # 新增：存储所有成功匹配的最小损失值

# 遍历每个A的数据
for index_a, row_a in A.iterrows():
    if row_a['mag']<4.5:
        continue
    total_events += 1  # 累计总事件数
    found_events = []

    for index_b, row_b in B.iterrows():
        # 计算时间差（以秒为单位）
        time_diff = abs((row_a['time'] - row_b['time']).total_seconds())

        # 计算距离差
        distance_diff = haversine(row_a['longitude'], row_a['latitude'],
                                  row_b['longitude'], row_b['latitude'])

        # 检查条件，时间不超过2天（172800秒），距离不超过3000km
        if time_diff <= 60*60*24*3 and distance_diff <= 150:
            found_events.append((row_b,index_b, time_diff, distance_diff))

    if not found_events:
        # 若没有找到符合条件的事件
        print(f"A中事件{index_a + 1}（时间：{row_a['time']}, 经纬度：{row_a['latitude']}, {row_a['longitude']},{row_a['mag']}): 没找到")
    else:
        # 找到符合条件的事件，累计成功计数
        success_count += 1

        # 计算loss并找到最小值
        min_loss = float('inf')
        best_match = None

        for (event_b, index_b,time_diff, distance_diff) in found_events:
            # 计算损失（保持原有公式不变）
            loss = (((time_diff / (60 * 60 * 24 * 3)) ** 2) + (distance_diff / 150) ** 2) * 0.5

            if loss < min_loss:
                min_loss = loss
                best_match = event_b
                best_index = index_b
        total_losses.append(min_loss)  # 新增
        # print(float(best_match['magnitude']),float(row_a['mag']))
        total_mag.append(abs(float(best_match['mag'])-float(row_a['mag'])))
        # 打印找到的结果
        print(
            f"A中事件{index_a + 1}（时间：{row_a['time']}, 经纬度：{row_a['latitude']}, {row_a['longitude']},震级:{row_a['mag']}): "
            f"B中事件{best_index + 1}（时间：{best_match['time']}, 经纬度：{best_match['latitude']}, {best_match['longitude']},震级：{best_match['mag']}），最小损失：{min_loss}")

matching_rate = (success_count / total_events) * 100
print(f"\n{'=' * 50}\n成功匹配统计: 共处理{total_events}个事件，成功匹配{success_count}个")
print(f"匹配成功率: {matching_rate:.2f}%")
if success_count > 0:
    avg_loss = sum(total_losses)/success_count
    avg_mag = sum(total_mag) / success_count
    print(f"最小损失平均值: {avg_loss:.4f}")
    print(f"最小mag: {avg_mag:.4f}")

total_events_b=0
success_count_b = 0
for index_b, row_b in B.iterrows():
    if row_b['mag'] < 4.5:
        continue
    total_events_b += 1
    for index_a, row_a in A.iterrows():
        if row_a['mag'] < 4.5:
            continue
        time_diff = abs((row_b['time'] - row_a['time']).total_seconds())
    # 计算距离差
        distance_diff = haversine(row_b['longitude'], row_b['latitude'],row_a['longitude'], row_a['latitude'])

    # 检查条件，时间不超过2天（172800秒），距离不超过3000km
        if time_diff <= 60 * 60 * 24 * 3 and distance_diff <= 150:
            success_count_b +=1
            break

# 计算并输出最终统计结果



false_alarm_rate = ((total_events_b-success_count_b)/total_events_b)*100
print(f"误报率: {false_alarm_rate:.2f}%")