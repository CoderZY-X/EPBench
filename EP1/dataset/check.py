import os
import pandas as pd

# 指定输入文件
event_links_file = r"C:\Users\86150\Desktop\event_links.txt"  # 输入的链接文件
csv_file = r"C:\Users\86150\Desktop\dataset\25 years\original data\25 years.csv"  # 输入的 CSV 文件
unfind_file = r"C:\Users\86150\Desktop\dataset\unfind.csv"  # 输出的未找到 id 文件

# 读取 CSV 文件中的 event_id 和 time 列
try:
    df = pd.read_csv(csv_file)
    # 确保有 'id' 和 'time' 列
    if 'id' in df.columns and 'time' in df.columns:
        csv_event_ids = set(df['id'].astype(str).tolist())  # 获取所有 event_id
        event_time_dict = dict(zip(df['id'].astype(str).tolist(), df['time'].tolist()))  # 将 event_id 与 time 集合映射
    else:
        raise ValueError("CSV 文件中缺少 'id' 或 'time' 列。")
except Exception as e:
    print(f"读取 CSV 文件失败: {e}")
    csv_event_ids = set()
    event_time_dict = {}

# 从链接文件中提取标签
existing_labels = set()
with open(event_links_file, 'r') as file:
    for line in file:
        # 分割每行的标签和链接
        parts = line.strip().split(': ')
        if len(parts) == 2:
            label, url = parts
            existing_labels.add(label)

# 检查 CSV 中的 event_id 是否在 labels 中
missing_event_ids = {event_id: event_time_dict[event_id] for event_id in csv_event_ids if event_id not in existing_labels}

# 打印漏掉的标签和数量
missing_count = len(missing_event_ids)
if missing_event_ids:
    print("以下 event_id 在 event_links 中未找到及其对应时间:")
    for event_id, event_time in missing_event_ids.items():
        print(f"event_id: {event_id}, 时间: {event_time}")
    print(f"\n未找到的数量: {missing_count}")

    # 创建并保存未找到的 event_id 到 CSV 文件
    missing_ids_df = pd.DataFrame(list(missing_event_ids.keys()), columns=['id'])  # 创建 DataFrame
    missing_ids_df.to_csv(unfind_file, index=False)  # 保存为 unfind.csv
    print(f"\n未找到的 event_id 已保存到 {unfind_file}。")
else:
    print("所有 event_id 均在 event_links 中找到。")