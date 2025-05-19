import pandas as pd
import random
import os
import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

def load_rectangles_from_file(file_path: str) -> Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    rectangles = {}
    current_set = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if re.match(r'^\d+\.$', line):
                current_set = line[:-1]
                rectangles[current_set] = []
                continue

            matches = re.findall(r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)', line)
            if len(matches) == 2:
                try:
                    p1 = (float(matches[0][1]), float(matches[0][0]))  # (纬度, 经度)
                    p2 = (float(matches[1][1]), float(matches[1][0]))  # (纬度, 经度)

                    # 确定矩形的边界
                    ul = (max(p1[0], p2[0]), min(p1[1], p2[1]))  # 上左 (最大纬度，最小经度)
                    lr = (min(p1[0], p2[0]), max(p1[1], p2[1]))  # 下右 (最小纬度，最大经度)

                    rectangles[current_set].append((ul, lr))
                except ValueError as e:
                    print(f"坐标转换错误: {e}")
    return rectangles

def generate_random_points(rectangles: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
                            set_number: str,
                            time_range: Tuple[datetime, datetime],
                            num_points: int) -> List[Tuple[datetime, float, float]]:
    """根据矩形集合生成随机时间和坐标点"""
    if set_number not in rectangles:
        print(f"集合 {set_number} 不存在。")
        return []

    selected_rectangles = rectangles[set_number]
    generated_points = []

    for _ in range(num_points):
        # 从选中的矩形中随机挑选一个
        rectangle = random.choice(selected_rectangles)
        (ul_lat, ul_lon), (lr_lat, lr_lon) = rectangle

        # 生成随机纬度和经度
        random_lat = random.uniform(lr_lat, ul_lat)  # 随机纬度在矩形上下边界之间
        random_lon = random.uniform(ul_lon, lr_lon)  # 随机经度在矩形左右边界之间

        # 生成随机时间
        random_time = time_range[0] + timedelta(seconds=random.randint(0, int((time_range[1] - time_range[0]).total_seconds())))
        random_mag = random.uniform(6, 9)
        generated_points.append((random_time, random_lat, random_lon, random_mag))

    return generated_points

def main_with_random_generation():
    # 文件路径配置
    rect_file = r"C:\Users\86150\Desktop\region.txt"

    # 加载矩形数据
    rectangles = load_rectangles_from_file(rect_file)

    # 指定矩阵集编号和时间范围
    set_number = "4"  # 示例的矩阵集编号
    start_time = "2011-01-01 00:00:00"  # 示例开始时间
    end_time = "2012-01-01 00:00:00"    # 示例结束时间

    time_range = (datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"),
                  datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))

    # 生成点的数量
    num_points = 80  # 示例生成点的数量

    # 生成随机点
    random_points = generate_random_points(rectangles, set_number, time_range, num_points)

    # 保存生成的随机点到CSV文件
    results = []
    for point in random_points:
        results.append({
            'time': point[0].strftime('%Y-%m-%d %H:%M:%S'),
            'latitude': point[1],
            'longitude': point[2],
            'mag': point[3]  # 添加随机生成的mag值
        })

    # 创建DataFrame并保存到CSV
    result_df = pd.DataFrame(results)
    output_path = os.path.join(r"C:\Users\86150\Desktop\dataset\results", 'random_points.csv')
    result_df.to_csv(output_path, index=False)
    print(f"随机生成的结果已保存至：{output_path}")

if __name__ == "__main__":
    main_with_random_generation()