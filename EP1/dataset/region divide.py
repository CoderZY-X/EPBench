import pandas as pd
import re
import os
from typing import Dict, List, Tuple, Set


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
                    # 关键修复：交换坐标顺序
                    p1 = (float(matches[0][1]), float(matches[0][0]))  # (纬度,经度)
                    p2 = (float(matches[1][1]), float(matches[1][0]))  # (纬度,经度)

                    # 计算矩形边界
                    ul = (max(p1[0], p2[0]), min(p1[1], p2[1]))  # 最大纬度，最小经度
                    lr = (min(p1[0], p2[0]), max(p1[1], p2[1]))  # 最小纬度，最大经度

                    rectangles[current_set].append((ul, lr))
                except ValueError as e:
                    print(f"坐标转换错误: {e}")
    return rectangles


def point_in_rectangle(point: Tuple[float, float],
                       rectangle: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
    """判断点是否在矩形内（修正参数顺序）"""
    lat, lon = point  # 修正为正确的纬度经度顺序
    (ul_lat, ul_lon), (lr_lat, lr_lon) = rectangle
    return (lr_lat <= lat <= ul_lat) and (ul_lon <= lon <= lr_lon)


def filter_points(rectangles: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
                  points: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, int]:
    """多集合匹配过滤（返回匹配数据、未匹配数据、不重复成功计数）"""
    categorized = {k: [] for k in rectangles}
    unmatched = []
    matched_indices: Set[int] = set()

    for idx, row in points.iterrows():
        record = row.to_dict()
        lat = record['latitude']
        lon = record['longitude']
        matched = False

        # 遍历所有集合
        for set_id, rects in rectangles.items():
            # 检查当前集合的所有矩形
            for rect in rects:
                if point_in_rectangle((lat, lon), rect):
                    categorized[set_id].append(record)
                    matched = True
                    matched_indices.add(idx)  # 记录成功索引
                    break  # 只需匹配集合中一个矩形

        if not matched:
            unmatched.append(record)

    return (
        {k: pd.DataFrame(v) for k, v in categorized.items() if v},
        pd.DataFrame(unmatched),
        len(matched_indices)
    )


def save_to_csv(categorized_data: Dict[str, pd.DataFrame],
                original_filename: str,
                output_base_dir: str):
    """增强版数据保存"""
    for set_id, df in categorized_data.items():
        output_dir = os.path.join(output_base_dir, str(set_id))
        os.makedirs(output_dir, exist_ok=True)

        # 保持原文件名，不包含扩展名
        base_name = os.path.splitext(original_filename)[0]
        safe_name = f"{base_name}.csv"  # 仍然保存为 CSV 格式

        # 保存每个区域的数据
        df.to_csv(os.path.join(output_dir, safe_name), index=False, encoding='utf_8_sig')


def process_files(input_dir: str, rectangles: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]], output_base_dir: str):
    """处理输入目录中的所有 CSV 文件"""
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            data_file_path = os.path.join(input_dir, filename)
            points = pd.read_csv(data_file_path)

            # 执行过滤
            categorized, unmatched, success_count = filter_points(rectangles, points)

            # 保存结果，传入原文件名
            save_to_csv(categorized, filename, output_base_dir)

            # 如果需要，也可以保存未匹配的数据
            if not unmatched.empty:
                unmatched_output_path = os.path.join(output_base_dir, 'unmatched', filename)
                os.makedirs(os.path.dirname(unmatched_output_path), exist_ok=True)
                unmatched.to_csv(unmatched_output_path, index=False, encoding='utf_8_sig')

                # 打印未匹配数据
                print(f"\n未匹配的数据（文件: {filename}）:")
                print(unmatched)

            print(f"处理文件: {filename} | 匹配成功数量: {success_count}")


def main():
    # 文件路径配置
    rect_file = r"C:\Users\86150\Desktop\region.txt"
    input_dir = r"C:\Users\86150\Desktop\dataset\1995~2020" # 需要处理的CSV文件夹
    output_base_dir = r"C:\Users\86150\Desktop\dataset\1995~2020\test"  # 输出文件夹

    # 加载矩形区域数据
    rectangles = load_rectangles_from_file(rect_file)

    # 处理所有输入文件
    process_files(input_dir, rectangles, output_base_dir)


if __name__ == "__main__":
    main()