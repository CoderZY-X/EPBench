"""
This Python file is used to partition data according to predefined regions.
For region delineation, please refer to the "region selection" file located in the same directory.
"""
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

                    p1 = (float(matches[0][1]), float(matches[0][0]))  # (Latitude, Longitude)
                    p2 = (float(matches[1][1]), float(matches[1][0]))  # (Latitude, Longitude)

                    # 计算矩形边界
                    ul = (max(p1[0], p2[0]), min(p1[1], p2[1]))  # Max latitude, Min longitude
                    lr = (min(p1[0], p2[0]), max(p1[1], p2[1]))  # Min latitude, Max longitude

                    rectangles[current_set].append((ul, lr))
                except ValueError as e:
                    print(f"Coordinate conversion error: {e}")
    return rectangles

"""Determine if a point is inside a rectangle"""
def point_in_rectangle(point: Tuple[float, float],
                       rectangle: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
    lat, lon = point
    (ul_lat, ul_lon), (lr_lat, lr_lon) = rectangle
    return (lr_lat <= lat <= ul_lat) and (ul_lon <= lon <= lr_lon)

"""Multiple set filtering and matching (returns matched data, unmatched data, and count of successful matches)"""
def filter_points(rectangles: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
                  points: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, int]:
    categorized = {k: [] for k in rectangles}
    unmatched = []
    matched_indices: Set[int] = set()

    for idx, row in points.iterrows():
        record = row.to_dict()
        lat = record['latitude']
        lon = record['longitude']
        matched = False

        # Iterate through all sets
        for set_id, rects in rectangles.items():
            # Check all rectangles in the current set
            for rect in rects:
                if point_in_rectangle((lat, lon), rect):
                    categorized[set_id].append(record)
                    matched = True
                    matched_indices.add(idx)  # Record successful index
                    break   # Only need to match one rectangle in the set

        if not matched:
            unmatched.append(record)

    return (
        {k: pd.DataFrame(v) for k, v in categorized.items() if v},
        pd.DataFrame(unmatched),
        len(matched_indices)
    )

"""data saving function"""
def save_to_csv(categorized_data: Dict[str, pd.DataFrame],
                original_filename: str,
                output_base_dir: str):

    for set_id, df in categorized_data.items():
        output_dir = os.path.join(output_base_dir, str(set_id))
        os.makedirs(output_dir, exist_ok=True)

        # Keep original filename, excluding extension
        base_name = os.path.splitext(original_filename)[0]
        safe_name = f"{base_name}.csv" # Save as CSV format

        # Save data for each region
        df.to_csv(os.path.join(output_dir, safe_name), index=False, encoding='utf_8_sig')

"""Process all CSV files in the input directory"""
def process_files(input_dir: str, rectangles: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]], output_base_dir: str):

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            data_file_path = os.path.join(input_dir, filename)
            points = pd.read_csv(data_file_path)


            categorized, unmatched, success_count = filter_points(rectangles, points)


            save_to_csv(categorized, filename, output_base_dir)


            if not unmatched.empty:
                unmatched_output_path = os.path.join(output_base_dir, 'unmatched', filename)
                os.makedirs(os.path.dirname(unmatched_output_path), exist_ok=True)
                unmatched.to_csv(unmatched_output_path, index=False, encoding='utf_8_sig')

                # Print unmatched data
                print(f"\nUnmatched data (file: {filename}):")
                print(unmatched)

            print(f"Processed file: {filename} | Successful matches: {success_count}")


def main():
    rect_file = r".../region.txt"
    input_dir = r".../dataset/1995~2020" # Folder containing CSV files to process
    output_base_dir = r".../test"   # Output folder

    # Load rectangle region data
    rectangles = load_rectangles_from_file(rect_file)


    process_files(input_dir, rectangles, output_base_dir)


if __name__ == "__main__":
    main()