import pandas as pd
import os

def merge_csv_files(input_dir: str, output_file: str):
    """合并指定目录及其所有子目录中的所有 CSV 文件"""

    # 初始化一个空的列表，用于存储读取的 DataFrame
    dataframes = []

    # 遍历目录及子目录
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)  # 将读取的 DataFrame 加入列表

    # 合并所有 DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # 保存为新的 CSV 文件
    merged_df.to_csv(output_file, index=False, encoding='utf_8_sig')
    print(f"合并完成，保存为: {output_file}")

def main():
    input_dir = r"C:\Users\86150\Desktop\dataset\1970~1995\train"  # CSV 文件所在目录
    output_file = r"C:\Users\86150\Desktop\EarthQuake1970~2021\data\1970~1995\original data\train.csv"  # 合并后保存的文件路径

    merge_csv_files(input_dir, output_file)

if __name__ == "__main__":
    main()