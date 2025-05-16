import requests
import os

# 定义函数下载数据
def download_data(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        with open(filename, 'w') as file:
            file.write(response.text)
        print(f"成功下载: {filename}")
    except Exception as e:
        print(f"下载失败: {filename}，错误: {e}")

# 指定输入和输出文件
input_file = r"C:\Users\86150\Desktop\event_links.txt"  # 输入的链接文件
output_dir = r"C:\Users\86150\Desktop\EarthQuake1970~2021\multimodal data\waveform"  # 目标文件夹

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取链接文件并下载数据
with open(input_file, 'r') as file:
    for line in file:
        # 分割每行的标签和链接
        parts = line.strip().split(': ')
        if len(parts) == 2:
            label, url = parts
            # 创建输出文件路径
            output_file = os.path.join(output_dir, f"{label}.csv")
            # 检查文件是否已存在
            if not os.path.exists(output_file):
                # 只有不存在时才下载
                download_data(url, output_file)
            else:
                print(f"文件已存在，跳过： {output_file}")