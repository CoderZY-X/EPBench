import pandas as pd

# 读取CSV文件
csv_file_path = r"C:\Users\86150\Desktop\2001~2011.csv"  # 请替换为你的CSV文件路径
csv_data = pd.read_csv(csv_file_path)

# 读取Excel文件
csv_file_path = r"C:\Users\86150\Desktop\2001~2011-mt.csv"  # 请替换为你的Excel文件路径
csv_data1 = pd.read_csv(csv_file_path)

# 选择需要的列
selected_columns = csv_data[['id','time', 'latitude', 'longitude', 'depth', 'mag']]

# 将CSV中选取的列与Excel数据进行合并，基于ID列
merged_data = pd.merge(csv_data1, selected_columns, on='id', how='left')

# 保存到新的Excel文件
output_file_path = "C:\\Users\86150\Desktop\\2001~2011-merge.csv"  # 请替换为你想要保存的新Excel文件路径
merged_data.to_csv(output_file_path, index=False,encoding='utf-8-sig')

print("数据合并完成，输出文件为:", output_file_path)