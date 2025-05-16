import pandas as pd

# 读取原始 CSV
input_path = r"C:\Users\86150\Desktop\dataset\1940~1970\train\1\1940~1970.csv"
df = pd.read_csv(input_path)

# 转换为 datetime 类型（不转为字符串）
df['time'] = pd.to_datetime(
    df['time'],
    format='%Y-%m-%dT%H:%M:%S.%fZ',  # 原始格式
    errors='coerce'  # 强制无效时间为 NaT
)

# 删除无效时间行（可选）
df = df.dropna(subset=['time'])

# 保存到新 CSV（保留 datetime 类型）
output_path = r"C:\Users\86150\Desktop\dataset\1940~1970\train\1\1940~1970-1.csv"
df.to_csv(output_path, index=False)