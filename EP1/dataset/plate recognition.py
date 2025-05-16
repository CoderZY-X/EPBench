import os
import geopandas as gpd
from shapely.geometry import Point

# 设置 GDAL 配置
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# 加载板块多边形数据
plates = gpd.read_file(r"C:\Users\86150\Desktop\tectonicplates-master\PB2002_plates.shp")
plates = plates.to_crs("EPSG:4326")  # 确保为 WGS84 坐标系

# 打印所有字段名（确认板块名称字段）
print("所有字段名:", plates.columns.tolist())

# 打印所有板块名称（假设字段名为 'PlateName'，根据实际情况调整）
print("\n所有板块名称:")
for plate in plates['PlateName'].unique():
    print(plate)

# 创建测试点
point = Point(-95.4074, -19.9042)  # 经度, 纬度
point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")

# 空间查询
matches = gpd.sjoin(point_gdf, plates, predicate="within")

# 输出结果
if not matches.empty:
    print(f"\n该点位于板块：{matches.iloc[0]['PlateName']}")
else:
    print("\n未找到匹配的板块")