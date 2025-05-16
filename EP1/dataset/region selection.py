import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np


# 矩形绘制的回调函数
def on_click(event):
    global points, color_index
    if event.inaxes is not None:
        points.append((event.xdata, event.ydata))
        if len(points) == 2:
            # 计算矩形的左上角和右下角
            x1, y1 = points[0]
            x2, y2 = points[1]
            left_top = (min(x1,x2), max(y1, y2))  # 左上角
            right_bottom = (max(x1,x2), min(y1, y2))  # 右下角

            # 绘制矩形
            color = colors[color_index % len(colors)]  # 选择颜色
            plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color=color)

            # 打印左上角和右下角的坐标
            print(f"{color}: {left_top}, {right_bottom}")

            plt.draw()

            # 更新点和颜色索引
            points = []  # 重置点
            color_index += 1  # 切换到下一个颜色


# 设置全局变量
points = []
color_index = 0  # 初始化颜色索引
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow']  # 定义颜色列表

# 创建一个地图
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# 绘制基础地图
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.OCEAN)

# 设置地图的长宽比一致，避免缩放
ax.set_aspect('equal')

# 连接点击事件
cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

plt.title("Click to draw rectangles on the world map.")
plt.show()