"""
This Python file is used to delineate regions. Please follow the steps below for the delineation process:

1.Run the program and select two opposite corners on the world map plot to generate a matrix.
2.After each matrix is delineated, the program will output and print the coordinates of the top-left and bottom-right corners of the matrix, regardless of the order in which you clicked the opposite corners. The "color" in the printed information corresponds to the color of the rectangle on the map.
3.Save the coordinates of the two points from the printed information into a txt file, following the format of the example file "region.txt" located in the same directory.
4.Once the regions are delineated, run the "region division" script to partition the data into regions. The divided data will be saved in folders named after the region sets you defined.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy


# Callback function for rectangle drawing
def on_click(event):
    global points, color_index
    if event.inaxes is not None:
        points.append((event.xdata, event.ydata))
        if len(points) == 2:
            # Calculate top-left and bottom-right corners of the rectangle
            x1, y1 = points[0]
            x2, y2 = points[1]
            left_top = (min(x1,x2), max(y1, y2))  # 左上角
            right_bottom = (max(x1,x2), min(y1, y2))  # 右下角

            # Draw the rectangle
            color = colors[color_index % len(colors)]  # 选择颜色
            plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color=color)

            # Print coordinates of top-left and bottom-right corners
            print(f"{color}: {left_top}, {right_bottom}")

            plt.draw()

            # Update points and color index
            points = []  # Reset points
            color_index += 1  # # Move to next color


# Initialize global variables
points = []
color_index = 0
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow']  # Define color palette

# Create a map
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Draw base map
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.OCEAN)

# Set equal aspect ratio to prevent distortion
ax.set_aspect('equal')

# Connect click event
cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

plt.title("Click to draw rectangles on the world map.")
plt.show()