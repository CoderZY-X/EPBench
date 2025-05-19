"""
This Python file is about generating data used for training neural network models.
It includes dividing the original data into monthly sub-files and preprocessing the moment tensor information.
"""
import pandas as pd
import numpy as np
import os

# Define input and output file paths
input_file = r".../train/6/1995~2020.csv" # Replace with your CSV file path
output_folder = r".../train/6" # # Output folder path
os.makedirs(output_folder, exist_ok=True)


data = pd.read_csv(input_file)

# Print all column names
print("All column names:", data.columns.tolist())

# Convert the 'time' column to datetime objects without timezone
data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)
time_array = np.array(data['time'])
unique_months = np.unique(time_array.astype('datetime64[M]'))   # Get unique months

#a function to compute the cosine of an angle given in degrees
def convert_to_cos(value):
    if isinstance(value, str):
        value = value.replace('°', '')  # 去掉度符号并转换为 float
        angle = float(value)  # 转换为 float
        return np.cos(np.radians(angle))  # 计算 cos 值，确保已转换为弧度
    return value

# Loop through each month and process the data
for month in unique_months:

    indices = np.where(time_array.astype('datetime64[M]') == month)[0]

    # Ensure indices is a valid array
    if indices.size == 0:
        print(f"Warning: No data found for month {month}!")
        continue

    # Include the 'time' column in the current month's data
    monthly_data = {name: np.array(data[name].iloc[indices]) for name in data.columns}

    # Create a DataFrame to store the current month's data
    monthly_df = pd.DataFrame()

    # Variables to process
    variable_columns = [
        'NP1_Strike', 'NP1_Dip', 'NP1_Rake',
        'NP2_Strike', 'NP2_Dip', 'NP2_Rake',
        'T_Value', 'T_Plunge', 'T_Azimuth',
        'N_Value', 'N_Plunge', 'N_Azimuth',
        'P_Value', 'P_Plunge', 'P_Azimuth',
        'depth', 'magnitude', 'latitude', 'longitude'
    ]

    for col in variable_columns:
        if col in monthly_data:
            if col in ['depth', 'magnitude', 'latitude', 'longitude']:
                # Store depth, magnitude, latitude, and longitude directly
                monthly_df[col] = monthly_data[col]
            else:
                # Convert angles to cosine values and process data
                monthly_data[col] = np.array([convert_to_cos(value) for value in monthly_data[col]])
                monthly_df[col] = monthly_data[col]

    # Add the 'time' column, preserving all data's time
    monthly_df['time'] = monthly_data['time']

    # Save as CSV file, named by the month
    output_file = os.path.join(output_folder, f"{month}.csv")
    monthly_df.to_csv(output_file, index=False)

    print(f"Successfully generated file: {output_file}")

print("All monthly CSV files have been generated.")