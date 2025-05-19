import pandas as pd
import numpy as np

# Read CSV files
A = pd.read_csv(r".../EP/metric/example file/example test.csv") # ground truth
B = pd.read_csv(r".../EP/metric/example file/example prediction.csv") # prediction

# Process time data, convert to datetime format ensuring second-level precision
A['time'] = pd.to_datetime(A['time'], errors='coerce').dt.tz_localize(None)
B['time'] = pd.to_datetime(B['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
if 'mag' not in B.columns:
    B['mag'] = B['magnitude']
# B['time'] = pd.to_datetime(
#     B['time'],
#     format='%Y-%m-%d %H:%M:%S.%f',  # 明确包含微秒格式
#     errors='coerce'
# ).dt.floor('s')  # 将时间精度统一到秒级

# Earth radius in kilometers
R = 6371.0


# Haversine formula function
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # Return distance in kilometers


# Initialize global statistics variables
success_count = 0  # Successful match counter
total_events = 0  # Total event counter
total_losses = []  # stores all matches' minimum loss values
total_Δmag = []  # stores all matches' magnitude discrepancies


for index_a, row_a in A.iterrows():
    if row_a['mag']<4.5:
        continue
    total_events += 1 # Increment total event count
    found_events = []

    for index_b, row_b in B.iterrows():
        # Calculate time difference (in seconds)
        time_diff = abs((row_a['time'] - row_b['time']).total_seconds())

        # Calculate distance difference
        distance_diff = haversine(row_a['longitude'], row_a['latitude'],
                                  row_b['longitude'], row_b['latitude'])

        # # Check conditions: time within 3 days, distance within 150km
        if time_diff <= 60*60*24*3 and distance_diff <= 150:
            found_events.append((row_b,index_b, time_diff, distance_diff))

    if not found_events:
        # If no matching events found
        print(
            f"Event {index_a + 1} in A (Time: {row_a['time']}, Coordinates: {row_a['latitude']}, {row_a['longitude']}, Magnitude: {row_a['mag']}): Not found")
    else:
        # If matching events found, increment success counter
        success_count += 1

        # Calculate loss and find minimum
        min_loss = float('inf')
        best_match = None

        for (event_b, index_b,time_diff, distance_diff) in found_events:
            loss = (((time_diff / (60 * 60 * 24 * 3)) ** 2) + (distance_diff / 150) ** 2) * 0.5

            if loss < min_loss:
                min_loss = loss
                best_match = event_b
                best_index = index_b
        total_losses.append(min_loss)
        total_Δmag.append(abs(float(best_match['mag'])-float(row_a['mag'])))

        print(
            f"Event {index_a + 1} in A (Time: {row_a['time']}, Coordinates: {row_a['latitude']}, {row_a['longitude']}, Magnitude:{row_a['mag']}): "
            f"Event {best_index + 1} in B (Time: {best_match['time']}, Coordinates: {best_match['latitude']}, {best_match['longitude']}, Magnitude: {best_match['mag']}), Minimum loss: {min_loss}")

matching_rate = (success_count / total_events) * 100
print(f"\n{'=' * 50}\nMatching statistics: Processed {total_events} events, successfully matched {success_count} events")
print(f"Matching success rate: {matching_rate:.2f}%")
if success_count > 0:
    avg_loss = sum(total_losses)/success_count
    avg_Δmag = sum(total_Δmag) / success_count
    print(f"Average minimum loss: {avg_loss:.4f}")
    print(f"Average magnitude discrepancy: {avg_Δmag:.4f}")

# Calculating the False Alarm Rate
total_events_b=0
success_count_b = 0
for index_b, row_b in B.iterrows():
    if row_b['mag'] < 4.5:
        continue
    total_events_b += 1
    for index_a, row_a in A.iterrows():
        if row_a['mag'] < 4.5:
            continue
        time_diff = abs((row_b['time'] - row_a['time']).total_seconds())

        distance_diff = haversine(row_b['longitude'], row_b['latitude'],row_a['longitude'], row_a['latitude'])


        if time_diff <= 60 * 60 * 24 * 3 and distance_diff <= 150:
            success_count_b +=1
            break



false_alarm_rate = ((total_events_b-success_count_b)/total_events_b)*100
print(f"False alarm rate: {false_alarm_rate:.2f}%")