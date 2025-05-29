import pandas as pd
import matplotlib.pyplot as plt
scada_path='C:/Users/Hp/Desktop/Project of Fundamentals of Big Data - Ying Yan/scada_data.csv'
fault_path='C:/Users/Hp/Desktop/Project of Fundamentals of Big Data - Ying Yan/fault_data.csv'
scada_df = pd.read_csv(scada_path)
fault_df = pd.read_csv(fault_path)
print(scada_df.columns)
print(fault_df.columns)
scada_df['Time'] = pd.to_datetime(scada_df['Time'], unit='s', errors='coerce')
fault_df['Time'] = pd.to_datetime(fault_df['Time'], unit='s', errors='coerce')
print(scada_df['Time'].dtype)
print(fault_df['Time'].dtype)
scada_df = scada_df.sort_values('Time')
fault_df = fault_df.sort_values('Time')
df_aligned = pd.merge_asof(
    fault_df,
    scada_df,
    on='Time',
    direction='backward',
    tolerance=pd.Timedelta('1min')
)
matched_times = df_aligned['Time'].dropna().unique()
scada_df_filtered = scada_df[~scada_df['Time'].isin(matched_times)]
print(len(scada_df_filtered))



wind_cols = ['WEC: ava. windspeed', 'WEC: max. windspeed', 'WEC: min. windspeed']
plt.figure(figsize=(14, 6))
for col in wind_cols:
    plt.plot(scada_df_filtered['Time'], scada_df_filtered[col], label=col)
plt.title('Wind Speed Trends Over Time')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



power_cols = ['WEC: ava. Power', 'WEC: max. Power', 'WEC: min. Power',
              'WEC: ava. reactive Power', 'WEC: max. reactive Power', 'WEC: min. reactive Power']
plt.figure(figsize=(14, 6))
for col in power_cols:
    plt.plot(scada_df_filtered['Time'], scada_df_filtered[col], label=col)
plt.title('Power Trends Over Time')
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



angle_rotation_cols = ['WEC: ava. Rotation', 'WEC: max. Rotation', 'WEC: min. Rotation',
                       'WEC: ava. blade angle A']
plt.figure(figsize=(14, 6))
for col in angle_rotation_cols:
    plt.plot(scada_df_filtered['Time'], scada_df_filtered[col], label=col)
plt.title('Angle & Rotation Trends Over Time')
plt.xlabel('Time')
plt.ylabel('Angle (rad) & Rotation (rad/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



temp_cols = [
    'Ambient temp.', 'Tower temp.', 'Blade A temp.', 'Nacelle temp.',
    'Rear bearing temp.', 'Front bearing temp.', 'Control cabinet temp.'
]
plt.figure(figsize=(14, 6))
for col in temp_cols:
    plt.plot(scada_df_filtered['Time'], scada_df_filtered[col], label=col)
plt.title('Temp Trends Over Time')
plt.xlabel('Time')
plt.ylabel('Temp (K)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




