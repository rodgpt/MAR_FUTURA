# Code to plot heatmaps of snapping shrimp clicks ---
# Heatmap with hourly and 30 minutes resolution---
# Margherita Silvestri---
# 7-10-25----

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess

root = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\RECORDINGS\LASCRUCES\output_all_2"
output_dir = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\plots_LASCRUCES_3"
os.makedirs(output_dir, exist_ok=True)

# Load all records
folders = glob.glob(os.path.join(root, "*"))
records = []
for folder in folders:
    base = os.path.basename(folder)
    parts = base.split('_')
    if len(parts) < 3:
        continue
    site = parts[0]
    date = parts[1]
    time = parts[2]
    labels_file = os.path.join(folder, f"{base}.labels.txt")
    if not os.path.exists(labels_file):
        continue
    starts = []
    with open(labels_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split('\t')
            try:
                starts.append(float(fields[0]))
            except Exception:
                pass
    if len(starts) > 1:
        duration = starts[-1] - starts[0]
        num_clicks = len(starts)
        clicks_per_min = num_clicks / (duration / 60) if duration > 0 else 0
    else:
        clicks_per_min = 0
    if len(time) >= 4:
        hour = int(time[0:2])
        minute = int(time[2:4])
    else:
        hour = None
        minute = None
    records.append({
        'site': site,
        'datetime': pd.to_datetime(date + time, format='%Y%m%d%H%M%S'),
        'date': pd.to_datetime(date, format='%Y%m%d'),
        'hour': hour,
        'minute': minute,
        'clicks_per_min': clicks_per_min
    })
df = pd.DataFrame(records)

# ========== HOURLY AGGREGATION FOR SMOOTHER TRENDS ==========
# Create hourly bins
df['hourly'] = df['datetime'].dt.floor('H')

# Aggregate to hourly data (mean clicks per minute for each hour)
hourly_data = df.groupby('hourly').agg({
    'clicks_per_min': 'mean',
    'site': 'first'
}).reset_index()


# ========== Plot 1: 30-minute heatmap ==========
bin_size = 30 # adjust resolution as needed
df['time_bin'] = (df['hour'] * 60 + df['minute']) // bin_size
df['time_bin_label'] = df['time_bin'].apply(lambda x: f"{int((x*bin_size)//60):02d}:{int((x*bin_size)%60):02d}")

df['date_str'] = df['date'].dt.strftime('%a %Y-%m-%d')
ordered_dates = sorted(df['date'].unique())
ordered_date_strs = [pd.to_datetime(d).strftime('%a %Y-%m-%d') for d in ordered_dates]

heatmap_data = df.pivot_table(
    index='date_str',
    columns='time_bin_label',
    values='clicks_per_min',
    aggfunc='mean'
)
heatmap_data = heatmap_data.reindex(ordered_date_strs)

plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Mean clicks per minute'})
plt.title("Snapping Shrimp Click Activity: Days vs. Time Blocks (30-Min Resolution)")
plt.xlabel("Time of Day")
plt.ylabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "clicks_per_minute_heatmap_30min.png"), dpi=300)
plt.close()

# ========== Plot 2: Hourly heatmap (smoother) ==========
# Create hourly heatmap
hourly_data['date_str'] = hourly_data['hourly'].dt.strftime('%a %Y-%m-%d')
hourly_data['hour_label'] = hourly_data['hourly'].dt.strftime('%H:00')

hourly_heatmap = hourly_data.pivot_table(
    index='date_str',
    columns='hour_label',
    values='clicks_per_min',
    aggfunc='mean'
)

# Reorder to chronological order
ordered_hourly_dates = [pd.to_datetime(d).strftime('%a %Y-%m-%d') 
                       for d in sorted(hourly_data['hourly'].dt.date.unique())]
hourly_heatmap = hourly_heatmap.reindex(ordered_hourly_dates)

plt.figure(figsize=(16, 8))
sns.heatmap(hourly_heatmap, cmap="YlGnBu", cbar_kws={'label': 'Mean clicks per minute'})
plt.title("Snapping Shrimp Click Activity: Days vs. Hours (Hourly Resolution)")
plt.xlabel("Hour of Day")
plt.ylabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "clicks_per_minute_heatmap_hourly.png"), dpi=300)
plt.close()

print("All plots saved. Check your plots folder for:")
print("- clicks_per_minute_heatmap_10min.png (original 30-min heatmap)")
print("- clicks_per_minute_heatmap_hourly.png (hourly heatmap)")
