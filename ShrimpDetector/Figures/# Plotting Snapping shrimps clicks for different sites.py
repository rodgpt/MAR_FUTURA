# SNAPPING SHRIMP CLIKS----
# Margherita Silvestri----
# 7-10-25

# ========================================================================
# CODE TO PLOT SNAPPING SHRIMP CLICKS WITH INTERACTIVE AND DAILY VIEWS
# ======================================================================== 

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from matplotlib.patches import Patch
import plotly.graph_objects as go

# ========== CONFIGURATION ==========
# Name of the site folder to analyze
SITE_NAME = "LASCRUCES"  # e.g. "LASCRUCES", "VENTANAS", "ZAPALLAR", etc.

# Date‐range filters: list of (start, end) in YYYYMMDDHHMMSS.
# Use end=None for “from start until the end of the dataset.”:
FILTER_RANGES = [
    # Example for LASCRUCES
    ("20240619162200", None)
    # or # Example for VENTANAS :  
    # [] # Empty list => include all recordings.
]

# ========== GLOBAL Y-AXIS LIMITS FOR ALL PLOTS ==========
GLOBAL_Y_MIN = 300
GLOBAL_Y_MAX = 600

# Paths
root       = rf"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\RECORDINGS\{SITE_NAME}\output_all_2"
output_dir = rf"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\plots\interactive\{SITE_NAME}"
os.makedirs(output_dir, exist_ok=True)

# ========== LOAD & FILTER RECORDINGS ==========
# Find all recording subfolders
folders = glob.glob(os.path.join(root, "*"))
# Load & filter records
records = [] # Will hold each recording’s timestamp and click rate
excluded = 0 # Count of recordings filtered out

# Pre‐convert filter ranges to Timestamp objects
parsed_ranges = []
for start, end in FILTER_RANGES:
    s = pd.to_datetime(start, format='%Y%m%d%H%M%S')
    e = pd.to_datetime(end,   format='%Y%m%d%H%M%S') if end else None
    parsed_ranges.append((s, e))

# Loop through each folder, parse the timestamp, apply filters, read .labels.txt
for folder in folders:
    base = os.path.basename(folder)
    parts = base.split('_')
    if len(parts)<3: continue
    date, time = parts[1], parts[2]
    dt = pd.to_datetime(date+time, format='%Y%m%d%H%M%S')

    # Apply multi-range filter
    if parsed_ranges:
        keep = False
        for s,e in parsed_ranges:
            if dt>=s and (e is None or dt<=e):
                keep=True
                break
        if not keep:
            excluded += 1
            continue
    # Path to the labels file containing click times
    lf = os.path.join(folder, f"{base}.labels.txt")
    if not os.path.exists(lf): continue
    # Read click start times from labels file
    starts=[]
    with open(lf) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            try: starts.append(float(line.split('\t')[0]))
            except: pass
    # Compute clicks per minute for this recording
    if len(starts)>1:
        dur = starts[-1]-starts[0]
        cpm = len(starts)/(dur/60) if dur>0 else 0
    else:
        cpm=0
    # Store the result
    records.append({'datetime':dt,'clicks_per_min':cpm,'recording_name':f"{base}.wav"})

# Convert to DataFrame
df=pd.DataFrame(records)
print(f"Loaded {len(df)} recordings (excluded {excluded})")
if df.empty: raise RuntimeError("No data loaded.")

# ========== AGGREGATE TO HOURLY & SMOOTH ==========
# Round timestamps down to the nearest hour
df['hourly']=df['datetime'].dt.floor('H')

# Compute mean clicks/min for each hour
hourly=df.groupby('hourly')['clicks_per_min'].mean().reset_index()

# Add 3‐hour and 6‐hour centered rolling averages 
# we can use this part with more recordings
#hourly['r3']=hourly['clicks_per_min'].rolling(3,center=True).mean()
#hourly['r6']=hourly['clicks_per_min'].rolling(6,center=True).mean()

# ========== INTERACTIVE FULL TIME‐SERIES PLOT ==========
# Label each point as Day or Night
df['period']=np.where(df['datetime'].dt.hour.between(6,19),'Day','Night')
df['color']=df['period'].map({'Day':'gold','Night':'navy'})


# Initialize Plotly figure
fig=go.Figure()

# Add scatter of clicks per minute
fig.add_trace(go.Scatter(
    x=df['datetime'],y=df['clicks_per_min'],mode='markers',
    marker=dict(color=df['color'],size=6,opacity=0.7),
    text=df['recording_name'],
    hovertemplate="<b>%{text}</b><br>%{x}<br>Clicks/min: %{y:.1f}<extra></extra>"
))

# Add night shading rectangles for each date
shapes=[]
for d in df['datetime'].dt.normalize().unique():
    shapes += [
        dict(type="rect",xref="x",yref="paper",
             x0=d,x1=d+pd.Timedelta(hours=6),y0=0,y1=1,
             fillcolor="navy",opacity=0.1,layer="below",line_width=0),
        dict(type="rect",xref="x",yref="paper",
             x0=d+pd.Timedelta(hours=20),x1=d+pd.Timedelta(days=1),
             y0=0,y1=1,fillcolor="navy",opacity=0.1,layer="below",line_width=0)
    ]
# Finalize layout
fig.update_layout(
    title=f"{SITE_NAME} Full Time Series",
    xaxis_title="DateTime",yaxis_title="Clicks/min",
    yaxis=dict(range=[GLOBAL_Y_MIN,GLOBAL_Y_MAX]),
    shapes=shapes,width=1200,height=600
)
# Add legend entries for Day/Night
fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',
                         marker=dict(size=10,color='gold'),name='Day'))
fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',
                         marker=dict(size=10,color='navy'),name='Night'))
# Save interactive HTML
fig.write_html(os.path.join(output_dir,f"full_ts_{SITE_NAME}.html"))
print("Saved full interactive time series.")

# ========== INTERACTIVE DAILY SUMMARY PLOT ==========
# Compute daily mean clicks per minute
df['date']=df['datetime'].dt.date
daily=df.groupby('date')['clicks_per_min'].mean().reset_index()

fig2=go.Figure(go.Bar(
    x=daily['date'].astype(str),
    y=daily['clicks_per_min'],
    marker_color='teal',
    hovertemplate="<b>Date:</b> %{x}<br><b>Mean clicks/min:</b> %{y:.1f}<extra></extra>"
))
# Create a bar chart for daily averages
fig2.update_layout(
    title=f"{SITE_NAME} Daily Mean Clicks",
    xaxis_title="Date",yaxis_title="Mean clicks/min",
    yaxis=dict(range=[GLOBAL_Y_MIN,GLOBAL_Y_MAX])
)
# Finalize daily layout
fig2.write_html(os.path.join(output_dir,f"daily_summary_{SITE_NAME}.html"))
print("Saved daily summary interactive plot.")
