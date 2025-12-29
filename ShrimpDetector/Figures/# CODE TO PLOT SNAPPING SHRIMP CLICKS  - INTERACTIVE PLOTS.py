# CODE TO PLOT SNAPPING SHRIMP CLICKS 
# # - MULTI-SITE COMPARISON -
# Margherita Silvestri
# 7-10-25

import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ========== CONFIGURATION ==========
SITES = ["LASCRUCES", "VENTANAS", "ZAPALLAR"]
SITE_COLORS = {
    "LASCRUCES": "crimson",
    "VENTANAS":  "seagreen",
    "ZAPALLAR":  "royalblue"
}

ROOT_BASE = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\RECORDINGS"
OUTPUT_DIR = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\plots\interactive\ALL_SITES_5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date-range filters per site (strings; empty list = all)
#FILTER_RANGES = {
#    "LASCRUCES": [],
#    "VENTANAS":  [],
#    "ZAPALLAR":  []
#}

# Convert string filters to Timestamp pairs
#for site, ranges in FILTER_RANGES.items():
#    new_ranges = []
#    for s, e in ranges:
#        ts = pd.to_datetime(s, format="%Y%m%d%H%M%S")
#        te = pd.to_datetime(e, format="%Y%m%d%H%M%S") if e else None
#        new_ranges.append((ts, te))
    #FILTER_RANGES[site] = new_ranges

GLOBAL_Y_MIN = 50
GLOBAL_Y_MAX = 800

# ========== LOAD & COMBINE RECORDINGS ==========
records = []
for site in SITES:
    folder_root = os.path.join(ROOT_BASE, site, "output_all_3")
    folders = glob.glob(os.path.join(folder_root, "*"))
    #parsed = FILTER_RANGES[site]
    for folder in folders:
        base = os.path.basename(folder)
        parts = base.split("_")
        if len(parts) < 3:
            continue
        dt = pd.to_datetime(parts[1] + parts[2], format="%Y%m%d%H%M%S")
        #if parsed and not any(dt >= start and (end is None or dt <= end) for start, end in parsed):
        #    continue
        labels_file = os.path.join(folder, f"{base}.labels.txt")
        if not os.path.exists(labels_file):
            continue
        starts = []
        with open(labels_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                try:
                    starts.append(float(line.split("\t")[0]))
                except:
                    pass
        cpm = len(starts) / ((starts[-1] - starts[0]) / 60) if len(starts) > 1 else 0
        records.append({
    "site": site,
    "datetime": dt,
    "clicks_per_min": cpm,
    "recording": base         # <-- Add this line!
        })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No data loaded for any site.")

# ========== INTERACTIVE FULL TIME-SERIES PLOT WITH SITE COLORS ==========
fig = go.Figure()
for site, grp in df.groupby("site"):
    fig.add_trace(go.Scatter(
    x=grp["datetime"],
    y=grp["clicks_per_min"],
    mode="markers",
    marker=dict(color=SITE_COLORS[site], size=6, opacity=0.7),
    name=site,
    text=grp["recording"],  # This text is only visible on hover
    hovertemplate=(
        f"<b>{site}</b><br>%{{x}}<br>Clicks/min: %{{y:.1f}}<br>Recording: %{{text}}<extra></extra>"
    )
))


# Shared night shading 
# I have used Night = 20:00 to 6:00; Day = 6:00 to 20:00 but we can change this
shapes = []
for d in df["datetime"].dt.normalize().unique():
    shapes += [
        dict(type="rect", xref="x", yref="paper",
             x0=d, x1=d + pd.Timedelta(hours=6), y0=0, y1=1,
             fillcolor="navy", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", xref="x", yref="paper",
             x0=d + pd.Timedelta(hours=20), x1=d + pd.Timedelta(days=1),
             y0=0, y1=1, fillcolor="navy", opacity=0.1, layer="below", line_width=0)
    ]

fig.update_layout(
    title="Snapping Shrimp Clicks: All Sites Full Time Series",
    xaxis_title="Date & Time",
    yaxis_title="Clicks per minute",
    yaxis=dict(range=[GLOBAL_Y_MIN, GLOBAL_Y_MAX]),
    shapes=shapes,
    width=1400,
    height=700
)

fig.write_html(os.path.join(OUTPUT_DIR, "all_sites_full_ts.html"))
print("Saved all-sites full time series with site colors.")


# ========== INTERACTIVE DAILY SUMMARY ==========
df["date"] = df["datetime"].dt.date
daily = df.groupby(["site", "date"])["clicks_per_min"].mean().reset_index()

fig2 = go.Figure()
for site, grp in df.groupby("site"):
    fig.add_trace(go.Scatter(
        x=grp["datetime"],
        y=grp["clicks_per_min"],
        mode="markers",
        marker=dict(color=SITE_COLORS[site], size=6, opacity=0.7),
        name=site,
        text=grp["recording"],             # <-- This enables per-point text
        hovertemplate=(
            f"<b>{site}</b><br>%{{x}}<br>Clicks/min: %{{y:.1f}}<br>Recording: %{{text}}<extra></extra>"
        )
    ))


fig2.update_layout(
    title="Snapping Shrimp Clicks: All Sites Daily Means",
    xaxis_title="Date",
    yaxis_title="Mean clicks per minute",
    yaxis=dict(range=[GLOBAL_Y_MIN, GLOBAL_Y_MAX]),
    barmode="group",
    width=1000, height=500
)
fig2.write_html(os.path.join(OUTPUT_DIR, "all_sites_daily_summary.html"))
print("Saved all-sites daily summary.")
