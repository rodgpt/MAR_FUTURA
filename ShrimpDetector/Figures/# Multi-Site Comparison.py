"""
Snapping Shrimp Temporal Analysis and Multi-Site Comparison
Author: Margherita Silvestri
Date: October 8, 2025

Multi-Site Click Rate Analysis
This script loads detection results from multiple recording sites,
computes clicks per minute, and creates interactive visualizations.

Required packages:
    conda install pandas plotly
"""

import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Define recording sites and assign consistent visualization colors
SITES = ["LASCRUCES", "VENTANAS", "ZAPALLAR"]

SITE_COLORS = {
    "LASCRUCES": "crimson",
    "VENTANAS": "seagreen",
    "ZAPALLAR": "royalblue"
}

# Set base directory containing all recordings
ROOT_BASE = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\RECORDINGS" 

# Create output directory for interactive plots
OUTPUT_DIR = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\plots\interactive\ALL_SITES_HOURLY_ROLLING2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global y-axis limits ensure consistency across all plots
GLOBAL_Y_MIN = 50
GLOBAL_Y_MAX = 800

# ============================================================================
# LOAD AND AGGREGATE DETECTION RESULTS
# ============================================================================

# Initialize empty list to store records
records = []

for site in SITES:
    folder_root = os.path.join(ROOT_BASE, site, "output_all_3")
    folders = glob.glob(os.path.join(folder_root, "*"))
    
    for folder in folders:
        base = os.path.basename(folder)
        parts = base.split("_")
        
        if len(parts) < 3:
            continue
        
        date_str = parts[1]
        time_str = re.sub(r'\D', '', parts[2])
        
        if len(time_str) != 6:
            continue
        
        try:
            dt = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M%S")
        except Exception:
            continue
        
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
                except Exception:
                    pass
        
        cpm = len(starts) / ((starts[-1] - starts[0]) / 60) if len(starts) > 1 else 0
        
        records.append({
            "site": site,
            "datetime": dt,
            "clicks_per_min": cpm,
            "recording": base
        })

# Create pandas DataFrame
df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No data loaded for any site.")

# ============================================================================
# INTERACTIVE FULL TIME-SERIES VISUALIZATION
# ============================================================================

# Create new Plotly figure
fig = go.Figure()

# Add one Scatter trace per site
for site, grp in df.groupby("site"):
    fig.add_trace(go.Scatter(
        x=grp["datetime"],
        y=grp["clicks_per_min"],
        mode="markers",
        marker=dict(color=SITE_COLORS[site], size=6, opacity=0.7),
        name=site,
        text=grp["recording"],
        hovertemplate=(
            f"<b>{site}</b><br>%{{x}}<br>Clicks/min: %{{y:.1f}}"
            "<br>Recording: %{{text}}<extra></extra>"
        )
    ))

# Add nighttime shading (20:00-06:00)
shapes = []
for d in df["datetime"].dt.normalize().unique():
    # Night period 1: 00:00-06:00
    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=d, x1=d + pd.Timedelta(hours=6),
        y0=0, y1=1, fillcolor="navy",
        opacity=0.1, layer="below", line_width=0
    ))
    # Night period 2: 20:00-24:00
    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=d + pd.Timedelta(hours=20),
        x1=d + pd.Timedelta(days=1),
        y0=0, y1=1, fillcolor="navy",
        opacity=0.1, layer="below", line_width=0
    ))

# Update layout with titles, axes, and styling
fig.update_layout(
    title="Snapping Shrimp Clicks: All Sites Full Time Series",
    xaxis_title="Date & Time",
    yaxis_title="Clicks per minute",
    yaxis=dict(range=[GLOBAL_Y_MIN, GLOBAL_Y_MAX]),
    shapes=shapes,
    width=1400,
    height=700
)

# Save interactive plot as HTML
fig.write_html(os.path.join(OUTPUT_DIR, "all_sites_full_ts.html"))
print("Saved all-sites full time series with site colors.")

# ============================================================================
# DAILY SUMMARY ANALYSIS
# ============================================================================

# Add 'date' column to DataFrame
df["date"] = df["datetime"].dt.date

# Compute daily mean clicks per minute for each site
daily = df.groupby(["site", "date"])["clicks_per_min"].mean().reset_index()

# Create new Plotly figure for daily summary
fig2 = go.Figure()

# Add one Scatter trace per site (markers + lines)
for site, grp in daily.groupby("site"):
    fig2.add_trace(go.Scatter(
        x=grp["date"],
        y=grp["clicks_per_min"],
        mode="markers+lines",
        marker=dict(color=SITE_COLORS[site], size=8),
        name=site,
        hovertemplate=(
            f"<b>{site}</b><br>%{{x}}<br>"
            "Mean clicks/min: %{y:.1f}<extra></extra>"
        )
    ))

# Update layout and save
fig2.update_layout(
    title="Snapping Shrimp Clicks: All Sites Daily Means",
    xaxis_title="Date",
    yaxis_title="Mean clicks per minute",
    yaxis=dict(range=[GLOBAL_Y_MIN, GLOBAL_Y_MAX]),
    width=1000,
    height=500
)
fig2.write_html(os.path.join(OUTPUT_DIR, "all_sites_daily_summary.html"))
print("Saved all-sites daily summary.")

# ============================================================================
# HOURLY ROLLING MEAN ANALYSIS
# ============================================================================

# Compute hourly averages
df["hour"] = df["datetime"].dt.floor("H")
hourly = df.groupby(["site", "hour"]).agg({
    "clicks_per_min": "mean"
}).reset_index()

# Compute rolling mean (window = 3 hours)
hourly["rolling_cpm"] = hourly.groupby("site")["clicks_per_min"].transform(
    lambda x: x.rolling(window=3, center=True, min_periods=1).mean()
)

# Create figure with hourly rolling mean
fig3 = go.Figure()

for site, grp in hourly.groupby("site"):
    fig3.add_trace(go.Scatter(
        x=grp["hour"],
        y=grp["rolling_cpm"],
        mode="lines+markers",
        marker=dict(color=SITE_COLORS[site], size=6, opacity=0.7),
        line=dict(width=2),
        name=f"{site} (3-hour rolling mean)",
        hovertemplate=(
            f"<b>{site}</b><br>%{{x}}<br>"
            "Rolling mean CPM: %{y:.1f}<extra></extra>"
        )
    ))

# Add nighttime shading
shapes_rolling = []
for d in hourly["hour"].dt.normalize().unique():
    shapes_rolling += [
        dict(type="rect", xref="x", yref="paper",
             x0=d, x1=d + pd.Timedelta(hours=6), y0=0, y1=1,
             fillcolor="navy", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", xref="x", yref="paper",
             x0=d + pd.Timedelta(hours=20), x1=d + pd.Timedelta(days=1),
             y0=0, y1=1, fillcolor="navy", opacity=0.1, layer="below", line_width=0)
    ]

# Update layout
fig3.update_layout(
    title="Snapping Shrimp Clicks: Hourly Rolling Mean Across Sites (3-hour window)",
    xaxis_title="Date & Hour",
    yaxis_title="Rolling mean clicks per minute",
    yaxis=dict(range=[GLOBAL_Y_MIN, GLOBAL_Y_MAX]),
    shapes=shapes_rolling,
    width=1400,
    height=700
)

# Save interactive plot
fig3.write_html(os.path.join(OUTPUT_DIR, "all_sites_hourly_mean_rolling.html"))
print("Saved all-sites hourly rolling mean plot.")

print("\nAll plots generated successfully!")
