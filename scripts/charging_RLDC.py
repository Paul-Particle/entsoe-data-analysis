import pandas as pd
import plotly.graph_objects as go
import numpy as np
from numba import njit
import sys
sys.path.insert(1,'./')
import scripts.plotting_utils as u
from pathlib import Path

@njit
def cumsum_clip(a, xmin=-np.inf, xmax=np.inf):
    res = np.empty_like(a)
    c = 0
    for i in range(len(a)):
        c = min(max(c + a[i], xmin), xmax)
        res[i] = c
    return res

# --- Configuration ---
rqfn = '/Users/peter/My Drive/Programming/FCA_python/cfp_data_analysis/data/ruhnau_qvist_input_time_series.feather'
HOURS_OF_DATA = 35 * 8760  # 35 Years of data
NUM_BINS = 200             # 100 percentile steps
SEVERITY_THRESHOLDS = np.array([3, 12, 24, 24*7, 24*30])  # Hours of average demand
# OVERBUILD_RANGE = np.arange(0.3, 1.3, 0.05) # Slider from 1.0x to 1.55x
OVERBUILD_RANGE = np.arange(1.2, 1.3, 0.05) # Slider from 1.0x to 1.55x

def calculate_scenario(df, overbuild_factor, avg_demand, mix_cf_composite, sev_labels, sev_bins, ordered_sevs):
    """Runs the shortfall/storage simulation for a specific overbuild factor."""
    
    # 1. Scaling
    total_capacity = (avg_demand / mix_cf_composite.mean()) * overbuild_factor
    generation = total_capacity * mix_cf_composite
    residual_load = df['load'] - generation # >0 Shortfall, <0 Surplus
    
    # 2. Storage Deficit Loop (Optimized)
    # Using numpy array iteration is faster than pandas series
    res_load_norm = residual_load.values/avg_demand
    accumulated_debt = []
    
    # Run simulation
    for i, sev_t in enumerate(SEVERITY_THRESHOLDS):
        accumulated_debt.append(cumsum_clip(res_load_norm, -sev_t, 0))
        
    accumulated_debt = np.stack(accumulated_debt)
    # 3. Binning
    # Shortfall Magnitude Logic
    shortfall_pct_load = (np.clip(residual_load, 0, None) / df['load'].max()) * 100
    
    # Severity Binning
    # Use pd.cut directly on the numpy array for speed
    severity_cats = pd.cut(accumulated_debt, bins=sev_bins, labels=sev_labels)
    
    # Time Percentile Binning
    # Rank descending (Worst shortfall first)
    # We use a temp dataframe for the grouping logic
    temp_df = pd.DataFrame({
        'shortfall_pct_load': shortfall_pct_load,
        'severity_bin': severity_cats
    })
    
    # Rank and Quantile Cut
    temp_df['sorted_rank'] = temp_df['shortfall_pct_load'].rank(method='first', ascending=False)
    temp_df['quantile_bin'] = pd.cut(temp_df['sorted_rank'], bins=NUM_BINS, labels=False)
    
    # 4. Aggregation
    grouped = temp_df.groupby(['quantile_bin', 'severity_bin'], observed=False).agg(
        hours=('shortfall_pct_load', 'count')
    ).reset_index()
    
    # Bin Heights (Avg Shortfall Power)
    bin_stats = temp_df.groupby('quantile_bin')['shortfall_pct_load'].mean().reset_index(name='bin_height_power')
    grouped = pd.merge(grouped, bin_stats, on='quantile_bin')
    
    # 5. Format for Plotting
    # We need to guarantee the data shape matches the visualization requirements
    quantile_order = range(NUM_BINS)
    traces_data = {sev: {'x': [], 'y': [], 'text': []} for sev in ordered_sevs}
    
    for q_bin in quantile_order:
        bin_data = grouped[grouped['quantile_bin'] == q_bin]
        
        if bin_data.empty:
             # Add zero-height placeholder to keep arrays aligned
            for sev in ordered_sevs:
                traces_data[sev]['x'].append(q_bin)
                traces_data[sev]['y'].append(0)
                traces_data[sev]['text'].append("")
            continue
            
        total_hours_in_bin = bin_data['hours'].sum()
        bin_height = bin_data['bin_height_power'].iloc[0] 
        
        pct_start = (q_bin / NUM_BINS) * 100
        pct_end = ((q_bin + 1) / NUM_BINS) * 100
        
        for sev in ordered_sevs:
            sev_row = bin_data[bin_data['severity_bin'] == sev]
            h = sev_row.iloc[0]['hours'] if not sev_row.empty else 0
            
            # Segment Height
            segment_height = bin_height * (h / total_hours_in_bin) if total_hours_in_bin > 0 else 0
            
            traces_data[sev]['x'].append(q_bin)
            traces_data[sev]['y'].append(segment_height)
            traces_data[sev]['text'].append(
                f"<b>Time Percentile: {pct_start:.0f}% - {pct_end:.0f}%</b><br>" +
                f"Severity: {sev}<br>" +
                f"Hours in bin: {h}<br>" +
                f"Avg Shortfall: {bin_height:.1f}%"
            )
            
    return traces_data

# --- Main Analysis Script ---

# 1. Get Data
# df = generate_mock_data()
df = pd.read_feather(rqfn)
avg_demand = df['load'].mean()
mix_cf_composite = 0.75 * df['on'] + 0.25 * df['pv']

print(f"Average Demand: {avg_demand:.2f} GW")

# 2. Setup Constants for Binning
sev_bins = [-1] + [t * avg_demand for t in SEVERITY_THRESHOLDS] + [float('inf')]
sev_labels = [f'< {SEVERITY_THRESHOLDS[0]}h Avg Load']
for i in range(len(SEVERITY_THRESHOLDS) - 1):
    sev_labels.append(f'{SEVERITY_THRESHOLDS[i]}h - {SEVERITY_THRESHOLDS[i+1]}h Avg Load')
sev_labels.append(f'> {SEVERITY_THRESHOLDS[-1]}h Avg Load')
ordered_sevs = sev_labels[::-1] # Stack order

# Color Map
# color_palette = ['#d53e4f', '#f46d43', '#fdae61', '#66c2a5', '#3288bd']
color_palette = u.fca_colorway_v1
color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(ordered_sevs)}

# 3. Generate Frames for Animation
frames = []
print("Pre-calculating frames for slider...")

# We store the first frame data separately to initialize the chart
initial_data = None
initial_factor = OVERBUILD_RANGE[0]

for factor in OVERBUILD_RANGE:
    print(f"Processing Overbuild Factor: {factor:.2f}x")
    scenario_data = calculate_scenario(df, factor, avg_demand, mix_cf_composite, sev_labels, sev_bins, ordered_sevs)
    
    # Create Frame Traces
    frame_traces = []
    for sev in ordered_sevs:
        frame_traces.append(go.Bar(
            x=scenario_data[sev]['x'],
            y=scenario_data[sev]['y'],
            text=scenario_data[sev]['text'],
            marker_color=color_map[sev],
            textposition='none', # Important: Hide text on bars to improve speed/readability
            hoverinfo='text'     # Only show text on hover
        ))
        
    frames.append(go.Frame(
        data=frame_traces,
        name=f"{factor:.2f}" # Name matches slider value
    ))
    
    if abs(factor - initial_factor) < 0.001:
        initial_data = frame_traces

# 4. Create Figure
fig = go.Figure(
    data=initial_data,
    frames=frames
)

# 5. Configure Layout & Slider
tick_step = 10
tick_vals = list(range(0, NUM_BINS + 1, int(NUM_BINS / (100/tick_step))))
tick_text = [f"{int(v * 100 / NUM_BINS)}%" for v in tick_vals]

# Slider Steps
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Overbuild Factor: "},
    pad={"t": 50},
    steps=[dict(
        method="animate",
        args=[[f"{f:.2f}"], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=100))],
        label=f"{f:.2f}x"
    ) for f in OVERBUILD_RANGE]
)]

fig.update_layout(
    title=f"Animated Shortfall Duration Curve<br><sub>Adjust Overbuild Factor to see impact on Storage Needs</sub>",
    xaxis=dict(
        title="Percentile of Time (%)",
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        range=[-0.5, NUM_BINS - 0.5]
    ),
    yaxis=dict(
        title="Shortfall Power Magnitude (% of Load)",
        range=[0, 105]
    ),
    barmode='stack',
    bargap=0, 
    template=u.fca_template,
    legend_title="Required Storage Capacity",
    hovermode="x unified",
    sliders=sliders,
    # updatemenus=[dict(
    #     type="buttons",
    #     showactive=False,
    #     x=0.05, y=1.15,
    #     buttons=[dict(
    #         label="Play",
    #         method="animate",
    #         args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
    #     ), dict(
    #         label="Pause",
    #         method="animate",
    #         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
    #     )]
    # )]
)

# Fix Trace Names for Legend (Initial data doesn't carry names automatically in this construction)
for i, trace in enumerate(fig.data):
    trace.name = ordered_sevs[i]
    trace.showlegend = True

fig.write_html("shortfall_analysis.html")
fig.show()