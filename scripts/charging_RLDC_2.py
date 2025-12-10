import pandas as pd
import plotly.graph_objects as go
import numpy as np
from numba import njit
import sys
sys.path.insert(1,'./')
import scripts.plotting_utils as u

@njit
def cumsum_clip(a, xmin=-np.inf, xmax=np.inf):
    res = np.empty_like(a)
    c = 0
    for i in range(len(a)):
        c = min(max(c + a[i], xmin), xmax)
        res[i] = c
    return res

rqfn = '/Users/peter/My Drive/Programming/FCA_python/cfp_data_analysis/data/ruhnau_qvist_input_time_series.feather'
HOURS_OF_DATA = 35 * 8760  # 35 Years of data
NUM_BINS = 200             # 100 percentile steps
STORAGE_LEVELS = np.array([3, 12, 24, 24*7, 24*30])  # Hours of average demand
OVERBUILD_RANGE = np.arange(0.3, 1.3, 0.05) # Slider from 1.0x to 1.55x

def calculate_scenario(df, overbuild_factor, avg_demand, mix_cf_composite):
    
    total_capacity = (avg_demand / mix_cf_composite.mean()) * overbuild_factor
    generation = total_capacity * mix_cf_composite
    residual_load = df['load'] - generation # >0 Shortfall, <0 Surplus
    
    res_load_norm = residual_load.values/avg_demand
    accumulated_debt = []
    
    for i, sev_t in enumerate(STORAGE_LEVELS):
        accumulated_debt.append(cumsum_clip(res_load_norm, -sev_t, 0))
        
    accumulated_debt = np.stack(accumulated_debt)
    
    res_load_battery = res_load_norm + accumulated_debt
    
    return [res_load_battery[i] for i in range(len(STORAGE_LEVELS))]


# --- Main Analysis Script ---

# 1. Get Data
# df = generate_mock_data()
df = pd.read_feather(rqfn)
avg_demand = df['load'].mean()
mix_cf_composite = 0.75 * df['on'] + 0.25 * df['pv']

battery_cap_labels = [f'< {STORAGE_LEVELS[0]}h Avg Load']
for i in range(len(STORAGE_LEVELS) - 1):
    battery_cap_labels.append(f'{STORAGE_LEVELS[i]}h - {STORAGE_LEVELS[i+1]}h Avg Load')
battery_cap_labels.append(f'> {STORAGE_LEVELS[-1]}h Avg Load')
ordered_battery_caps = battery_cap_labels[::-1] # Stack order

# Color Map
color_palette = u.fca_colorway_v1
color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(ordered_battery_caps)}

# 3. Generate Frames for Animation
frames = []
print("Pre-calculating frames for slider...")

# We store the first frame data separately to initialize the chart
initial_data = None
initial_factor = OVERBUILD_RANGE[0]

for factor in OVERBUILD_RANGE:
    print(f"Processing Overbuild Factor: {factor:.2f}x")
    scenario_data = calculate_scenario(df, factor, avg_demand, mix_cf_composite)
    
    # Create Frame Traces
    frame_traces = []
    for data, cap in zip(scenario_data, ordered_battery_caps):
        frame_traces.append(go.Scatter(
            x=np.arange(len(data)),
            y=data,
            text=cap,
            marker_color=color_map[cap],
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
    trace.name = ordered_battery_caps[i]
    trace.showlegend = True

fig.write_html("shortfall_analysis.html")
fig.show()