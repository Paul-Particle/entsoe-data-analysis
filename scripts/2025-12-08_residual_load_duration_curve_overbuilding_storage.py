import pandas as pd
import plotly.graph_objects as go
import numpy as np
from numba import njit
import sys
sys.path.insert(1,'./')
import scripts.plotting_utils as u


# --- Configuration ---
RQFN = '/Users/peter/My Drive/Programming/FCA_python/cfp_data_analysis/data/ruhnau_qvist_input_time_series.feather' # Uncomment to use your file
HOURS_OF_DATA = 35 * 8760
# STORAGE_LEVELS = [3, 12, 24, 24*2, 24*4, 24*8, 24*30]  # Hours of average demand
STORAGE_LEVELS = [12]  # Hours of average demand
OVERBUILD_RANGE = np.arange(1.0, 1.6, 0.05) # Slider values
PLOT_DOWNSAMPLE = 400 # Number of points to plot per line (for performance)

# --- 1. Simulation Logic (Numba Optimized) ---

@njit
def simulate_storage(residual_load, storage_capacity_mwh):
    """
    Simulates a battery operating on a residual load profile.
    
    Args:
        residual_load (np.array): Load - Generation (MW).
                                  >0 implies Deficit (Discharge needed).
                                  <0 implies Surplus (Charge possible).
        storage_capacity_mwh (float): Max energy capacity.
        
    Returns:
        np.array: The new residual load after storage operation.
    """
    n = len(residual_load)
    new_residual = np.zeros(n)
    soc = 0.5 * storage_capacity_mwh # Start at 50% charge
    
    for i in range(n):
        load = residual_load[i]
        
        if load > 0: # DEFICIT: Try to discharge
            discharge = min(load, soc)
            soc -= discharge
            new_residual[i] = load - discharge
            
        elif load < 0: # SURPLUS: Try to charge
            charge_space = storage_capacity_mwh - soc
            charge = min(-load, charge_space) # -load is positive magnitude
            soc += charge
            new_residual[i] = load + charge # load is neg, adding charge moves it toward 0
            
        else:
            new_residual[i] = 0.0
            
    return new_residual

def get_duration_curve(series, n_points=None):
    """Sorts data descending and optionally downsamples for plotting."""
    sorted_data = np.sort(series)[::-1] # Sort descending
    
    if n_points and len(sorted_data) > n_points:
        # Downsample by picking indices evenly spaced
        indices = np.linspace(0, len(sorted_data) - 1, n_points).astype(int)
        return sorted_data[indices]
    
    return sorted_data

def main():
    df = pd.read_feather(RQFN)
    mix_cf = (0.75 * df['on'] + 0.25 * df['pv']).values
    
    # Calculate basics
    avg_demand = df['load'].mean()
    load = df['load'].values
    
    # Pre-calculate curves for all slider steps
    # Structure: results[overbuild_val][storage_level] = [curve_data]
    results = {} 
    
    print(f"Calculating scenarios for {len(OVERBUILD_RANGE)} overbuild steps...")
    
    for ob_factor in OVERBUILD_RANGE:
        ob_factor = round(ob_factor, 2)
        
        # 1. Scale Generation Capacity
        # Total Capacity needed to meet annual demand on average * Overbuild
        total_capacity_mw = (avg_demand / mix_cf.mean()) * ob_factor
        generation = total_capacity_mw * mix_cf
        
        # 2. Raw Residual Load (Load - Gen)
        raw_residual = load - generation
        
        # Store curves for this overbuild factor
        step_curves = {}
        
        # baseline (no renewables)
        step_curves['Load'] = get_duration_curve(load / avg_demand, PLOT_DOWNSAMPLE)
        # Base case (No storage)
        step_curves['No Storage'] = get_duration_curve(raw_residual / avg_demand, PLOT_DOWNSAMPLE)
        
        # Storage cases
        for hours in STORAGE_LEVELS:
            cap_mwh = avg_demand * hours
            res_load_storage = simulate_storage(raw_residual, cap_mwh)
            
            # Normalize by avg demand for easier reading (p.u.)
            curve = get_duration_curve(res_load_storage / avg_demand, PLOT_DOWNSAMPLE)
            if hours < 24: 
                step_curves[f'{hours}h Storage'] = curve
            else:
                step_curves[f'{hours // 24}d Storage'] = curve
            
        results[ob_factor] = step_curves

    # --- 4. Create Plotly Figure ---
    print("Creating interactive plot...")
    
    fig = go.Figure()

    # We need to add traces for the FIRST step initially (visible=True)
    initial_ob = round(OVERBUILD_RANGE[0], 2)
    scenarios = list(results[initial_ob].keys()) # ['No Storage', '3h', '12h'...]
    
    # X-axis (Duration %)
    x_axis = np.linspace(0, 100, PLOT_DOWNSAMPLE)

    # Add all traces for ALL steps, but set visible=False for non-initial ones
    # This is how Plotly sliders work efficiently without callbacks
    
    # To make the slider work, we flatten the logic:
    # We add ALL lines for Step 1, then ALL lines for Step 2...
    # Then the slider just toggles the visibility of the "chunk" of lines corresponding to that step.
    
    # colors = ['black', '#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    colors = u.fca_colorway_v1
    
    for ob_factor in OVERBUILD_RANGE:
        ob_factor = round(ob_factor, 2)
        data_map = results[ob_factor]
        is_visible = (ob_factor == initial_ob)
        
        for i, (label, curve) in enumerate(data_map.items()):
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=curve,
                # mode='lines+markers',
                mode='lines',
                name=label,
                visible=is_visible,
                # marker_opacity=0,
                line=dict(width=2, color=colors[i % len(colors)], shape='spline'),
                legendgroup=label # Keeps legend clean when switching
            ))

    # Create Slider Steps
    steps = []
    num_traces_per_step = len(scenarios)
    total_steps = len(OVERBUILD_RANGE)
    
    for i, ob_factor in enumerate(OVERBUILD_RANGE):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Residual Load Duration Curve (Overbuild: {ob_factor:.2f}x)"}],
            label=f"{ob_factor:.2f}x"
        )
        # Calculate which traces to turn on
        start_idx = i * num_traces_per_step
        for j in range(num_traces_per_step):
            step["args"][0]["visible"][start_idx + j] = True
            
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "RE Overbuild Factor: "},
        pad={"t": 50},
        steps=steps
    )]
    
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,  
            xanchor="left",
            y=1.1, 
            yanchor="top",
            buttons=[
                dict(
                    label="Linear Scale",
                    method="relayout",
                    args=[{"xaxis.type": "linear"}]
                ),
                dict(
                    label="Log Scale",
                    method="relayout",
                    args=[{"xaxis.type": "log"}]
                )
            ]
        )
    ]

    fig.update_layout(
        title=f"Residual Load Duration Curve (Overbuild: {initial_ob:.2f}x)",
        xaxis_title="Duration (% of time)",
        yaxis_title="Residual Load (p.u. of Avg Demand)",
        sliders=sliders,
        updatemenus=updatemenus,
        template=u.fca_template,
        hovermode="x unified",
        legend=dict(y=0.95, x=1.05, xanchor='left', bgcolor='rgba(255,255,255,0.8)')
    )

    fig.write_image('res_load_dur_stor.svg')
    fig.show()

if __name__ == "__main__":
    main()