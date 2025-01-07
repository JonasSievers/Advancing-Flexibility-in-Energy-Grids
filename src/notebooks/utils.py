import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import gurobipy as gp
from gurobipy import GRB

# Load, PV, Price Data Generation --------------------------------------------------------
def generate_load_data(time_index, random_seed):

    # Set the random seed if provided
    np.random.seed(random_seed)
    
    #Baseload
    base_load = np.random.uniform(1, 1.5)

    # Daily load pattern: higher in the morning and evening
    hour = time_index.hour
    daily_variation = np.where((hour >= 7) & (hour <= 9), 1.2, 1.0)
    daily_variation = np.where((hour >= 18) & (hour <= 21), 1.3, daily_variation)
    
    # Seasonal variation: higher consumption in winter
    day_of_year = time_index.dayofyear
    seasonal_variation = 1 + 0.1 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    
    # Random noise
    noise = np.random.normal(0, 0.1, size=len(time_index))
    
    # Load calculation
    load = base_load * daily_variation * seasonal_variation + noise
    load = np.clip(load, 0, None)  # Ensure load is non-negative
    
    return pd.Series(load, index=time_index)

def generate_price_data(time_index, random_seed):
    """
    Generate realistic dynamic electricity prices for Germany in €/kWh.

    Parameters:
    - time_index (pd.DatetimeIndex): The datetime index for which to generate prices.
    - random_seed (int, optional): Seed for random number generator for reproducibility.

    Returns:
    - pd.Series: Electricity prices in €/kWh indexed by the provided time_index.
    """

    # Initialize random seed
    rng = np.random.default_rng(random_seed)

    # Base price in €/kWh
    base_price = 0.03  # Adjusted baseline for visual clarity

    # Daily pattern: two distinct peaks (morning and evening)
    hours = time_index.hour + time_index.minute / 60.0  # Fractional hours for smooth patterns

    # Morning peak (higher)
    morning_peak = 0.008 * np.exp(-0.5 * ((hours - 8) / 2) ** 2)  # Centered at 8 AM, narrow

    # Evening peak (highest)
    evening_peak = 0.012 * np.exp(-0.5 * ((hours - 19) / 2) ** 2)  # Centered at 7 PM, narrow

    # Combine morning and evening peaks
    daily_pattern = morning_peak + evening_peak

    # Weekly pattern: slightly lower prices on weekends
    day_of_week = time_index.dayofweek  # Monday=0, Sunday=6
    weekend = (day_of_week >= 5).astype(float)  # 1 for Saturday/Sunday, 0 otherwise
    weekly_adjustment = -0.005 * weekend  # Reduce prices on weekends

    # Seasonal trend: higher in winter and summer, lower in spring/autumn
    day_of_year = time_index.dayofyear
    seasonal_amplitude = 0.005  # Reduced amplitude for clarity
    seasonal_pattern = seasonal_amplitude * np.sin((day_of_year - 80) / 365 * 2 * np.pi)

    # Random noise
    noise_std = 0.002  # Reduced noise for clearer patterns
    noise = rng.normal(0, noise_std, size=len(time_index))

    # Combine all components
    prices = base_price + daily_pattern + weekly_adjustment + seasonal_pattern + noise

    # Ensure prices are within a realistic range (e.g., €0.02/kWh to €0.10/kWh)
    #prices = np.clip(prices, 0.02, 0.10)

    # Create a pandas Series
    price_series = pd.Series(prices, index=time_index)
    price_series.name = 'Electricity Price (€ / kWh)'

    return price_series

def generate_pv_data(time_index, random_seed):
    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create an array to hold the PV generation values
    pv_generation = np.zeros(len(time_index))
    
    # Define parameters for the bell curve
    # Random PV capacity between 2 and 5 kW
    peak_capacity = np.random.uniform(2, 5) # Maximum capacity of the PV panel (in kW)
    peak_time = 12      # Time of peak generation (12 PM)
    std_dev = 2         # Standard deviation for the bell curve

    # Generate synthetic PV data following a bell curve
    for i, t in enumerate(time_index):
        if 6 <= t.hour < 19:  # Daytime hours (6 AM to 6 PM)
            # Calculate the hour of the day (0-23)
            hour = t.hour + t.minute / 60
            
            # Calculate Gaussian distribution value
            gaussian_value = np.exp(-((hour - peak_time) ** 2) / (2 * std_dev ** 2))
            
            # Scale it to the peak capacity
            pv_generation[i] = peak_capacity * gaussian_value
            
            # Add random variation to simulate cloud cover
            random_variation = np.random.normal(0, 0.9)  # Random noise with mean=0 and std=10
            pv_generation[i] += random_variation
            
            # Ensure generation is non-negative
            pv_generation[i] = max(0, pv_generation[i])
        else:
            pv_generation[i] = 0  # Nighttime hours

    return pd.Series(pv_generation, index=time_index)

def generate_ems_data(ems_id, start_date, end_date, cong_rebate, cong_weekdays, cong_hours, random_seed):
    # Generate time index
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Generate common price data
    price_data = generate_price_data(time_index, random_seed)
        
    # Generate load data
    load_data = generate_load_data(time_index, random_seed)
        
    # Generate PV data
    pv_data = generate_pv_data(time_index, random_seed)
        
    # Combine data into a DataFrame
    ems_df = pd.DataFrame({
        'Time': time_index,
        'EMS_ID': ems_id,
        'Load': load_data.values,
        'PV_Generation': pv_data.values,
        'Price': price_data.values,
    })
    ems_df['Congestion'] = ((ems_df['Time'].dt.weekday.isin(cong_weekdays)) & (ems_df['Time'].dt.hour.isin(cong_hours))).astype(int)
    ems_df['Valid_Baseline_Day'] = (~ems_df.groupby(ems_df['Time'].dt.date)['Congestion'].transform('any')).astype(int)
    ems_df['FlexPrice'] = ems_df['Congestion'] * cong_rebate
    
    return ems_df

def plot_synthetic_data_v1(data, ems_id, start_date, end_date):

    mask = (data['EMS_ID'] == ems_id) & (data['Time'] >= start_date) & (data['Time'] < end_date)

    plt.figure(figsize=(15, 3))

    # Create the first y-axis
    ax1 = plt.gca()
    ax1.plot(data.loc[mask, 'Time'], data.loc[mask, 'Load'], label='Load')
    ax1.plot(data.loc[mask, 'Time'], data.loc[mask, 'PV_Generation'], label='PV Generation')

    # Create the second y-axis for Price
    ax2 = ax1.twinx()
    ax2.plot(data.loc[mask, 'Time'], data.loc[mask, 'Price'], color="green", label='Price')

    # Set titles and labels
    ax1.set_title(f'EMS_ID {ems_id} Load, PV Generation, and Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (kW)')
    ax2.set_ylabel('Price ($/kWh)')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.show()

# Optimization Problem for BESS Data Generation ------------------------------------------
def precompute_baseline_indices(data, baseline_lookback, max_lookback):

    n_hours_per_day = 24  # Assuming hourly data

    # Extract 'Valid_Baseline_Day' as a NumPy array for faster access
    valid_baseline = data['Valid_Baseline_Day'].to_numpy()

    # Get the actual indices of the DataFrame
    actual_indices = data.index.to_list()

    # Initialize the result dictionary
    baseline_indices = {}

    # Iterate over each index in the DataFrame
    for current_idx in actual_indices:
        past_indices = []
        # Iterate over each lookback day
        for n in range(1, max_lookback + 1):
            shifted_idx = current_idx - n * n_hours_per_day
            if shifted_idx not in actual_indices:
                continue  # Skip if shifted index is out of bounds
            else:
                # Find the position of shifted_idx in the actual_indices list
                shifted_pos = actual_indices.index(shifted_idx)
                if valid_baseline[shifted_pos] != 0:
                    past_indices.append(shifted_idx)
                    
                if len(past_indices) >= baseline_lookback:
                    break  # Collected enough baseline indices

        # Ensure the current index is included if past_indices is empty
        if not past_indices:
            past_indices.append(current_idx)
            
        baseline_indices[current_idx] = past_indices

    return baseline_indices

def build_model(data, baseline_indices, bess_params):
    
    # Create a Gurobi model
    model = gp.Model("BESS_Optimization")

    # Extract time periods
    time_periods = data.index.to_list()

    # Precompute data
    load = data['Load'].to_numpy()
    pv_gen = data['PV_Generation'].to_numpy()
    price = data['Price'].to_numpy()
    flex_price = data['FlexPrice'].to_numpy()
        
    # Decision Variables
    s_power  = model.addVars(time_periods, lb=-bess_params['s_max'], ub=bess_params['s_max'], name="s_power")
    SOC = model.addVars(time_periods, lb=bess_params['soc_min'], ub=bess_params['soc_max'], name="SOC")
    net_consumption = model.addVars(time_periods, lb=-100, ub=100, name="net_consumption")
    baseline = model.addVars(time_periods, lb=-100, ub=100, name="baseline")
    F_pos = model.addVars(time_periods, lb=0, ub=100, name="F_pos")

    # ----------------------------
    # Add Constraints
    # ----------------------------
    
    # SOC Dynamics Constraints
    model.addConstrs(
    (SOC[t] == (bess_params['soc_init'] if t == time_periods[0] else SOC[time_periods[time_periods.index(t) - 1]]) + s_power[t] * bess_params['eta'] for t in time_periods),
        name="SOC_dynamics"
    )   

    model.addConstrs(
        (net_consumption[t] == load[time_periods.index(t)] - pv_gen[time_periods.index(t)] + s_power[t] for t in time_periods),
        name="NetConsumption"
    )

    # Baseline Constraint
    for t in time_periods:
        model.addConstr(
            (baseline[t] == gp.quicksum(net_consumption[p] for p in baseline_indices[t]) / len(baseline_indices[t])),
            name=f"Baseline"
        )

    #Flexibility Constraint
    M=500
    for t in time_periods:
        z = model.addVar(vtype=GRB.BINARY, name=f"z_{t}")
        model.addConstr(F_pos[t] >= baseline[t] - net_consumption[t])
        model.addConstr(F_pos[t] >= 0)
        model.addConstr(F_pos[t] <= (baseline[t] - net_consumption[t]) + M * z)
        model.addConstr(F_pos[t] <= M * (1 - z))
   
    # Objective function
    total_cost = gp.quicksum(
        price[time_periods.index(t)] * (load[time_periods.index(t)] - pv_gen[time_periods.index(t)] + s_power[t])
        - flex_price[time_periods.index(t)] * F_pos[t]
        for t in time_periods
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    return model

def optimize_model(model):
    """
    Optimize the given Gurobi model.
    """
    # Set Gurobi parameters for better performance
    model.setParam('OutputFlag', 0)        # Suppress Gurobi output
    model.setParam('Threads', 8)            # Adjust based on your CPU (e.g., 8 threads for an 8-core CPU)
    model.setParam('Presolve', 2)           # Automatic presolve
    #model.setParam('Cuts', 2)               # Automatic cutting planes
    #model.setParam('Heuristics', 0.1)       # Allocate 10% of time to heuristics
    #model.Params.MIPGap = 0.01  # 1% optimality gap
    
    # Optimize the model
    model.optimize()
    return model

def extract_results(model, data):
    """
    Extract the optimization results and add them to the data DataFrame.
    """
    # Initialize lists to store variable values
    s_power_values = []
    SOC_values = []
    Baseline_values = []
    F_pos_values = []
    NetConsumption_Optimized = []

    # Iterate through each time period to extract variable values
    for t in data.index:
        # Extract s_power
        s_power_var = model.getVarByName(f"s_power[{t}]")
        s_power_values.append(s_power_var.X)
        
        # Extract SOC
        SOC_var = model.getVarByName(f"SOC[{t}]")
        SOC_values.append(SOC_var.X)
               
        # Extract Baseline
        Baseline_var = model.getVarByName(f"baseline[{t}]")
        Baseline_values.append(Baseline_var.X)
        
        # Extract F_pos
        F_pos_var = model.getVarByName(f"F_pos[{t}]")
        F_pos_values.append(F_pos_var.X)

        # Calculate net consumption after optimization
        net_consumption = data.loc[t, 'Load'] - data.loc[t, 'PV_Generation'] + s_power_values[-1]
        NetConsumption_Optimized.append(net_consumption)
    
    # Add extracted values to the DataFrame
    data['s_power'] = s_power_values
    data['SOC'] = SOC_values
    data['Baseline'] = Baseline_values
    data['F_pos'] = F_pos_values
    data['NetConsumption_Optimized'] = NetConsumption_Optimized

    return data

def summarize_results(data, ems_id=1):
    """
    Summarize the optimization results including total costs and savings.
    """
    data = data[data['ems_id'] == ems_id]

    # Total Cost after optimization
    total_cost_optimized = ((data['Price'] * data['NetConsumption_Optimized'] - data['FlexPrice'] * data['F_pos'])).sum()
    # Total Cost without optimization (No BESS actions)
    total_cost_unoptimized = (data['Price'] * (data['Load'] - data['PV_Generation'])).sum()
    # Total Flexibility Revenue
    total_flex_revenue = (data['FlexPrice'] * data['F_pos']).sum()
    # Total Cost Savings
    cost_savings = total_cost_unoptimized - total_cost_optimized
    
    # Display the summary
    print("----- Optimization Summary -----")
    print(f"Total Cost without Optimization: ${total_cost_unoptimized:.2f}")
    print(f"Total Cost after Optimization:    ${total_cost_optimized:.2f}")
    print(f"Total Flexibility Revenue:        ${total_flex_revenue:.2f}")
    print(f"Total Cost Savings:               ${cost_savings:.2f}")
    print("---------------------------------")

def visualize_results(data, ems_id=1):
    """
    Visualize the optimization results including storage power, SOC, net consumption, and flexibility.
    """
    data = data[data['ems_id'] == ems_id]
    
    # Ensure there are at least 100 data points
    delta=100
    num_periods = min(delta+100, len(data))
    data_subset = data.iloc[delta:num_periods].copy()

    # Calculate Cost and Revenue per Iteration
    data_subset['Cost'] = data_subset['Price'] * data_subset['NetConsumption_Optimized']
    data_subset['Revenue'] = data_subset['FlexPrice'] * data_subset['F_pos']
    # Calculate Cumulative Overall Costs
    data_subset['Cumulative_Cost'] = data_subset['Cost'].cumsum() - data_subset['Revenue'].cumsum()
    
    # Identify congestion periods
    congestion_periods = data_subset[data_subset['Congestion'] == 1]['Time']
    
    # Set up the plotting environment with 5 subplots using GridSpec for better layout control
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 2, 2, 2, 2], hspace=0.4)
    
    axs = gs.subplots(sharex=True)
    
    # ----------------------------
    # Plot Storage Power (s_power)
    # ----------------------------
    axs[0].step(data_subset['Time'], data_subset['s_power'], where='post', label='Storage Power', color='blue')
    axs[0].step(data_subset['Time'], data_subset['SOC'], where='post', label='State of Charge', color='orange')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title(f'Storage Charging/Discharging Schedule and SOC (First {num_periods} Periods)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Highlight congestion periods
    for ct in congestion_periods:
        axs[0].axvspan(ct, ct, color='red', alpha=0.9)
       
    # ----------------------------
    # Plot Net Consumption (NetConsumption_Optimized)
    # ----------------------------
    axs[1].step(data_subset['Time'], data_subset['NetConsumption_Optimized'], where='post', label='Net Consumption (Optimized)', color='green')
    axs[1].step(data_subset['Time'], data_subset['Baseline'], where='post', label='Baseline', color='red')
    axs[1].set_ylabel('Load (kW)')
    axs[1].set_title(f'Net Consumption and Baseline Over Time (First {num_periods} Periods)')
    axs[1].legend()
    axs[1].grid(True)

    # Highlight congestion periods
    for ct in congestion_periods:
        axs[1].axvspan(ct, ct, color='red', alpha=0.9)

    # ----------------------------
    # Plot Flexibility (Flexibility and F_pos)
    # ----------------------------
    #axs[2].step(data_subset['Time'], data_subset['Flexibility'], where='post', label='Flexibility', color='green')
    axs[2].step(data_subset['Time'], data_subset['F_pos'], where='post', label='F_pos', color='red')
    axs[2].set_ylabel('Load (kW)')
    axs[2].set_title(f'Flexibility and F_pos Over Time (First {num_periods} Periods)')
    axs[2].legend()
    axs[2].grid(True)
    
    # Highlight congestion periods
    for ct in congestion_periods:
        axs[2].axvspan(ct, ct, color='red', alpha=0.9)
    
    # ----------------------------
    # Plot Cost and Revenue per Iteration
    # ----------------------------
    axs[3].step(data_subset['Time'], data_subset['Cost'], where='post', label='Cost', color='red')
    axs[3].step(data_subset['Time'], data_subset['Revenue'], where='post', label='Revenue', color='purple')
    axs[3].set_ylabel('Amount ($)')
    axs[3].set_title(f'Cost and Revenue per Iteration (First {num_periods} Periods)')
    axs[3].legend()
    axs[3].grid(True)
    
    # Highlight congestion periods
    for ct in congestion_periods:
        axs[3].axvspan(ct, ct, color='red', alpha=0.9)
    
    # ----------------------------
    # Plot Cumulative Overall Costs
    # ----------------------------
    axs[4].plot(data_subset['Time'], data_subset['Cumulative_Cost'], label='Cumulative Cost', color='brown')
    axs[4].set_ylabel('Cumulative Cost ($)')
    axs[4].set_title(f'Cumulative Overall Costs (First {num_periods} Periods)')
    axs[4].legend()
    axs[4].grid(True)
    
    # Highlight congestion periods
    for ct in congestion_periods:
        axs[4].axvspan(ct, ct, color='red', alpha=0.9)
    
    # ----------------------------
    # Formatting the x-axis
    # ----------------------------
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.xlabel('Time')
    
    fig.tight_layout()
    plt.show()

def solve_optimization_iterativly(data, bess_params, baseline_lookback, chunk_size_days, max_lookback):

    ems_id = data['EMS_ID'].unique()
    if len(ems_id) > 1:
        print("Multiple EMS_IDs selected at once!")
    
    start_date = data['Time'].min().date()
    end_date = data['Time'].max().date()

    all_results = []

    current_start_date = start_date
    while current_start_date < (end_date - pd.Timedelta(days=chunk_size_days)):
        current_end_date = min(current_start_date + pd.Timedelta(days=chunk_size_days), end_date)

        chunk_data = data[(data['Time'].dt.date >= current_start_date) & (data['Time'].dt.date < current_end_date)].copy()

        baseline_indices = precompute_baseline_indices(chunk_data, baseline_lookback, max_lookback)
        model = build_model(chunk_data, baseline_indices, bess_params)
        model = optimize_model(model)

        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found for {current_start_date} to {current_end_date}.")
            result_data = extract_results(model, chunk_data)
            all_results.append(result_data)
            #display(final_results)
        elif model.status == GRB.INFEASIBLE:
            print(f"Model is infeasible for {current_start_date} to {current_end_date}.")
            model.computeIIS()
            model.write("model.ilp")
            print("IIS (Irreducible Inconsistent Subsystem) written to 'model.ilp'.")
        else:
            print(f"Optimization ended with status {model.status} for {current_start_date} to {current_end_date}.")

        overlap_days = max_lookback
        current_start_date = current_end_date - pd.Timedelta(days=overlap_days)

    final_results = pd.concat(all_results, ignore_index=False)
    final_results = final_results[~final_results.index.duplicated(keep='first')]
    final_results['EMS_ID'] = ems_id[0]

    return final_results