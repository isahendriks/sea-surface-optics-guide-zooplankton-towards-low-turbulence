"""Python translation of compiled_code_for_publication.r.

The script keeps the original staged workflow:
1. Load ADCP backscatter data
2. Detect ascent and descent transitions
3. Estimate slopes around the transitions
4. Join weather and wave data
5. Build the publication figure
"""

#%% Load packages
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from scipy.stats import fit, linregress
from statsmodels.formula.api import ols
from typing import cast

from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

import seaborn as sns


# Match the plotting style used in the other Python scripts in the repository.
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8


def mm_to_inch(mm: float) -> float:
    return mm / 25.4

def find_transitions(day_df: pd.DataFrame, threshold_factor: float = 0.5, min_run_len: int = 5):
    d1 = day_df["d1"].to_numpy()
    vals = day_df["smooth"].to_numpy()
    times = day_df["time"].to_numpy()

    thr = np.nanstd(d1) * threshold_factor
    rising = np.where(np.isnan(d1), False, d1 > thr)
    falling = np.where(np.isnan(d1), False, d1 < -thr)

    def extract_runs(mask: np.ndarray, direction: str):
        run_rows = []
        idx = 0
        while idx < len(mask):
            if not mask[idx]:
                idx += 1
                continue

            start = idx
            while idx < len(mask) and mask[idx]:
                idx += 1
            end = idx - 1

            if (end - start + 1) >= min_run_len:
                d_val = vals[end] - vals[start]
                if direction == "fall":
                    d_val = - d_val
                run_rows.append(
                    {
                        "start_idx": start,
                        "end_idx": end,
                        "time": times[start],
                        "change": d_val,
                    }
                )

        return pd.DataFrame(run_rows)

    rise_runs = extract_runs(rising, "rise")
    fall_runs = extract_runs(falling, "fall")

    return {
        "all_rise": rise_runs,
        "all_fall": fall_runs,
        "rising_best": rise_runs.loc[rise_runs["change"].idxmax()] if not rise_runs.empty else None,
        "falling_best": fall_runs.loc[fall_runs["change"].idxmax()] if not fall_runs.empty else None,
    }

def add_regression_ci(ax, x, y, x_line, slope, intercept, color="0.8", alpha=0.5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    y_hat = intercept + slope * x_line

    n = len(x)
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean) ** 2)

    # residual standard error
    resid = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))

    # 95% CI for the mean fitted line
    t_val = stats.t.ppf(0.975, df=n - 2)
    ci = t_val * s_err * np.sqrt((1 / n) + ((x_line - x_mean) ** 2 / ssx))

    ax.fill_between(x_line, y_hat - ci, y_hat + ci, color=color, alpha=alpha, linewidth=0, zorder=1)


#%% Define paths and load data
base_dir = Path(__file__).resolve().parent
mat_path = base_dir / "Trubadur_ADCP_beam5_m30sec.mat"
weather_path = base_dir / "r_weather.csv"
wave_path = base_dir / "w_truba.csv"

# Read weather data
weatherdata = pd.read_csv(weather_path, sep=";", decimal=",")
weatherdata["date"] = pd.to_datetime(weatherdata["date"], format="%Y.%m.%d")
weatherdata["time_utc"] = pd.to_timedelta(weatherdata["time_utc"].astype(str))
weatherdata["datetime"] = weatherdata["date"] + weatherdata["time_utc"]  

# Read wave data
wavedata = pd.read_csv(wave_path)
wavedata["time"] = pd.to_datetime(wavedata["time"])

# Read ADCP backscatter data
heat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

# adjust timezones
heat["tm"] = heat["tm"]  # already in CEST
wavedata["time"] = wavedata["time"] # already CEST
weatherdata["datetime"] = weatherdata["datetime"] + pd.Timedelta(hours=2)  # UTC to CEST

heat["tm"] = pd.to_datetime((np.asarray(heat["tm"]) - 719529) * 86400, unit="s")

time= heat["tm"]
val = pd.DataFrame(np.asarray(heat["a5m"]))
depth = np.asarray(heat["z"]).astype(float)

df_wide = pd.DataFrame({"time": time})
for idx, depth_value in enumerate(depth):
    df_wide[str(depth_value)] = val.iloc[:, idx].to_numpy()

heat2 = df_wide.melt(id_vars="time", var_name="depth", value_name="value")
heat2["depth"] = heat2["depth"].astype(float)

heat2slice = heat2.loc[(heat2["depth"] < -9) & (heat2["depth"] > -21)].copy()
heat2slicetotval = heat2slice.groupby("time", as_index=False)["value"].sum()

#%% Derivative, ascent/descent events
df = heat2slicetotval.copy()

df["time"] = pd.to_datetime(df["time"])
df = df.iloc[np.argsort(df["time"].to_numpy(dtype="datetime64[ns]"))].reset_index(drop=True)

# Smoothen the data using a rolling median with a window of 241 samples (approximately 2 hours)
df["smooth"] = df["value"].rolling(window=241, center=True, min_periods=241).median()
df["d1"] = df["smooth"].diff() # Calculate the first derivative of the smoothed data
df["day"] = df["time"].dt.date # Group the data by day

# Identify the ascent and descent events for each day
results_evening_rows = []
results_morning_rows = []

for day_value, day_df in df.groupby("day", sort=True):
    day_frame = cast(pd.DataFrame, day_df)
    out = find_transitions(day_frame)

    if not out["all_rise"].empty:
        temp = out["all_rise"].copy()
        temp.insert(0, "day", day_value)
        temp = temp.rename(columns={"time": "evening_onset", "change": "evening_change"})
        results_evening_rows.append(temp)

    if not out["all_fall"].empty:
        temp = out["all_fall"].copy()
        temp.insert(0, "day", day_value)
        temp = temp.rename(columns={"time": "morning_onset", "change": "morning_change"})
        results_morning_rows.append(temp)

results_evening = pd.concat(results_evening_rows, ignore_index=True) if results_evening_rows else pd.DataFrame()
results_morning = pd.concat(results_morning_rows, ignore_index=True) if results_morning_rows else pd.DataFrame()

# Manual exclusions 
if not results_morning.empty:
    manual_drop_morning = [184, 185, 192, 193, 194, 195, 201]
    results_morning = results_morning.drop(index=[idx for idx in manual_drop_morning if idx in results_morning.index])
    results_morning = results_morning.reset_index(drop=True)

if not results_evening.empty:
    manual_drop_evening = [122, 200]
    results_evening = results_evening.drop(index=[idx for idx in manual_drop_evening if idx in results_evening.index])
    results_evening = results_evening.reset_index(drop=True)

# Select the maximum change event for each day and prepare the final results
if not results_evening.empty:
    results_evening = results_evening.loc[results_evening.groupby("day")["evening_change"].idxmax()].reset_index(drop=True)
    results_evening = results_evening.rename(columns={"evening_onset": "onset", "evening_change": "change"})
    results_evening["type"] = "rise"

if not results_morning.empty:
    results_morning = results_morning.loc[results_morning.groupby("day")["morning_change"].idxmax()].reset_index(drop=True)
    results_morning = results_morning.rename(columns={"morning_onset": "onset", "morning_change": "change"})
    results_morning["type"] = "fall"

events = pd.concat([results_evening, results_morning], ignore_index=True, sort=False)

print(f"Total number of events detected: {len(events)}")
print(f"Columns in final dataframe: {events.columns.tolist()}")


#%% Slopes of the smoothed line around the transition events
window_minutes = 15
delta_seconds = 15 * 60

df_slopes_rows = []
for _, row in events.iterrows():
    lower = row["onset"] - pd.Timedelta(minutes=window_minutes)
    upper = row["onset"] + pd.Timedelta(minutes=window_minutes)

    local = df.loc[df["time"].between(lower, upper)].copy()
    if local.empty:
        continue

    x_seconds = (local["time"] - row["onset"]).dt.total_seconds().to_numpy()
    y_values = local["smooth"].to_numpy()
    slope = cast(float, linregress(x_seconds, y_values)[0])

    smooth_at_onset = df.loc[df["time"] == row["onset"], "smooth"]

    if smooth_at_onset.empty:
        continue

    smooth_value = float(smooth_at_onset.iloc[0])
    x_start = row["onset"] - pd.Timedelta(seconds=delta_seconds)
    x_end = row["onset"] + pd.Timedelta(seconds=delta_seconds)
    y_start = smooth_value - slope * delta_seconds
    y_end = smooth_value + slope * delta_seconds

    # calculate the slope of the line connecting the start and end points
    line = np.array([[x_start.value, y_start], [x_end.value, y_end]])
    slope_window = (y_end - y_start) / ((x_end - x_start).total_seconds())

    df_slopes_rows.append(
        {
            "onset": row["onset"],
            "smooth": smooth_value,
            "slope_point": slope,
            "type": row["type"],
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_end,
            "y_end": y_end,
            "slope_window": slope_window
        }
    )

df_slopes = pd.DataFrame(df_slopes_rows)

#%% Integrate the area under the curve for each transition event
values = df["value"].to_numpy()
values_smooth = df["smooth"].to_numpy()
time_hours = (df["time"] - df["time"].min()).dt.total_seconds().to_numpy() / 3600.0
# For each date, find the time closest to 12:00:00 and store the index of that time in a list

midday_points_idx = []
for day_value, day_df in df.groupby("day", sort=True):
    midday_time = pd.Timestamp(day_value) + pd.Timedelta(hours=12)
    closest_time_idx = (day_df["time"] - midday_time).abs().idxmin()
    midday_points_idx.append(closest_time_idx)

base_period_hours = 24.0
n_waves = 10
omega = 2 * np.pi / base_period_hours

basis = [np.ones_like(time_hours)]
for harmonic in range(1, n_waves + 1):
    basis.append(np.sin(harmonic * omega * time_hours))
    basis.append(np.cos(harmonic * omega * time_hours))

design_matrix = np.column_stack(basis)

finite_mask = np.isfinite(values) & np.isfinite(time_hours)
coefficients, *_ = np.linalg.lstsq(design_matrix[finite_mask], values[finite_mask], rcond=None)
carrier = design_matrix @ coefficients

# Find max peaks in the smoothed values to construct the upper envelope
min_peak_distance = 18 / (time_hours[1]-time_hours[0])  
peak_idx, _ = find_peaks(values_smooth, distance=min_peak_distance, prominence=np.nanstd(values_smooth) * 0.1)

max_peak_times = time_hours[peak_idx]
max_peak_vals = values_smooth[peak_idx]

upper_envelope_spline = CubicSpline(max_peak_times, max_peak_vals, bc_type="natural", extrapolate=True)
upper_envelope = upper_envelope_spline(time_hours)
upper_envelope = pd.Series(upper_envelope).rolling(window=241, center=True, min_periods=1).median().to_numpy()
upper_envelope = np.clip(upper_envelope, 0, None)

# find min peaks in the smoothed values to construct the lower envelope
min_peak_distance = 18 / (time_hours[1]-time_hours[0])  
min_peak_idx, _ = find_peaks(-values_smooth, distance=min_peak_distance, prominence=np.nanstd(values_smooth) * 0.1)

min_peak_times = time_hours[min_peak_idx]
min_peak_vals = values_smooth[min_peak_idx]

# interpolate the missing min peak times
t0 = min_peak_times[0]
expected = t0 + 24 * np.arange(27)

tol = 10  # hours
for te in expected:
    nearest = min_peak_times[np.argmin(np.abs(min_peak_times - te))]
    if abs(nearest - te) <= tol:
        # print(f"expected {te:.2f} -> matched {nearest:.2f}")
        pass
    else:
        # print(f"expected {te:.2f} -> missing")
        # print(f"Adding missing min peak at {te:.2f}")
        min_peak_times = np.append(min_peak_times, te)
        index = np.argmin(np.abs(time_hours - te))
        # val = smooth_lookup.iloc[index]
        min_peak_vals = np.append(min_peak_vals, values_smooth[index])
        min_peak_idx = np.append(min_peak_idx, index)

# sort the min peaks after adding missing ones
sorted_idx = np.argsort(min_peak_times)
min_peak_times = min_peak_times[sorted_idx]
min_peak_vals = min_peak_vals[sorted_idx]
min_peak_idx = min_peak_idx[sorted_idx]

lower_envelope_spline = CubicSpline(min_peak_times, min_peak_vals, bc_type="natural", extrapolate=True)
lower_envelope = lower_envelope_spline(time_hours)
lower_envelope = pd.Series(lower_envelope).rolling(window=241, center=True, min_periods=1).median().to_numpy()
lower_envelope = np.clip(lower_envelope, 0, None)

envelope = (upper_envelope + lower_envelope) / 2.0

# normalize carrier to +/-1 before modulation
carrier_norm = carrier / np.nanmax(np.abs(carrier))

# shift the modulation so the final curve is centered on the envelope
final_fit = envelope * carrier_norm 
final_fit = final_fit + (np.nanmean(envelope) - np.nanmean(final_fit))

weatherdata_time_hrs = (weatherdata["datetime"] - df["time"].min()).dt.total_seconds() / 3600.0
wavedata_time_hrs = (wavedata["time"] - df["time"].min()).dt.total_seconds() / 3600.0

# plot 
fig, axs = plt.subplots(2, 1, figsize=(10,8), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

ax1 = axs[0]
ax2 = axs[1]

### First panel: Significant wave height and wind speed
ax1.plot(wavedata_time_hrs, wavedata["smooth"], color="black", linewidth=0.8, label="Significant wave height")
ax1.set_ylabel("Significant wave height [m]", color="black")
ax1.tick_params(axis="y", labelcolor="black")

ax1_right = ax1.twinx()

ax1_right.plot(weatherdata_time_hrs, weatherdata["wind_speed_m_s"], color="steelblue", linewidth=0.8, label="Wind speed")
ax1_right.set_ylabel("Wind speed [m/s]", color="steelblue")
ax1_right.tick_params(axis="y", labelcolor="steelblue")

ax1.set_xticks(np.arange(0, np.max(time_hours), 24))
ax1.set_xticklabels([])
ax1.grid()

### Second panel: 
ax2.scatter(time_hours, values, label="Original", color="lightgray", s = 0.1)
ax2.plot(time_hours, values_smooth, label="Smoothed", color="purple", linewidth=0.8)
ax2.plot(time_hours, lower_envelope, label="Lower Envelope", color="orange", linewidth=0.8)
# ax2.scatter(min_peak_times, min_peak_vals, label="Min Peaks", color="blue", s=15)
# ax2.scatter(time_hours[midday_points_idx], values_smooth[midday_points_idx], label="Midday Points", color="red", s=15)

# shade the area under the curve for each peak and write the AUC value on the plot
auc_values = []
for idx in range(0, len(midday_points_idx)-1):

    if idx == 0:
        peak_start_idx = 121
        peak_end_idx = midday_points_idx[idx+1]
    elif idx == len(midday_points_idx) - 2:
        peak_start_idx = midday_points_idx[idx]
        peak_end_idx = midday_points_idx[idx+1] - 121
    else:
        peak_start_idx = midday_points_idx[idx]
        peak_end_idx = midday_points_idx[idx + 1]

    peak = values_smooth[peak_start_idx:peak_end_idx + 1]
    lower = lower_envelope[peak_start_idx:peak_end_idx + 1]
    peak = peak - lower
    for i in range(len(peak)):
        if peak[i] < 0:
            peak[i] = 0

    ax2.fill_between(time_hours[peak_start_idx:peak_end_idx + 1], lower, values_smooth[peak_start_idx:peak_end_idx + 1], color="orange", alpha=0.3)
    
    # write the AUC value on the plot
    auc = np.trapezoid(peak, time_hours[peak_start_idx:peak_end_idx + 1])
    auc_values.append(auc)
    auc_text = f"{auc:.0f}"
    text_x = time_hours[peak_start_idx] + (time_hours[peak_end_idx] - time_hours[peak_start_idx]) / 2
    text_y = np.nanmax(values_smooth[peak_start_idx:peak_end_idx + 1]) + 10
    ax2.text(text_x, text_y, auc_text, fontsize=8, ha="center", va="bottom", color="black")
    # ax2.vlines(x=time_hours[peak_start_idx], ymin=480, ymax=800, color="red", linestyle="--", linewidth=1)
       

ax2.set_xlabel("Date")
ax2.set_xticks(np.arange(0, np.max(time_hours), 24))
ax2.set_xticklabels([(df["time"].min() + pd.Timedelta(hours=hr)).strftime("%d/%m") for hr in np.arange(0, np.max(time_hours), 24)], rotation=45)
ax2.set_xlim(0, np.max(time_hours))
ax2.set_ylabel("Backscatter Value")
ax2.set_ylim(480, 800)
# ax2.set_title("Harmonic Fit and Envelope of Backscatter Data")
ax2.grid()
ax2.legend()

plt.figure(figsize=(5, 2))
plt.scatter(range(len(auc_values)), auc_values, marker="o", color="orange")
plt.xlabel("Peak Index")
plt.ylabel("AUC")
plt.title("Area Under the Curve for Each Peak")
plt.grid()

df["sin_fit"] = carrier
df["envelope"] = lower_envelope
df["final_fit"] = final_fit

#%% Weather and wave data

### Match weather data with slope events within a 2-hour window around the onset time
mean_windspeed_window_slope = []
mean_wave_window_slope = []

for _, slope_row in df_slopes[df_slopes["type"] == "rise"].iterrows():

    onset = slope_row["onset"]
    
    # Calculate the mean wind speed in window closest measurement +/- 1 hour
    closest_weather_time = weatherdata.loc[(weatherdata["datetime"] - onset).abs().idxmin(), "datetime"]
    weather_window_start = closest_weather_time - pd.Timedelta(hours=1)
    weather_window_end = closest_weather_time + pd.Timedelta(hours=1)
    weather_observations_in_window = weatherdata.loc[weatherdata["datetime"].between(weather_window_start, weather_window_end)].copy()
    mean_wind_speed = weather_observations_in_window["wind_speed_m_s"].mean()
    mean_windspeed_window_slope.append(mean_wind_speed)
    
    ### Calculate the mean wave height in the slope window
    t_start = slope_row['x_start']
    t_end = slope_row['x_end']

    # Find closest wave measurements to the slope window start and end times
    wave_window_start = wavedata.loc[(wavedata["time"] - t_start).abs().idxmin(), "time"]
    wave_window_end = wavedata.loc[(wavedata["time"] - t_end).abs().idxmin(), "time"]

    # Find the wave observations within the slope window
    wave_observations_in_window = wavedata.loc[wavedata["time"].between(wave_window_start, wave_window_end)].copy()
    
    # Calculate mean wave height within the window
    mean_wave_height = wave_observations_in_window["smooth"].mean()
    mean_wave_window_slope.append(mean_wave_height)

    # print(F" Onset: {onset}, wave window: {wave_window_start} to {wave_window_end} ({len(wave_observations_in_window)} observations), mean wave height: {mean_wave_height:.2f}")
    # print(f"Onset: {onset}, Weather window: {weather_window_start} to {weather_window_end} ({len(weather_observations_in_window)} observations), wave height window: {slope_row['x_start']} to {slope_row['x_end']}")

# Linear regression for mean wind speed vs slope and mean wave height vs slope
mean_windspeed_slope_window = np.array(mean_windspeed_window_slope)
mean_wave_slope_window = np.array(mean_wave_window_slope)

slope_values = df_slopes.loc[df_slopes["type"] == "rise", "slope_window"].to_numpy()

# print results in a table format
comparison = pd.DataFrame({
    "onset": df_slopes[df_slopes["type"] == "rise"]["onset"].values,
    "wind": mean_windspeed_slope_window,
    "wave": mean_wave_slope_window,
    "slope": slope_values,
})

# Linear regression for mean wind speed vs slope and mean wave height vs slope
wind_slope_model = linregress(mean_windspeed_slope_window, slope_values)
wave_slope_model = linregress(mean_wave_slope_window, slope_values)

# Match weather data with integrated peak areas 
mean_windspeed_window_auc = []
mean_wave_window_auc = []
for idx in range(0, len(midday_points_idx)-1):

    if idx == 0:
        peak_start_idx = 121
        peak_end_idx = midday_points_idx[idx+1]
    elif idx == len(midday_points_idx) - 2:
        peak_start_idx = midday_points_idx[idx]
        peak_end_idx = midday_points_idx[idx+1] - 121
    else:
        peak_start_idx = midday_points_idx[idx]
        peak_end_idx = midday_points_idx[idx + 1]

    t_start = df["time"].iloc[peak_start_idx]
    t_end = df["time"].iloc[peak_end_idx]

    # Calculate mean weather values within the peak time window
    weather_window = weatherdata.loc[weatherdata["datetime"].between(t_start, t_end)].copy()
    mean_windspeed_window_auc.append(weather_window["wind_speed_m_s"].mean()) 

    # calculate mean wave values within the peak time window
    wave_window = wavedata.loc[wavedata["time"].between(t_start, t_end)].copy()
    mean_wave_window_auc.append(wave_window["smooth"].mean())

# Linear regression for mean wind speed vs AUC
mean_windspeed_auc_window = np.array(mean_windspeed_window_auc)
mean_wave_auc_window = np.array(mean_wave_window_auc)
auc_values = np.array(auc_values)

wind_auc_model = linregress(mean_windspeed_auc_window, auc_values)
wave_auc_model = linregress(mean_wave_auc_window, auc_values)

def adj_r_squared(r_squared, n, k):
    return 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

# Print the regression results
print("\n=== Wind speed vs AUC regression ===")
print(f"P-value: {wind_auc_model.pvalue:.4f}")
print(f"Adjusted R-squared: {adj_r_squared(wind_auc_model.rvalue**2, len(mean_windspeed_auc_window), 1):.4f}")

print("\n=== Wave height vs AUC regression ===")
print(f"P-value: {wave_auc_model.pvalue:.4f}")
print(f"Adjusted R-squared: {adj_r_squared(wave_auc_model.rvalue**2, len(mean_wave_auc_window), 1):.4f}")

print("\n=== Wind speed vs Slope regression ===")
print(f"P-value: {wind_slope_model.pvalue:.4f}")
print(f"Adjusted R-squared: {adj_r_squared(wind_slope_model.rvalue**2, len(mean_windspeed_slope_window), 1):.4f}")

print("\n=== Wave height vs Slope regression ===")
print(f"P-value: {wave_slope_model.pvalue:.4f}")
print(f"Adjusted R-squared: {adj_r_squared(wave_slope_model.rvalue**2, len(mean_wave_slope_window), 1):.4f}")

# Make four subplots with the regression results
plt.figure(figsize=(10, 8), constrained_layout=True)
plt.subplot(2, 2, 1)
plt.scatter(mean_windspeed_auc_window, auc_values, color="orange", label="Data points")
plt.plot(mean_windspeed_auc_window, wind_auc_model.intercept + wind_auc_model.slope * mean_windspeed_auc_window, color="blue", label="Fit line")
plt.grid()
plt.xlabel("Mean Wind Speed [m/s]")
plt.ylabel("AUC")
plt.title(f"Wind Speed vs AUC, p={wind_auc_model.pvalue:.4f}")

plt.subplot(2, 2, 2)
plt.scatter(mean_wave_auc_window, auc_values, color="orange", label="Data points")
plt.plot(mean_wave_auc_window, wave_auc_model.intercept + wave_auc_model.slope * mean_wave_auc_window, color="blue", label="Fit line")
plt.grid()
plt.xlabel("Mean Wave Height [m]")
plt.ylabel("AUC")
plt.title(f"Wave Height vs AUC, p={wave_auc_model.pvalue:.4f}")

plt.subplot(2, 2, 3)
plt.scatter(mean_windspeed_slope_window, slope_values, color="orange", label="Data points")
plt.plot(mean_windspeed_slope_window, wind_slope_model.intercept + wind_slope_model.slope * mean_windspeed_slope_window, color="blue", label="Fit line")
plt.grid()
plt.xlim(0, 22)
plt.ylim(0,0.06)
plt.xlabel("Mean Wind Speed [m/s]")
plt.ylabel("Slope")
plt.title(f"Wind Speed vs Slope, p={wind_slope_model.pvalue:.4f}")

plt.subplot(2, 2, 4)
plt.scatter(mean_wave_slope_window, slope_values, color="orange", label="Data points")
plt.plot(mean_wave_slope_window, wave_slope_model.intercept + wave_slope_model.slope * mean_wave_slope_window, color="blue", label="Fit line")
plt.grid()
plt.xlim(0,3)
plt.ylim(0,0.06)
plt.xlabel("Mean Wave Height [m]")
plt.ylabel("Slope")
plt.title(f"Wave Height vs Slope, p={wave_slope_model.pvalue:.4f}")

#%% Plotting

def rescale_to_depth(x):
    return (x - s_min) / (s_max - s_min) * (d_max - d_min) + d_min

# Define analysis window
analysis_start = pd.Timestamp("2018-08-29 12:00:00") 
analysis_end = pd.Timestamp("2018-08-31 12:00:00")

sub = heat2.loc[heat2["time"].between(analysis_start, analysis_end)].copy()
sub = sub.loc[sub["depth"].between(-20.5, -9.5)].copy()

# set fontsize for all axes
plt.rcParams.update({"font.size": 8})
plt.rcParams.update({"axes.labelsize": 8})
plt.rcParams.update({"xtick.labelsize": 7})
plt.rcParams.update({"ytick.labelsize": 7})
plt.rcParams.update({"axes.titlesize": 8})

# reduce space between xticks and xlabels
plt.rcParams.update({"xtick.major.pad": 0.8})
plt.rcParams.update({"ytick.major.pad": 0.8})

# reduce space between axes and labels
plt.rcParams.update({"axes.labelpad": 0.4})

# disable the box around the plots
plt.rcParams.update({"axes.spines.top": False})
plt.rcParams.update({"axes.spines.right": False})
plt.rcParams.update({"axes.spines.left": False})
plt.rcParams.update({"axes.spines.bottom": False})

# convert smooth values to depth scale for plotting
d_min = float(sub["depth"].min())
d_max = float(sub["depth"].max())
s_min = float(df["smooth"].min())
s_max = float(df["smooth"].max())

d_min_w = float(heat2["depth"].min())
d_max_w = float(heat2["depth"].max())

# Depth range occupied by the integrated echo line
depth_min = -20.5
depth_max = -9.5

def intensity_to_depth(y):
    return (y - s_min) / (s_max - s_min) * (depth_max - depth_min) + depth_min

def depth_to_intensity(y):
    return (y - depth_min) / (depth_max - depth_min) * (s_max - s_min) + s_min

# Rescale smooth values to depth scale for plotting
df["scaled_smooth"] = rescale_to_depth(df["smooth"])
df["scaled_smooth_w"] = rescale_to_depth(df["smooth"])
df["sin_fit_scaled"] = rescale_to_depth(df["sin_fit"])
df["envelope_scaled"] = rescale_to_depth(df["envelope"])
df["final_fit_scaled"] = rescale_to_depth(df["final_fit"])
smooth_lookup = df.set_index("time")["smooth"]
 
# Rescale smooth values for evening and morning events    
results_evening = results_evening.copy()
results_evening["smooth"] = results_evening["onset"].map(smooth_lookup)
results_evening["scaled_smooth"] = rescale_to_depth(results_evening["smooth"])

# Rescale smooth values for morning events
results_morning = results_morning.copy()
results_morning["smooth"] = results_morning["onset"].map(smooth_lookup)
results_morning["scaled_smooth"] = rescale_to_depth(results_morning["smooth"])

# Rescale slope start and end values for plotting
df_slopes["y_start_scaled"] = rescale_to_depth(df_slopes["y_start"])
df_slopes["y_end_scaled"] = rescale_to_depth(df_slopes["y_end"])

fig = plt.figure(figsize=(mm_to_inch(185), mm_to_inch(105)))

# two main blocks:
#   top    = A + B + colorbar
#   bottom = C + D + E
outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.3)

# top block: A above B, with colorbar to the right of B
top = outer[0].subgridspec(2, 2, height_ratios=[0.55, 1.0], width_ratios=[1.0, 0.03], hspace=0, wspace=0.2)
bottom = outer[1].subgridspec(1, 3,width_ratios=[1.8, 1.0, 1.0], wspace=0.25)

ax_a = fig.add_subplot(top[0, 0])
ax_a_blank = fig.add_subplot(top[0, 1])
ax_a_blank.axis("off")

ax_b = fig.add_subplot(top[1, 0], sharex=ax_a)
ax_b_cm = fig.add_subplot(top[1, 1])

ax_c = fig.add_subplot(bottom[0, 0])
ax_d = fig.add_subplot(bottom[0, 1])
ax_e = fig.add_subplot(bottom[0, 2])

### Set labels
ax_a.text( -0.08, 1.2, "A", transform=ax_a.transAxes, ha="left", va="top", fontweight="bold", fontsize=10    )
ax_b.text( -0.08, 0.98, "B", transform=ax_b.transAxes, ha="left", va="top", fontweight="bold", fontsize=10    )
ax_c.text( -0.16, 0.98, "C", transform=ax_c.transAxes, ha="left", va="top", fontweight="bold", fontsize=10    )
ax_d.text( -0.12, 0.98, "D", transform=ax_d.transAxes, ha="left", va="top", fontweight="bold", fontsize=10    )
ax_e.text( -0.12, 0.98, "E", transform=ax_e.transAxes, ha="left", va="top", fontweight="bold", fontsize=10    )

### Set shared x axis for panel A and B
common_start = min(wavedata["time"].min(), heat2["time"].min())
common_end = max(wavedata["time"].max(), heat2["time"].max())

day_locator = mdates.DayLocator()
day_formatter = mdates.DateFormatter("%d/%m")

# Panel A: Significant weight height on the same time axis as the heatmap, with a secondary y-axis for wind speed.
ax_a.plot(wavedata["time"], wavedata["smooth"], color = "black",linewidth=0.5, label="Significant wave height")
ax_a.set_ylabel("S. wave \n height [m]")
ax_a.tick_params(axis="y", length=2)
ax_a.set_yticks(np.arange(0, 6, 1))
ax_a.set_ylim(0, 5)


# remote the x ticks completely
ax_a.tick_params(axis="x", bottom=False, labelbottom=False)
ax_a.set_xlim(common_start, common_end)
ax_a.grid(alpha=0.5)


# Panel B: heat map with integrated echo overlay
heat_matrix = heat2.pivot_table(index="depth", columns="time", values="value", aggfunc="mean").sort_index()
heat_times = mdates.date2num(heat_matrix.columns.to_numpy(dtype="datetime64[ns]"))
heat_depths = heat_matrix.index.to_numpy()

cmap = sns.color_palette("Blues", as_cmap=True)

# heatmap
im = ax_b.imshow(heat_matrix.to_numpy(), aspect="auto", origin="lower", extent=(heat_times.min(), heat_times.max(), heat_depths.min(), heat_depths.max()), cmap=cmap, interpolation="nearest")

# dashed horizontal reference lines
ax_b.axhline(-9.5, color="1", linestyle="--", linewidth=1)
ax_b.axhline(-20.5, color="1", linestyle="--", linewidth=1)

# black cutout rectangle
x0 = mdates.date2num(analysis_start)
x1 = mdates.date2num(analysis_end)
rect = Rectangle((x0, -20.5), x1 - x0, 11.0, fill=False, edgecolor="black", linewidth=1, zorder=20)
ax_b.add_patch(rect)

# axes styling
# ax_b.set_title("B", loc="left", fontweight="bold")
ax_b.set_ylabel("Depth [m]")
ax_b.set_xlabel("Date")
ax_b.set_ylim(-30, 0)
ax_b.set_xlim(common_start, common_end)
ax_b.xaxis.set_major_locator(day_locator)
ax_b.xaxis.set_major_formatter(day_formatter)
ax_b.tick_params(axis="x", rotation=45, length=2)
ax_b.tick_params(axis="y", length=2)
ax_b.margins(x=0, y=0)
ax_b.grid(False)

# secondary y-axis for integrated echo intensity
ax_b.plot(mdates.date2num(df["time"]), df["scaled_smooth_w"], color="#9c2727", linewidth=1)
secax = ax_b.secondary_yaxis(
    "right",
    functions=(depth_to_intensity, intensity_to_depth)
)
secax.set_ylabel("Integrated echo intensity")
secax.set_yticks(np.arange(550, 701, 50))
secax.tick_params(length=2)

# colorbar on the right
cbar = fig.colorbar(im, cax=ax_b_cm)
cbar.ax.set_title("Echo\nint. [dB]")
cbar.ax.tick_params(length=2)
cbar.outline.set_visible(False)

# Panel C: cutout plot with slope lines and transition markers.
sub_matrix = sub.pivot_table(index="depth", columns="time", values="value", aggfunc="mean").sort_index(ascending=False)
sub_x = mdates.date2num(sub_matrix.columns.to_numpy(dtype="datetime64[ns]"))
sub_y = sub_matrix.index.to_numpy()
sub_extent = (float(sub_x.min()), float(sub_x.max()), float(sub_y.min()), float(sub_y.max()))

# Show the heatmap 
ax_c.imshow(sub_matrix.to_numpy(), aspect="auto", origin="upper", extent=sub_extent, cmap="Blues")

# Overlay the smoothed line
ax_c.plot(mdates.date2num(df["time"]), df["scaled_smooth"], color="#9c2727", linewidth=0.7)
ax_c.plot(mdates.date2num(df["time"]), df["envelope_scaled"], color="darkgreen", linewidth=0.7)

# shade are between smoothed line and envelope
ax_c.fill_between(mdates.date2num(df["time"]), df["scaled_smooth"], df["envelope_scaled"], where=df["scaled_smooth"] > df["envelope_scaled"], color="darkgreen", alpha=0.3, label = 'AUC')

# Overlay the slope lines
for idx, slope_row in df_slopes.iterrows():
    if idx == 0:
        ax_c.plot(mdates.date2num([slope_row["x_start"], slope_row["x_end"]]), [slope_row["y_start_scaled"], slope_row["y_end_scaled"]], color="darkorange", linewidth=0.7, label = 'slope')
    else:
        ax_c.plot(mdates.date2num([slope_row["x_start"], slope_row["x_end"]]), [slope_row["y_start_scaled"], slope_row["y_end_scaled"]], color="darkorange", linewidth=0.7)

# Overlay the transition markers
ax_c.scatter(mdates.date2num(results_evening["onset"].to_numpy()), results_evening["scaled_smooth"], s=8, color="steelblue", label = 'Ascend')
ax_c.scatter(mdates.date2num(results_morning["onset"].to_numpy()), results_morning["scaled_smooth"], s=8, color="brown", label = 'Descend')

# Set the title and labels
# ax_c.set_title("C", loc="left", fontweight="bold")
ax_c.set_ylabel("Depth [m]")
ax_c.set_xlabel("Date")
legend = ax_c.legend(loc="lower center", ncol=4, fontsize=6, frameon=True, facecolor="grey", edgecolor="none", framealpha=1, labelspacing=0, columnspacing=0.2)

for text in legend.get_texts():
    text.set_color("white")

# Set the x and y limits, format the x-axis as dates, and rotate the x-axis labels for better readability
ax_c.xaxis.set_major_locator(mdates.DayLocator())
ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

# Set the x and y limits, format the x-axis as dates, and rotate the x-axis labels for better readability
ax_c.set_xlim(float(mdates.date2num(analysis_start)), float(mdates.date2num(analysis_end)))
ax_c.set_ylim(d_min, d_max) 
ax_c.xaxis_date()

# Panel D: regression plot for wave height vs ascent rate
# Fit the regression model and extract the necessary variables for plotting

caption_text = f"Adj. R² = {adj_r_squared(wave_slope_model.rvalue**2, len(mean_wave_slope_window), 1):.3f}\np = {wave_slope_model.pvalue:.3f}"
x_line = np.linspace(0, 3, 200)
y_line = wave_slope_model.intercept + wave_slope_model.slope * x_line

add_regression_ci(ax_d, mean_wave_slope_window, slope_values, x_line, wave_slope_model.slope, wave_slope_model.intercept)
ax_d.scatter(mean_wave_slope_window, slope_values, s=8, color="black", alpha=0.75)
ax_d.plot(x_line, y_line, color="steelblue", linewidth=1)
ax_d.text(0.98, 0.98, caption_text, transform=ax_d.transAxes, ha="right", va="top", fontsize=7)
ax_d.set_xlabel("Wave height [m]")
ax_d.set_ylabel("Ascent rate \n (slope) [a.u.]")
ax_d.grid(alpha=0.5)
ax_d.set_xlim(0, 3)
ax_d.set_yticks([0, 0.02, 0.04])
ax_d.set_yticklabels([])
ax_d.tick_params(axis="y", rotation = 45)

# Panel E: regression plot for significant wave height vs AUC
caption_text = f"Adj. R² = {adj_r_squared(wave_auc_model.rvalue**2, len(mean_wave_auc_window), 1):.3f}\np = {wave_auc_model.pvalue:.3f}"
y_line = wave_auc_model.intercept + wave_auc_model.slope * x_line

add_regression_ci(ax_e, mean_wave_auc_window, auc_values, x_line, wave_auc_model.slope, wave_auc_model.intercept)
ax_e.scatter(mean_wave_auc_window, auc_values, s=8, color="black", alpha=0.75)
ax_e.plot(x_line, y_line, color="steelblue", linewidth=1)
ax_e.text(0.98, 0.98, caption_text, transform=ax_e.transAxes, ha="right", va="top", fontsize=7)
ax_e.set_xlabel("Wave height [m]")
ax_e.set_ylabel("Migration magnitude\n (AUC) [a.u.]")
ax_e.grid(alpha=0.5)
ax_e.set_xlim(0, 3)
ax_e.set_yticks([500, 1000, 1500])
ax_e.set_yticklabels([])
ax_e.tick_params(axis="y", rotation = 45)
# ax_e.set_title("E", loc="left", fontweight="bold")

# final adjustments and save figure
# fig.subplots_adjust(hspace=0.15)

fig.savefig(base_dir / "ADCP_figure.svg", bbox_inches="tight")
fig.savefig(base_dir / "ADCP_figure.png", bbox_inches="tight", dpi=1200)


#%% Print summary statistics
print("\n=== Statistical summary ===")
print(f"Loaded ADCP data from: {mat_path.name}")
print(f"Weather table available: {have_weather_data}")
print(f"Wave table available: {have_wave_data}")

if not df_slopes.empty:
    print(f"Number of detected slope events: {len(df_slopes)}")
if windmodel is not None:
    print(f"Wind model adj. R^2: {windmodel.rsquared_adj:.3f}")
if truba_wavemodel is not None:
    print(f"Wave model adj. R^2: {truba_wavemodel.rsquared_adj:.3f}")

# %%
