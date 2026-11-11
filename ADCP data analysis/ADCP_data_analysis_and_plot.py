"""
Diel vertical migration of zooplankton from ADCP backscatter, related to waves and wind.

Steps
  1. Load the ADCP echo intensity (beam 5), the weather (wind) and the wave data.
  2. Sum the echo intensity over 9-21 m depth ("integrated echo") and smooth it.
  3. Detect the daily ascent (rise) and descent (fall) of the integrated echo.
  4. Ascent rate: slope of the smoothed echo in a 30 min window around the steepest
     point of every ascent.
  5. Migration magnitude: area under the curve (AUC) of the smoothed echo above its
     lower envelope, per day (midday to midday).
  6. Linear regression of ascent rate and migration magnitude against the mean wave
     height and wind speed, and the publication figure (ADCP_figure.svg/.png).

Input (in data/)
  Trubadur_ADCP_beam5_m30sec.mat   ADCP echo intensity, 30 s bins, time in CEST
  r_weather.csv                    weather station data, time in UTC
  w_truba.csv                      significant wave height, time in CEST
Output (in output/)
  ADCP_figure.svg, ADCP_figure.png
"""
#%% Imports and settings
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import linregress

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DEPTH_RANGE = (-21, -9)     # depth range summed into the integrated echo [m] (exclusive)
SMOOTH_WINDOW = 241         # rolling-median window [samples] (30 s samples, ~2 h)
EDGE_SAMPLES = 121          # samples at the start/end of the record without a smoothed value
SLOPE_HALF_WINDOW = pd.Timedelta(minutes=15)  # half width of the ascent-rate window

# Candidate events removed by hand after visual inspection (onset time, CEST)
EXCLUDED_RISE_ONSETS = ["2018-09-12 20:29:13", "2018-09-25 05:00:05"]
EXCLUDED_FALL_ONSETS = ["2018-09-21 06:26:43", "2018-09-21 22:40:35", "2018-09-22 17:44:35",
                        "2018-09-22 21:37:05", "2018-09-22 21:48:35", "2018-09-22 21:54:05",
                        "2018-09-23 16:14:05"]

# Two-day window shown in panel c
ANALYSIS_START = pd.Timestamp("2018-08-29 12:00:00")
ANALYSIS_END = pd.Timestamp("2018-08-31 12:00:00")

# Figure style
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 8,
    "xtick.major.pad": 0.8,
    "ytick.major.pad": 0.8,
    "axes.labelpad": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})


def mm_to_inch(mm):
    return mm / 25.4

#%% Functions

def find_transitions(day_df, threshold_factor=0.5, min_run_len=5):
    """
    Find the rises and falls of the smoothed echo within one day.

    A rise (fall) is a run of at least min_run_len consecutive samples in which the first
    derivative d1 is above (below) +/- threshold_factor * SD(d1) of that day.

    Returns two DataFrames (rises, falls) with one row per run:
      onset      time of the first sample of the run
      peak_time  time of the steepest rise/fall (maximum |d1|) within the run
      change     increase (rise) or decrease (fall) of the smoothed echo over the run
    """
    d1 = day_df["d1"].to_numpy()
    vals = day_df["smooth"].to_numpy()
    times = day_df["time"].to_numpy()

    thr = np.nanstd(d1) * threshold_factor
    rising = np.where(np.isnan(d1), False, d1 > thr)
    falling = np.where(np.isnan(d1), False, d1 < -thr)

    def extract_runs(mask, direction):
        rows = []
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
                change = vals[end] - vals[start]
                run_d1 = d1[start:end + 1]
                peak = start + int(np.argmax(run_d1) if direction == "rise" else np.argmin(run_d1))
                if direction == "fall":
                    change = -change
                rows.append({"onset": times[start], "peak_time": times[peak], "change": change})
        return pd.DataFrame(rows)

    return extract_runs(rising, "rise"), extract_runs(falling, "fall")


def detect_events(df, event_type, excluded_onsets):
    """
    Candidate rises or falls of every day, without the excluded ones, keeping the one with
    the largest change per day.
    """
    candidates = []
    for day, day_df in df.groupby("day", sort=True):
        rises, falls = find_transitions(day_df)
        runs = rises if event_type == "rise" else falls
        if not runs.empty:
            runs.insert(0, "day", day)
            candidates.append(runs)
    candidates = pd.concat(candidates, ignore_index=True)

    excluded = candidates["onset"].dt.floor("s").isin(pd.to_datetime(excluded_onsets))
    candidates = candidates[~excluded].reset_index(drop=True)

    events = candidates.loc[candidates.groupby("day")["change"].idxmax()].reset_index(drop=True)
    events["type"] = event_type
    return events


def event_slope(df, event):
    """
    Ascent/descent rate of one event: slope of a linear regression of the smoothed echo
    against time [s], over SLOPE_HALF_WINDOW on both sides of the steepest point.
    Returns None if there is no data at that time.
    """
    centre = event["peak_time"]
    local = df.loc[df["time"].between(centre - SLOPE_HALF_WINDOW, centre + SLOPE_HALF_WINDOW)]
    if local.empty:
        return None

    x_seconds = (local["time"] - centre).dt.total_seconds().to_numpy()
    slope = linregress(x_seconds, local["smooth"].to_numpy())[0]

    smooth_at_centre = df.loc[df["time"] == centre, "smooth"]
    if smooth_at_centre.empty:
        return None

    # End points of the fitted line through the smoothed echo at the steepest point, for plotting
    smooth_value = float(smooth_at_centre.iloc[0])
    half = SLOPE_HALF_WINDOW.total_seconds()
    x_start, x_end = centre - SLOPE_HALF_WINDOW, centre + SLOPE_HALF_WINDOW
    y_start, y_end = smooth_value - slope * half, smooth_value + slope * half

    return {
        "onset": event["onset"],
        "peak_time": centre,
        "type": event["type"],
        "x_start": x_start,
        "y_start": y_start,
        "x_end": x_end,
        "y_end": y_end,
        "slope": (y_end - y_start) / (x_end - x_start).total_seconds(),  # equals the regression slope
    }


def lower_envelope(time_hours, values_smooth, n_days):
    """
    Lower envelope of the smoothed echo: a natural cubic spline through the daily minima,
    smoothed with a rolling median and clipped at 0.

    Daily minima are local minima at least 18 h apart. A day without a detected minimum
    (no minimum within 10 h of t0 + 24 h * n) gets the smoothed value at that time instead.
    """
    min_peak_distance = 18 / (time_hours[1] - time_hours[0])  # 18 h in samples
    min_peak_idx, _ = find_peaks(-values_smooth, distance=min_peak_distance, prominence=np.nanstd(values_smooth) * 0.1)
    min_peak_times = time_hours[min_peak_idx]
    min_peak_vals = values_smooth[min_peak_idx]

    expected_times = min_peak_times[0] + 24 * np.arange(n_days)
    for te in expected_times:
        nearest = min_peak_times[np.argmin(np.abs(min_peak_times - te))]
        if abs(nearest - te) > 10:
            index = np.argmin(np.abs(time_hours - te))
            min_peak_times = np.append(min_peak_times, te)
            min_peak_vals = np.append(min_peak_vals, values_smooth[index])

    order = np.argsort(min_peak_times)
    spline = CubicSpline(min_peak_times[order], min_peak_vals[order], bc_type="natural", extrapolate=True)
    envelope = pd.Series(spline(time_hours)).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
    return np.clip(envelope, 0, None)


def daily_windows(df):
    """
    (start, end) sample indices of the daily migration windows, from the sample closest to
    noon on one day to the sample closest to noon on the next day. The first and last
    window are cut at the edge of the smoothed record.
    """
    midday_idx = []
    for day, day_df in df.groupby("day", sort=True):
        midday = pd.Timestamp(day) + pd.Timedelta(hours=12)
        midday_idx.append((day_df["time"] - midday).abs().idxmin())

    windows = []
    for i in range(len(midday_idx) - 1):
        start, end = midday_idx[i], midday_idx[i + 1]
        if i == 0:
            start = EDGE_SAMPLES
        elif i == len(midday_idx) - 2:
            end = midday_idx[i + 1] - EDGE_SAMPLES
        windows.append((start, end))
    return windows


def adj_r_squared(r_squared, n, k):
    """Adjusted R^2 for n observations and k predictors."""
    return 1 - (1 - r_squared) * (n - 1) / (n - k - 1)


def add_regression_ci(ax, x, y, x_line, slope, intercept, color="0.8", alpha=0.5):
    """Shade the 95% confidence interval of the fitted regression line."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_hat = intercept + slope * x_line

    n = len(x)
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean) ** 2)
    resid = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))  # residual standard error

    t_val = stats.t.ppf(0.975, df=n - 2)
    ci = t_val * s_err * np.sqrt((1 / n) + ((x_line - x_mean) ** 2 / ssx))
    ax.fill_between(x_line, y_hat - ci, y_hat + ci, color=color, alpha=alpha, linewidth=0, zorder=1)

#%% Load data
weatherdata = pd.read_csv(DATA_DIR / "r_weather.csv", sep=";", decimal=",")
weatherdata["date"] = pd.to_datetime(weatherdata["date"], format="%Y.%m.%d")
weatherdata["time_utc"] = pd.to_timedelta(weatherdata["time_utc"].astype(str))
weatherdata["datetime"] = weatherdata["date"] + weatherdata["time_utc"] + pd.Timedelta(hours=2)  # UTC -> CEST

wavedata = pd.read_csv(DATA_DIR / "w_truba.csv")
wavedata["time"] = pd.to_datetime(wavedata["time"])  # CEST

adcp = loadmat(DATA_DIR / "Trubadur_ADCP_beam5_m30sec.mat", squeeze_me=True, struct_as_record=False)
adcp_time = pd.to_datetime((np.asarray(adcp["tm"]) - 719529) * 86400, unit="s")  # MATLAB datenum -> datetime, CEST
depth = np.asarray(adcp["z"]).astype(float)

# Echo intensity in long format: one row per (time, depth)
df_wide = pd.DataFrame({"time": adcp_time})
echo = pd.DataFrame(np.asarray(adcp["a5m"]))
for idx, depth_value in enumerate(depth):
    df_wide[str(depth_value)] = echo.iloc[:, idx].to_numpy()
heat2 = df_wide.melt(id_vars="time", var_name="depth", value_name="value")
heat2["depth"] = heat2["depth"].astype(float)

#%% Integrated echo and migration events
in_layer = (heat2["depth"] < DEPTH_RANGE[1]) & (heat2["depth"] > DEPTH_RANGE[0])
df = heat2.loc[in_layer].groupby("time", as_index=False)["value"].sum()
df = df.sort_values("time").reset_index(drop=True)

df["smooth"] = df["value"].rolling(window=SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).median()
df["d1"] = df["smooth"].diff()  # first derivative of the smoothed echo
df["day"] = df["time"].dt.date

results_evening = detect_events(df, "rise", EXCLUDED_RISE_ONSETS)  # ascent, around dusk
results_morning = detect_events(df, "fall", EXCLUDED_FALL_ONSETS)  # descent, around dawn
events = pd.concat([results_evening, results_morning], ignore_index=True, sort=False)
print(f"Total number of events detected: {len(events)}")

#%% Ascent rate around every event
df_slopes = pd.DataFrame([s for _, event in events.iterrows() if (s := event_slope(df, event)) is not None])

#%% Migration magnitude: area under the curve above the lower envelope, per day
values_smooth = df["smooth"].to_numpy()
time_hours = (df["time"] - df["time"].min()).dt.total_seconds().to_numpy() / 3600.0

df["envelope"] = lower_envelope(time_hours, values_smooth, n_days=27)
windows = daily_windows(df)

auc_values = []
for start, end in windows:
    above_envelope = np.clip(values_smooth[start:end + 1] - df["envelope"].to_numpy()[start:end + 1], 0, None)
    auc_values.append(np.trapezoid(above_envelope, time_hours[start:end + 1]))
auc_values = np.array(auc_values)

#%% Mean wind speed and wave height per event and per day
rises = df_slopes[df_slopes["type"] == "rise"]
slope_values = rises["slope"].to_numpy()

mean_wind_slope = []
mean_wave_slope = []
for _, rise in rises.iterrows():
    # Wind: mean over +/- 1 h around the weather observation closest to the onset
    closest = weatherdata.loc[(weatherdata["datetime"] - rise["onset"]).abs().idxmin(), "datetime"]
    in_window = weatherdata["datetime"].between(closest - pd.Timedelta(hours=1), closest + pd.Timedelta(hours=1))
    mean_wind_slope.append(weatherdata.loc[in_window, "wind_speed_m_s"].mean())

    # Waves: mean over the ascent-rate window (between the wave observations closest to its ends)
    wave_start = wavedata.loc[(wavedata["time"] - rise["x_start"]).abs().idxmin(), "time"]
    wave_end = wavedata.loc[(wavedata["time"] - rise["x_end"]).abs().idxmin(), "time"]
    mean_wave_slope.append(wavedata.loc[wavedata["time"].between(wave_start, wave_end), "smooth"].mean())
mean_wind_slope = np.array(mean_wind_slope)
mean_wave_slope = np.array(mean_wave_slope)

mean_wind_auc = []
mean_wave_auc = []
for start, end in windows:
    t_start, t_end = df["time"].iloc[start], df["time"].iloc[end]
    mean_wind_auc.append(weatherdata.loc[weatherdata["datetime"].between(t_start, t_end), "wind_speed_m_s"].mean())
    mean_wave_auc.append(wavedata.loc[wavedata["time"].between(t_start, t_end), "smooth"].mean())
mean_wind_auc = np.array(mean_wind_auc)
mean_wave_auc = np.array(mean_wave_auc)

#%% Linear regressions
wind_slope_model = linregress(mean_wind_slope, slope_values)
wave_slope_model = linregress(mean_wave_slope, slope_values)
wind_auc_model = linregress(mean_wind_auc, auc_values)
wave_auc_model = linregress(mean_wave_auc, auc_values)

for name, model, n in [("Wind speed vs AUC", wind_auc_model, len(mean_wind_auc)),
                       ("Wave height vs AUC", wave_auc_model, len(mean_wave_auc)),
                       ("Wind speed vs Slope", wind_slope_model, len(mean_wind_slope)),
                       ("Wave height vs Slope", wave_slope_model, len(mean_wave_slope))]:
    print(f"\n=== {name} regression ===")
    print(f"P-value: {model.pvalue:.4f}")
    print(f"Adjusted R-squared: {adj_r_squared(model.rvalue**2, n, 1):.4f}")

# Fitted values at still (Hs = 0 m) and rough (Hs = 3 m) conditions, for the results text
for name, model in [("Ascent rate", wave_slope_model), ("Migration magnitude", wave_auc_model)]:
    fit_0 = model.intercept
    fit_3 = model.intercept + model.slope * 3
    print(f"{name}: Hs=0 m: {fit_0:.4g}, Hs=3 m: {fit_3:.4g}, ratio (0/3): {fit_0 / fit_3:.2f}, p = {model.pvalue:.4f}")

#%% Publication figure
# The smoothed echo is drawn on the depth axis of the heat maps: rescale it linearly onto
# the depth range of the panel-c cutout.
sub = heat2.loc[heat2["time"].between(ANALYSIS_START, ANALYSIS_END) & heat2["depth"].between(-20.5, -9.5)]
d_min, d_max = float(sub["depth"].min()), float(sub["depth"].max())
s_min, s_max = float(df["smooth"].min()), float(df["smooth"].max())


def rescale_to_depth(x):
    return (x - s_min) / (s_max - s_min) * (d_max - d_min) + d_min


# Secondary y-axis of panel b (integrated echo intensity <-> depth)
depth_min, depth_max = -20.5, -9.5


def intensity_to_depth(y):
    return (y - s_min) / (s_max - s_min) * (depth_max - depth_min) + depth_min


def depth_to_intensity(y):
    return (y - depth_min) / (depth_max - depth_min) * (s_max - s_min) + s_min


df["scaled_smooth"] = rescale_to_depth(df["smooth"])
df["envelope_scaled"] = rescale_to_depth(df["envelope"])
smooth_lookup = df.set_index("time")["smooth"]
for events_type in (results_evening, results_morning):
    events_type["scaled_smooth"] = rescale_to_depth(events_type["peak_time"].map(smooth_lookup))
df_slopes["y_start_scaled"] = rescale_to_depth(df_slopes["y_start"])
df_slopes["y_end_scaled"] = rescale_to_depth(df_slopes["y_end"])

fig = plt.figure(figsize=(mm_to_inch(185), mm_to_inch(105)))

# Layout: top block = a above b (with colour bar), bottom block = c, d, e
outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.3)
top = outer[0].subgridspec(2, 2, height_ratios=[0.55, 1.0], width_ratios=[1.0, 0.03], hspace=0, wspace=0.2)
bottom = outer[1].subgridspec(1, 3, width_ratios=[1.8, 1.0, 1.0], wspace=0.25)

ax_a = fig.add_subplot(top[0, 0])
fig.add_subplot(top[0, 1]).axis("off")
ax_b = fig.add_subplot(top[1, 0], sharex=ax_a)
ax_b_cm = fig.add_subplot(top[1, 1])
ax_c = fig.add_subplot(bottom[0, 0])
ax_d = fig.add_subplot(bottom[0, 1])
ax_e = fig.add_subplot(bottom[0, 2])

for ax, label, x, y in [(ax_a, "a", -0.08, 1.2), (ax_b, "b", -0.08, 0.98), (ax_c, "c", -0.16, 0.98),
                        (ax_d, "d", -0.12, 0.98), (ax_e, "e", -0.12, 0.98)]:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=10)

common_start = min(wavedata["time"].min(), heat2["time"].min())
common_end = max(wavedata["time"].max(), heat2["time"].max())

### Panel a: significant wave height
ax_a.plot(wavedata["time"], wavedata["smooth"], color="black", linewidth=0.5, label="Significant wave height")
ax_a.set_ylabel("S. wave \n height [m]")
ax_a.tick_params(axis="y", length=2)
ax_a.set_yticks(np.arange(0, 6, 1))
ax_a.set_ylim(0, 5)
ax_a.tick_params(axis="x", bottom=False, labelbottom=False)
ax_a.set_xlim(common_start, common_end)
ax_a.grid(alpha=0.5)

### Panel b: echo intensity heat map with the integrated echo
heat_matrix = heat2.pivot_table(index="depth", columns="time", values="value", aggfunc="mean").sort_index()
heat_times = mdates.date2num(heat_matrix.columns.to_numpy(dtype="datetime64[ns]"))
heat_depths = heat_matrix.index.to_numpy()

im = ax_b.imshow(heat_matrix.to_numpy(), aspect="auto", origin="lower",
                 extent=(heat_times.min(), heat_times.max(), heat_depths.min(), heat_depths.max()),
                 cmap=sns.color_palette("Blues", as_cmap=True), interpolation="nearest")

# Integrated depth layer (dashed) and the cutout shown in panel c (box)
ax_b.axhline(-9.5, color="1", linestyle="--", linewidth=1)
ax_b.axhline(-20.5, color="1", linestyle="--", linewidth=1)
x0, x1 = mdates.date2num(ANALYSIS_START), mdates.date2num(ANALYSIS_END)
ax_b.add_patch(Rectangle((x0, -20.5), x1 - x0, 11.0, fill=False, edgecolor="black", linewidth=1, zorder=20))

ax_b.set_ylabel("Depth [m]")
ax_b.set_xlabel("Date")
ax_b.set_ylim(-30, 0)
ax_b.set_xlim(common_start, common_end)
ax_b.xaxis.set_major_locator(mdates.DayLocator())
ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax_b.tick_params(axis="x", rotation=45, length=2)
ax_b.tick_params(axis="y", length=2)
ax_b.margins(x=0, y=0)
ax_b.grid(False)

ax_b.plot(mdates.date2num(df["time"]), df["scaled_smooth"], color="#9c2727", linewidth=1)
secax = ax_b.secondary_yaxis("right", functions=(depth_to_intensity, intensity_to_depth))
secax.set_ylabel("Integrated echo intensity")
secax.set_yticks(np.arange(550, 701, 50))
secax.tick_params(length=2)

cbar = fig.colorbar(im, cax=ax_b_cm)
cbar.ax.set_title("Echo\nint. [dB]")
cbar.ax.tick_params(length=2)
cbar.outline.set_visible(False)

### Panel c: two-day cutout with envelope, AUC, ascent-rate lines and events
sub_matrix = sub.pivot_table(index="depth", columns="time", values="value", aggfunc="mean").sort_index(ascending=False)
sub_x = mdates.date2num(sub_matrix.columns.to_numpy(dtype="datetime64[ns]"))
sub_y = sub_matrix.index.to_numpy()
ax_c.imshow(sub_matrix.to_numpy(), aspect="auto", origin="upper",
            extent=(float(sub_x.min()), float(sub_x.max()), float(sub_y.min()), float(sub_y.max())), cmap="Blues")

time_num = mdates.date2num(df["time"])
ax_c.plot(time_num, df["scaled_smooth"], color="#9c2727", linewidth=0.7)
ax_c.plot(time_num, df["envelope_scaled"], color="darkgreen", linewidth=0.7)
ax_c.fill_between(time_num, df["scaled_smooth"], df["envelope_scaled"], where=df["scaled_smooth"] > df["envelope_scaled"],
                  color="darkgreen", alpha=0.3, label='AUC')

for i, slope_row in df_slopes.iterrows():
    ax_c.plot(mdates.date2num([slope_row["x_start"], slope_row["x_end"]]), [slope_row["y_start_scaled"], slope_row["y_end_scaled"]],
              color="darkorange", linewidth=0.7, label='slope' if i == 0 else None)

ax_c.scatter(mdates.date2num(results_evening["peak_time"].to_numpy()), results_evening["scaled_smooth"], s=8, color="steelblue", label='Ascend')
ax_c.scatter(mdates.date2num(results_morning["peak_time"].to_numpy()), results_morning["scaled_smooth"], s=8, color="brown", label='Descend')

ax_c.set_ylabel("Depth [m]")
ax_c.set_xlabel("Date")
legend = ax_c.legend(loc="lower center", ncol=4, fontsize=6, frameon=True, facecolor="grey", edgecolor="none",
                     framealpha=1, labelspacing=0, columnspacing=0.2)
for text in legend.get_texts():
    text.set_color("white")

ax_c.xaxis.set_major_locator(mdates.DayLocator())
ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax_c.set_xlim(float(x0), float(x1))
ax_c.set_ylim(d_min, d_max)
ax_c.xaxis_date()

### Panels d and e: regressions on wave height (y axes in arbitrary units: tick labels hidden)
x_line = np.linspace(0, 3, 200)
for ax, x, y, model, ylabel, yticks in [
        (ax_d, mean_wave_slope, slope_values, wave_slope_model, "Ascent rate \n (slope) [dB s$^{-1}$]", [0, 0.02, 0.04]),
        (ax_e, mean_wave_auc, auc_values, wave_auc_model, "Migration magnitude\n (AUC) [dB·h]", [0, 500, 1000, 1500])]:
    caption = f"Adj. R² = {adj_r_squared(model.rvalue**2, len(x), 1):.3f}\np = {model.pvalue:.3f}"
    add_regression_ci(ax, x, y, x_line, model.slope, model.intercept)
    ax.scatter(x, y, s=8, color="black", alpha=0.75)
    ax.plot(x_line, model.intercept + model.slope * x_line, color="steelblue", linewidth=1)
    ax.text(0.98, 0.98, caption, transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.set_xlabel("Wave height [m]")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(bottom=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels([""] * len(yticks))
    ax.tick_params(axis="y", rotation=0)

OUTPUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUTPUT_DIR / "ADCP_figure.svg", bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "ADCP_figure.png", bbox_inches="tight", dpi=1200)
