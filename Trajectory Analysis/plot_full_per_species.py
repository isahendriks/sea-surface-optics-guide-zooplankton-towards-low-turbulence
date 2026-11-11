"""
Swimming direction and vertical velocity per light regime, per species, and of the
controls (no light) of all species.

Every figure has one column per light regime (or species, for the controls):
  top:    polar histogram of the swimming direction (speed weighted), with the
          speed-weighted circular variance V_w (mean ± SE over the tanks)
  bottom: "raincloud" of the vertical velocity dy: density curve (right), random sample
          of frames (left) and boxplot, with the letters of Tukey's HSD on the tank means
The overlay version of every figure adds the distribution and mean ± SD of every tank.

Before plotting and testing, dy is smoothed with a rolling mean of SMOOTH_WINDOW frames
within every track, and every tank is subsampled to the same number of frames (that of
the smallest tank), so every tank weighs equally.

Input:  <TRACKING_DATA_DIR>/<species>/trajectories_scaled/traj_<regime>_scaled.pkl (from load_and_save.py)
Output: figures in output/
"""
#%% Imports and settings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import gaussian_kde

# Folder with one subfolder per species (as written by load_and_save.py)
TRACKING_DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 0  # random seed of the plotted sample of frames, so the figures can be reproduced exactly

SPECIES_TANKS = {'Artemia_0805': [1, 2, 3], 'Cladocerans_2605': [1, 2, 3, 4, 5], 'BarnacleLarvae_2102': [1, 2, 3]}
SPECIES_LABELS = {'Artemia_0805': "Artemia larvae", 'Cladocerans_2605': "Water fleas", 'BarnacleLarvae_2102': "Barnacle larvae"}
CONDITIONS = ['still', 'breeze', 'stormy']

# Vertical axis range [mm/s] per species, raised automatically if a boxplot whisker does not fit
YLIM_VIOL = {'Artemia_0805': 6, 'Cladocerans_2605': 3.5, 'BarnacleLarvae_2102': 2}
YLIM_VIOL_CONTROL = 3.5
YLIM_MARGIN = 0.1   # if the manual ylim is raised, it is raised to the largest whisker + this fraction

WEIGHTED_POLAR = True   # weight the polar histograms by speed (direction of the distance travelled)
WEIGHTED_VERT = False   # weight the dy densities by speed
SMOOTH_DY = True        # smooth dy with a rolling mean before plotting and testing
SMOOTH_WINDOW = 20      # [frames]
KDE_BW = 0.05           # width of the Gaussian smoothing of the dy densities [mm/s] (None = Scott's rule)
ALPHA = 0.05            # significance level of Tukey's HSD

# Layout
XLIM_VIOL = 0.25                     # horizontal axis range of the raincloud panels
CLOUD_WIDTH = XLIM_VIOL * 0.7        # width of the sample of frames (left of 0)
CLOUD_OFFSET = 0.1 * XLIM_VIOL
BOX_WIDTH = CLOUD_WIDTH * 0.4
BOX_LINEWIDTH = 1
VIOL_CURVE_FRAC = 0.75               # highest density peak of a figure, as fraction of XLIM_VIOL
POINTS_SCATTER = 3000                # frames shown per panel
POLAR_BINS = np.linspace(0, 2 * np.pi, 50 + 1)
CONDITION_COLORS = sns.color_palette("Blues", 4)[1:]
CONTROL_COLOR = sns.color_palette("Blues", 4)[2]
TANK_COLORS = ["#0072B2", "#B20000", "#E69F00", "#009E73", "#CC79A7"]

mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'svg.fonttype': 'none',          # keep text editable in the SVGs
    'mathtext.fontset': 'custom',    # math text in Arial as well
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
})


def mm_to_inch(mm):
    return mm / 25.4

#%% Data preparation

def load_trajectories(measurement, regime):
    """Cleaned tracks of one species and regime, with the smoothed dy in the column 'smooth_dy'."""
    df = pd.read_pickle(TRACKING_DATA_DIR / measurement / 'trajectories_scaled' / f'traj_{regime}_scaled.pkl')
    df = df.sort_values(['tank', 'particle', 't']).reset_index(drop=True)
    df['smooth_dy'] = df.groupby(['tank', 'particle'])['dy'].transform(
        lambda x: x.rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean())
    return df


def subsample_tanks(df, tanks):
    """
    Randomly subsample every (condition, tank) to the size of the smallest one (random_state=42).
    Subsampled tanks are moved to the end of the table.
    """
    n_min = min(len(df[(df['Condition'] == c) & (df['tank'] == tank)]) for c in CONDITIONS for tank in tanks)
    for c in CONDITIONS:
        for tank in tanks:
            df_tank = df[(df['Condition'] == c) & (df['tank'] == tank)]
            if len(df_tank) > n_min:
                df = pd.concat([df.drop(df_tank.index), df_tank.sample(n=n_min, random_state=42)], ignore_index=True)
    return df


def subsample_control_tanks(df, species_tanks):
    """
    Randomly subsample every (species, tank) of the controls to the size of the smallest
    non-empty one (random_state=42). Returns the tanks concatenated in order.
    """
    parts = [df[(df['Species'] == species) & (df['tank'] == tank)] for species, tanks in species_tanks for tank in tanks]
    n_min = min(len(part) for part in parts if len(part) > 0)
    parts = [part.sample(n=n_min, random_state=42) if len(part) > n_min else part for part in parts]
    return pd.concat(parts, ignore_index=True)


def condition_data(measurement):
    """dy (smoothed), speed, orientation and tank of the three light regimes of one species, subsampled per tank."""
    dy_col = 'smooth_dy' if SMOOTH_DY else 'dy'
    dfs = {c: load_trajectories(measurement, c) for c in CONDITIONS}
    df = pd.DataFrame({
        'dy': np.concatenate([dfs[c][dy_col].values for c in CONDITIONS]),
        'speed': np.concatenate([dfs[c]['speed'].values for c in CONDITIONS]),
        'orientation': np.concatenate([dfs[c]['orientation'].values for c in CONDITIONS]),
        'tank': np.concatenate([dfs[c]['tank'].values for c in CONDITIONS]),
        'Condition': sum(([c] * len(dfs[c]) for c in CONDITIONS), []),
    })
    return subsample_tanks(df, SPECIES_TANKS[measurement])

#%% Statistics

def circular_variance(theta, weights=None):
    """1 - mean resultant length: 0 = all in the same direction, 1 = uniformly spread."""
    if weights is None:
        weights = np.ones_like(theta)
    R = np.abs(np.sum(weights * np.exp(1j * theta))) / np.sum(weights)
    return 1 - R


def tukey_letters(samples, names):
    """Compact letter display of Tukey's HSD: groups that share a letter do not differ significantly."""
    p_tukey = stats.tukey_hsd(*samples).pvalue
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"  {names[i]} vs {names[j]}: p = {p_tukey[i, j]:.4f}")

    order = list(np.argsort([-np.mean(s) for s in samples]))  # highest mean first
    letter_sets = []
    for i in order:
        group = []
        for j in order:
            if p_tukey[i, j] >= ALPHA and all(p_tukey[j, k] >= ALPHA for k in group):
                group.append(j)
        if not any(set(group) <= s for s in letter_sets):
            letter_sets.append(set(group))

    letters = {}
    for i, name in enumerate(names):
        letters[name] = ''.join('abcdefgh'[n] for n, s in enumerate(letter_sets) if i in s)
        print(f"  {name:>8}: {letters[name]}")
    return letters


def condition_statistics(df, tanks):
    """
    Per tank: mean dy and speed-weighted circular variance V_w. Prints mean ± SE over the
    tanks, a one-way ANOVA and Tukey's HSD over the light regimes.
    Returns the per-tank table and the Tukey letters per variable.
    """
    rows = []
    for condition in CONDITIONS:
        dy_tank, vw_tank = [], []
        for tank in tanks:
            d = df[(df['Condition'] == condition) & (df['tank'] == tank)].dropna(subset=['dy', 'orientation', 'speed'])
            dy_mean = np.mean(d['dy'])
            vw_mean = np.mean(circular_variance(np.mod(d['orientation'].values, 2*np.pi), d['speed'].values))
            dy_tank.append(dy_mean)
            vw_tank.append(vw_mean)
            rows.append({'Condition': condition, 'Tank': tank, 'dy_mean': dy_mean, 'vw_mean': vw_mean})

        dy_se = np.std(dy_tank, ddof=1) / np.sqrt(len(tanks))
        vw_se = np.std(vw_tank, ddof=1) / np.sqrt(len(tanks))
        print(f"  {condition}: mean dy = {np.mean(dy_tank):.3f} ± {dy_se:.3f} (SE), circular variance (speed weighted) = "
              f"{np.mean(vw_tank):.3f} ± {vw_se:.3f} (SE), n = {len(tanks)} tanks")

    df_results = pd.DataFrame(rows)
    letters = {}
    for variable in ['dy_mean', 'vw_mean']:
        samples = [df_results[df_results['Condition'] == c][variable].values for c in CONDITIONS]
        f, p = stats.f_oneway(*samples)
        print(f"\n{variable}: ANOVA F = {f:.2f}, p = {p:.4f}")
        letters[variable] = tukey_letters(samples, CONDITIONS)
    return df_results, letters

#%% Plot functions

def polar_hist(theta, speed):
    """Density histogram of the swimming direction on POLAR_BINS, speed weighted if WEIGHTED_POLAR."""
    counts, _ = np.histogram(theta, bins=POLAR_BINS, weights=speed if WEIGHTED_POLAR else None, density=True)
    return counts


def vert_density(d, y_grid):
    """Kernel density of dy on y_grid, with the area normalized to 1."""
    d = d.dropna(subset=['dy', 'speed'] if WEIGHTED_VERT else ['dy'])
    dy = d['dy'].values
    bw = None if KDE_BW is None else KDE_BW / np.std(dy)  # gaussian_kde expects a factor of the SD
    density = gaussian_kde(dy, weights=d['speed'].values if WEIGHTED_VERT else None, bw_method=bw)(y_grid)
    return density / np.trapezoid(density, y_grid)


def whisker_extent(values):
    """Largest absolute boxplot whisker (1.5 x IQR rule, as matplotlib's boxplot)."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return max(abs(values[values >= q1 - 1.5*iqr].min()), abs(values[values <= q3 + 1.5*iqr].max()))


def viol_ylim(lim_manual, panels):
    """Symmetric vertical range: lim_manual, raised (rounded up to 0.1 mm/s) if a whisker + YLIM_MARGIN does not fit."""
    lim_whisker = max(whisker_extent(v) for v in panels) * (1 + YLIM_MARGIN)
    if lim_whisker > lim_manual:
        lim = np.ceil(lim_whisker * 10) / 10
        print(f"ylim {lim_manual} mm/s is smaller than the whiskers, raised to {lim} mm/s")
        return lim
    return lim_manual


def viol_ticks(lim):
    """Ticks every 0.5 (lim < 2), 1 (lim <= 5) or 2 mm/s, symmetric around 0."""
    step = 0.5 if lim < 2 else 1 if lim <= 5 else 2
    n = int(np.floor(lim / step + 1e-9))
    return list(np.arange(-n, n + 1) * step)


def raincloud_figure(groups, ylim, width_mm, height_factor, colors=None, overlay=False,
                     titles=None, letters=None, rasterized=False,
                     overlay_alpha=(0.5, 0.7), tank_mean_marker=True):
    """
    One column per group: polar histogram of the swimming direction above a raincloud of dy.

    groups: list of (label, data, tanks), data with columns dy, speed, orientation, tank
    colors: fill colour per group (main version)
    overlay: True = grey fill with a line per tank and the tank mean ± SD, instead of a
             coloured fill with a black outline and a boxplot
    titles, letters: text above each polar histogram / at the top of each raincloud
    overlay_alpha: alpha of the grey polar bars and density fill in the overlay version
    tank_mean_marker: draw an extra point under the tank mean errorbars (overlay version)
    """
    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(width_mm) * height_factor))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)
    axes_pol = [fig.add_subplot(gs[0, i], projection='polar') for i in range(3)]
    axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]
    y_grid = np.linspace(-ylim, ylim, 600)
    width = POLAR_BINS[1] - POLAR_BINS[0]

    ### Polar histograms, one radial scale per figure (10% above the highest bar or line)
    peak_pol = 0
    for i, (ax, (label, data, tanks)) in enumerate(zip(axes_pol, groups)):
        d = data.dropna(subset=['orientation', 'speed'])
        counts = polar_hist(np.mod(d['orientation'].values, 2*np.pi), d['speed'].values)
        if overlay:
            ax.bar(POLAR_BINS[:-1], counts, width=width, align='edge', color='gray', alpha=overlay_alpha[0], edgecolor='none')
        else:
            ax.bar(POLAR_BINS[:-1], counts, width=width, align='edge', color=colors[i], alpha=1, edgecolor='none')
        peak_pol = max(peak_pol, counts.max())

        if overlay:
            for tank in tanks:
                d_tank = d[d['tank'] == tank]
                counts_tank = polar_hist(np.mod(d_tank['orientation'].values, 2*np.pi), d_tank['speed'].values)
                ax.plot(POLAR_BINS[:-1] + width/2, counts_tank, color=TANK_COLORS[tank-1], linewidth=1.2, alpha=0.8)
                peak_pol = max(peak_pol, counts_tank.max())

        ax.set_thetamin(0)
        if titles is not None:
            ax.set_title(titles[i], fontsize=6, pad=4)
        ax.set_xticklabels([])

    ylim_pol = 1.1 * peak_pol
    for ax in axes_pol:
        ax.set_ylim(0, ylim_pol)
        ax.set_yticks([ylim_pol/3, 2*ylim_pol/3])
        ax.set_yticklabels(['', ''])

    ### Density curves (area 1), scaled with one factor per figure so the highest peak fits
    densities = {label: vert_density(data, y_grid) for label, data, tanks in groups}
    densities_tank = {}
    if overlay:
        for label, data, tanks in groups:
            for tank in tanks:
                d_tank = data[data['tank'] == tank].dropna(subset=['dy'])
                if len(d_tank) > 1:
                    densities_tank[(label, tank)] = vert_density(d_tank, y_grid)
    peak_viol = max(dens.max() for dens in list(densities.values()) + list(densities_tank.values()))
    curve_width = VIOL_CURVE_FRAC * XLIM_VIOL / peak_viol

    ### Rainclouds
    for i, (ax, (label, data, tanks)) in enumerate(zip(axes_viol, groups)):
        if i > 0:
            ax.tick_params(axis='y', labelleft=False)
            ax.set_yticklabels([])

        dy = data['dy'].dropna().values
        density = densities[label] * curve_width
        if overlay:
            ax.fill_betweenx(y_grid, 0, density, color='gray', alpha=overlay_alpha[1], zorder=2)
            for tank in tanks:
                if (label, tank) in densities_tank:
                    ax.plot(densities_tank[(label, tank)] * curve_width, y_grid, color=TANK_COLORS[tank-1], linewidth=1, zorder=3, alpha=0.8)
        else:
            ax.plot(density, y_grid, color='black', linewidth=1, zorder=3)
            ax.fill_betweenx(y_grid, 0, density, color=colors[i], alpha=0.7, zorder=2)

        # Random sample of frames, jittered horizontally
        n_scatter = min(POINTS_SCATTER, len(dy))
        x_jitter = -np.random.uniform(CLOUD_OFFSET, CLOUD_WIDTH - CLOUD_OFFSET, size=n_scatter)
        ax.scatter(x_jitter, np.random.choice(dy, size=n_scatter, replace=False), s=1.5, color='grey', alpha=0.12,
                   linewidth=0, zorder=4, rasterized=rasterized)

        if overlay:
            # Mean ± SD of every tank
            for x, tank in zip(np.linspace(-CLOUD_WIDTH*0.85, -CLOUD_WIDTH*0.35, len(tanks)), tanks):
                dy_tank = data[data['tank'] == tank]['dy'].dropna().values
                tank_mean, tank_sd = np.mean(dy_tank), np.std(dy_tank, ddof=1)
                if tank_mean_marker:
                    ax.scatter(x, tank_mean, s=10, color=TANK_COLORS[tank-1], edgecolor='none', linewidth=0.4, zorder=6)
                ax.errorbar(x, tank_mean, yerr=tank_sd, fmt='o', color=TANK_COLORS[tank-1], markersize=3, capsize=2, linewidth=1, zorder=7)
        else:
            ax.boxplot(dy, orientation='vertical', positions=[-CLOUD_WIDTH/2], widths=BOX_WIDTH, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='white', edgecolor='black', linewidth=BOX_LINEWIDTH, alpha=0.8),
                       medianprops=dict(color='black', linewidth=BOX_LINEWIDTH),
                       whiskerprops=dict(color='black', linewidth=BOX_LINEWIDTH),
                       capprops=dict(color='black', linewidth=BOX_LINEWIDTH),
                       zorder=5)

        if letters is not None:
            ax.text(0.56, 0.97, letters[i], transform=ax.transAxes, ha='left', va='top', fontsize=8, zorder=6)

        ax.set_xlabel(label, fontsize=10)
        ax.tick_params(axis='y', labelsize=6, length=0, rotation=0)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(pad=2)
        ax.grid(True, alpha=1, axis='y', zorder=0)
        ax.axvline(x=0, color='gray', alpha=1, linewidth=1, zorder=0)
        ax.set_xticks([])
        ax.set_yticks(viol_ticks(ylim))
        ax.set_xlim(-XLIM_VIOL, XLIM_VIOL)
        ax.set_ylim([-ylim, ylim])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Frame around the raincloud panels
    pos_first, pos_last = axes_viol[0].get_position(), axes_viol[-1].get_position()
    fig.add_artist(Rectangle((pos_first.x0, pos_first.y0), pos_last.x1 - pos_first.x0, pos_first.y1 - pos_first.y0,
                             transform=fig.transFigure, fill=False, edgecolor='black', linewidth=1, zorder=100))
    return fig


def save(fig, folder, name):
    path = OUTPUT_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(path / f'{name}.{ext}', dpi=1200, transparent=True, bbox_inches='tight')

#%% Figures per species

def plot_species(measurement):
    print(f"\n##### {measurement} #####")
    tanks = SPECIES_TANKS[measurement]
    df = condition_data(measurement)

    ylim = viol_ylim(YLIM_VIOL[measurement], [df[df['Condition'] == c]['dy'].dropna().values for c in CONDITIONS])
    print(f"Vertical axis range: ±{ylim} mm/s")

    df_results, letters = condition_statistics(df, tanks)

    # V_w above every polar histogram: mean ± SE over the tanks
    titles = []
    for c in CONDITIONS:
        vw = df_results[df_results['Condition'] == c]['vw_mean'].values
        titles.append(rf'$V_w = {np.mean(vw):.2f} \pm {np.std(vw, ddof=1) / np.sqrt(len(vw)):.2f}$')

    groups = [(c, df[df['Condition'] == c], tanks) for c in CONDITIONS]
    width_mm = 210/3*0.9

    fig = raincloud_figure(groups, ylim, width_mm, 1.3, colors=CONDITION_COLORS, titles=titles,
                           letters=[letters['dy_mean'][c].upper() for c in CONDITIONS])
    save(fig, 'verthist_polarhist_perframe_combined', f'{measurement}_full_per_species')

    fig = raincloud_figure(groups, ylim, width_mm, 1.3, overlay=True, titles=titles)
    save(fig, 'verthist_polarhist_perframe_overlay', f'{measurement}_full_per_species')

    # Legend of the tank colours
    fig_legend = plt.figure(figsize=(2, 0.1))
    legend_ax = fig_legend.add_subplot(111)
    legend_ax.legend(handles=[Line2D([0], [0], color=TANK_COLORS[i], lw=2, label=f'Tank {tank}') for i, tank in enumerate(tanks)],
                     ncol=len(tanks), fontsize=8)
    legend_ax.axis('off')
    save(fig_legend, 'verthist_polarhist_perframe_combined', f'{measurement}_tank_legend')

#%% Figures of the controls (no light) of all species

def plot_controls():
    measurements = list(SPECIES_TANKS)
    dfs = []
    for m in measurements:
        df_m = load_trajectories(m, 'nothing')
        df_m['Species'] = SPECIES_LABELS[m]
        df_m['Measurement'] = m
        dfs.append(df_m)
    df_control_all = pd.concat(dfs, ignore_index=True)

    species_order = [SPECIES_LABELS[m] for m in measurements]
    df_control_plot = subsample_control_tanks(df_control_all, [(SPECIES_LABELS[m], SPECIES_TANKS[m]) for m in measurements])
    if SMOOTH_DY:
        df_control_plot['dy'] = df_control_plot['smooth_dy']  # df_control_all keeps the raw dy for the summaries

    print("\nControl, circular variance of the swimming direction")
    for species in species_order:
        d = df_control_all[df_control_all['Species'] == species].dropna(subset=['orientation', 'speed'])
        theta = np.mod(d['orientation'].values, 2*np.pi)
        V_tanks = [circular_variance(np.mod(d_t['orientation'].values, 2*np.pi), d_t['speed'].values) for _, d_t in d.groupby('tank')]
        print(f"{species:>16}: {circular_variance(theta, d['speed'].values):.3f} (speed weighted), "
              f"{circular_variance(theta):.3f} (unweighted); "
              f"per tank (speed weighted): {np.mean(V_tanks):.2f} ± {np.std(V_tanks, ddof=1):.2f} (mean ± SD, n = {len(V_tanks)})")

    ylim = viol_ylim(YLIM_VIOL_CONTROL, [df_control_plot[df_control_plot['Species'] == s]['dy'].dropna().values for s in species_order])
    print(f"Control vertical axis range: ±{ylim} mm/s")

    groups = []
    for species in species_order:
        data = df_control_plot[df_control_plot['Species'] == species]
        groups.append((species, data, sorted(data['tank'].dropna().astype(int).unique())))
    width_mm = 210/2*0.9

    fig = raincloud_figure(groups, ylim, width_mm, 1.2, colors=[CONTROL_COLOR] * 3, rasterized=True)
    save(fig, 'control_species_combined', 'control_all_species')

    fig = raincloud_figure(groups, ylim, width_mm, 1.2, overlay=True, rasterized=True,
                           overlay_alpha=(0.4, 0.4), tank_mean_marker=False)
    save(fig, 'control_species_overlay', 'control_all_species_overlay')

    print("Control means ± SD:")
    for species in species_order:
        dy = df_control_plot[df_control_plot['Species'] == species]['dy'].dropna().values
        print(f"{species}: mean = {np.mean(dy):.2f}, std = {np.std(dy):.2f}, n = {len(dy):,} frames")
        # Per replicate: mean dy per tank (all data), then mean ± SD over the tanks
        tank_means = df_control_all[df_control_all['Species'] == species].groupby('tank')['dy'].mean()
        print(f"{'':>{len(species)}}  per tank: mean = {tank_means.mean():.2f}, SD = {tank_means.std(ddof=1):.2f}, n = {len(tank_means)} tanks")

#%% Make all figures
if __name__ == "__main__":
    np.random.seed(SEED)
    for measurement in SPECIES_TANKS:
        plot_species(measurement)
    plot_controls()
