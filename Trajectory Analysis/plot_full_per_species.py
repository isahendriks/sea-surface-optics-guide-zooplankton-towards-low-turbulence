#%% Import packages
import os
import numpy as np
import pandas as pd
import matplotlib as mpl

# Set font for all plots
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10          # Optional default font size
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from post_processing_functions import *
import pickle
from scipy.stats import gaussian_kde



# Create a single A4-sized Figure and a 1x9 grid with small spacing
mm_to_inch = lambda mm: mm / 25.4
A4_W_MM = 210.0
fig_w = mm_to_inch(18)
fig_h = mm_to_inch(18*1.5)

# Define Gaussian Kernel for smoothing
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2.0
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()

# Load path
path_to_tracking_data = "R:\\LU24A1047-PLS\\TrackingData" #Research data folder

#%% Prepare data for plot
per_traj = False # Set to True to use per-trajectory data, False for per-frame data

# Define measurement 
# measurement = 'Cladocerans_2605'
measurement = 'Artemia_0805'
# measurement = 'BarnacleLarvae_2102'  

### Load data
path_to_trajectories = os.path.join(path_to_tracking_data, measurement, 'trajectories_scaled')

df_still = pickle.load(open(os.path.join(path_to_trajectories, 'traj_still_scaled.pkl'), 'rb'))
df_breeze = pickle.load(open(os.path.join(path_to_trajectories, 'traj_breeze_scaled.pkl'), 'rb'))
df_stormy = pickle.load(open(os.path.join(path_to_trajectories, 'traj_stormy_scaled.pkl'), 'rb'))
df_nothing = pickle.load(open(os.path.join(path_to_trajectories, 'traj_nothing_scaled.pkl'), 'rb'))

path_to_figure_savelocation = os.path.join(path_to_tracking_data, "manus_figures", 'verthist_polarhist_perframe_combined')
path_to_figure_savelocation_overlay = os.path.join(path_to_tracking_data, "manus_figures", 'verthist_polarhist_perframe_overlay')
save_control_main = os.path.join(path_to_tracking_data, "manus_figures", "control_species_combined")
save_control_overlay = os.path.join(path_to_tracking_data, "manus_figures", "control_species_overlay")

os.makedirs(save_control_main, exist_ok=True)
os.makedirs(save_control_overlay, exist_ok=True)
os.makedirs(path_to_figure_savelocation, exist_ok=True)
os.makedirs(path_to_figure_savelocation_overlay, exist_ok=True)
### Calculate mean and standard error for each treatment
# Define treatments
treatments = ['still', 'breeze', 'stormy', 'control']
conditions = ['still', 'breeze', 'stormy']

if measurement == 'Cladocerans_2605':
    tanks = [1, 2, 3, 4, 5]
else:
    tanks = [1, 2, 3]

# Extract data for each treatment
still_means = [np.mean(df_still['dy'][df_still['tank'] == tank]) for tank in tanks]
breeze_means = [np.mean(df_breeze['dy'][df_breeze['tank'] == tank]) for tank in tanks]
stormy_means = [np.mean(df_stormy['dy'][df_stormy['tank'] == tank]) for tank in tanks]
control_means = [np.mean(df_nothing['dy'][df_nothing['tank'] == tank]) for tank in tanks]

### Calculate mean and standard error for each treatment
means = [np.mean(still_means), np.mean(breeze_means), np.mean(stormy_means), np.mean(control_means)]
STDs = [np.std(still_means), np.std(breeze_means), np.std(stormy_means), np.std(control_means)]
 
# create dataframe for plotting
data = {'dy': np.concatenate([df_still['dy'].values, df_breeze['dy'].values, df_stormy['dy'].values]),
        'speed': np.concatenate([df_still['speed'].values, df_breeze['speed'].values, df_stormy['speed'].values]),
        'orientation': np.concatenate([df_still['orientation'].values, df_breeze['orientation'].values, df_stormy['orientation'].values]),
        'tank': np.concatenate([df_still['tank'].values, df_breeze['tank'].values, df_stormy['tank'].values]),
        'Condition': ['still'] * len(df_still) + ['breeze'] * len(df_breeze) + ['stormy'] * len(df_stormy)}

df = pd.DataFrame(data)
numobs = []
# Before normalization
for condition in conditions:
    # print(f"\n{condition}")
    for tank in tanks:
        n = len(df[(df['Condition'] == condition) & (df['tank'] == tank)])
        numobs.append(n)

# normalize number of counts per tank to the same number for each condition to avoid bias in the density estimation and plotting
min_numobs = min(numobs)

for condition in conditions:
    for tank in tanks:
        df_tank = df[(df['Condition'] == condition) & (df['tank'] == tank)]
        if len(df_tank) > min_numobs:
            df_tank_sampled = df_tank.sample(n=min_numobs, random_state=42)
            df = df.drop(df_tank.index)
            df = pd.concat([df, df_tank_sampled], ignore_index=True)

# Binning and smoothing parameters per measurement
if measurement == 'BarnacleLarvae_2102':
    TEST=False
    # Polar hist parameters
    ylim_pol_main = 0.6
    ylim_pol_overlay = 0.6
    
    # Violin plot parameters
    ylim_viol_main = 5
    ytick_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    xlim_viol = 0.25
    f_curve = 1.2
    f_curve_overlay = 0.8
    
    num_bins = ylim_viol_main*10 + 1 # Uneven number for centered pl
    smooth_kernel_size = 7  # odd number recommended
    smooth_sigma = 1      # controls smoothness; higher = smoother
    
elif measurement == 'Artemia_0805':
    TEST=False
    ylim_pol_main = 0.6
    ylim_pol_overlay = 0.6
    
    ylim_viol_main = 8
    ytick_list = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    xlim_viol = 0.25
    f_curve = 3.5
    f_curve_overlay = 2.8
    
    num_bins = ylim_viol_main*10 + 1 # Uneven number for centered pl
    smooth_kernel_size = 9  # odd number recommended
    smooth_sigma = 1     # controls smoothness; higher = smoother
        

elif measurement == 'Cladocerans_2605':
    TEST=False
    ylim_pol_main = 0.35
    ylim_pol_overlay = 0.45
    
    ylim_viol_main = 5
    ytick_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    xlim_viol = 0.25
    f_curve = 2 # fraction of cloud width to use for curve width
    f_curve_overlay = 1.6 # fraction of cloud width to use for overlay curve width
    num_bins = ylim_viol_main*10 + 1 # Uneven number for centered pl
    smooth_kernel_size = 9  # odd number recommended
    smooth_sigma = 1     # controls smoothness; higher = smoother


else:
    raise ValueError("Measurement not recognized. Please check the measurement variable.")

#%% Plot 


# Make the plot
bin_edges = np.linspace(-ylim_viol_main, ylim_viol_main, num_bins + 1)  # 3x more bins to capture quantization
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

kernel = gaussian_kernel(smooth_kernel_size, smooth_sigma)

conditions = ['still', 'breeze', 'stormy']
colors = sns.color_palette("Blues", 4)[1:]
# colors = sns.color_palette("Set2", 3)

# Create figure with cloud and boxplot for each condition in separate subplots, and polar histogram with swimming angle distribution on top of each subplot # Add some extra space for the cloud and boxplot
cloud_width = xlim_viol * 0.7  # Max width of the density cloud (80% of xlim)
curve_width = xlim_viol * f_curve  # Max width of the density curve (80% of cloud width)
cloud_offset = 0.1 * xlim_viol
box_width = cloud_width * 0.4  # Boxplot width as a fraction of cloud width
box_linewidth = 1

w_main_mm = 210/3*0.9
f_main_height = 1.3

max_points = len(data) # subsample per treatment for plotting only
points_scatter = 3000

cloud_x = cloud_width - cloud_offset

fig = plt.figure(figsize=(mm_to_inch(w_main_mm), mm_to_inch(w_main_mm) * f_main_height))
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection='polar') for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

y_grid = np.linspace(-ylim_viol_main, ylim_viol_main, 600)

for ax, condition, color in zip(axes_pol, conditions, colors):
    # Define treatment

    d = df[df['Condition'] == condition].dropna(subset=['orientation', 'speed'])
    orientation_vals = d['orientation'].values
    speed_vals = d['speed'].values

    # Robust, seam-free polar histogram
    theta = np.mod(orientation_vals, 2*np.pi)
    nbins = 50
    edges = np.linspace(0, 2*np.pi, nbins + 1)  # nbins + 1 edges for nbins bins
    counts, _ = np.histogram(theta, bins=edges, density=True)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align='edge', color=color, alpha=1, edgecolor='none')

    ax.set_thetamin(0)    
    ax.set_ylim(0, ylim_pol_main)
    
    # if condition == 'breeze':
        # ax.set_title(r'Distribution of $\theta$', fontsize=10, pad=5)
    
    # disable polar labels and ticks
    ax.set_yticks([ylim_pol_main/3, 2*ylim_pol_main/3])
    ax.set_yticklabels(['', ''])
    ax.set_xticklabels([])
    # ax.set_xticklabels([0, '' ,90, '', 180, '', 270, ''])


for ax, condition, color in zip(axes_viol, conditions, colors):

    if condition != 'still':
        ax.tick_params(axis='y', labelleft=False)
        ax.set_yticklabels([])
                
    data = df[df['Condition'] == condition]['dy'].dropna().values
    rand_ind = np.random.choice(len(data), size=min(100, len(data)), replace=False)
    ### Density cloud
    if TEST:
        kde = gaussian_kde(data[rand_ind])
    else:
        kde = gaussian_kde(data)
        
        
    density = kde(y_grid)

    # Normalize the area under the density curve to 1, then scale to fit the subplot width
    area = np.trapezoid(density, y_grid)
    density = density / area
    
    # Scale density to fit subplot width
    density = density * curve_width 

    ### Plot density curves and filled area
    ax.plot(density, y_grid, color='black', linewidth=1, zorder=3)
    ax.fill_betweenx(y_grid, 0, density, color=color, alpha=0.7, zorder=2)
    
    ### Plot histogram with black line and colored fille
    # ax.barh(bin_centers, -counts, height=bin_width, alpha=0.5, color='gray', zorder=1, antialiased=False, snap=True)
        
    print(f"for condition {condition}: mean = {mean:.3f}, std = {std:.3f}, n = {len(data)}")
    
    x_jitter = -np.random.uniform(cloud_offset, cloud_width-cloud_offset, size=points_scatter)
    ax.scatter(x_jitter, np.random.choice(data, size=points_scatter, replace=False), s=1.5, color='grey', alpha=0.12, linewidth=0, zorder=4)
    # ax.errorbar(-cloud_width/2, mean, yerr=std, fmt='o', color='black', markersize=3, zorder=5)
    
    # ### Boxplot
    ax.boxplot(data,
                vert=True,
                positions=[-cloud_width/2],
                widths=box_width,                
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='white', edgecolor='black', linewidth=box_linewidth, alpha=0.8),
                medianprops=dict(color='black', linewidth=box_linewidth),
                whiskerprops=dict(color='black', linewidth=box_linewidth),
                capprops=dict(color='black', linewidth=box_linewidth),
                zorder = 5)

    ### Styling
    # if condition == 'breeze':
    #     ax.set_title('Distribution of dy', fontsize=10, pad=5)
    ax.set_xlabel(condition, fontsize=10)

    ax.tick_params(axis='y', labelsize=8, length=0, rotation=60)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis='y', zorder=0)
    ax.axvline(x=0, color='gray', alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main - 0.5, ylim_viol_main + 0.5])

    for spine in ax.spines.values():
        spine.set_visible(False)


# Add outer rectangle, same as your current plot
ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor='black',
    linewidth=1,
    zorder=100
)

ax_bg = fig.add_artist(rect)

if not TEST:
    fig.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_full_per_species.png'), dpi=1200, transparent=True, bbox_inches='tight')
    fig.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_full_per_species.svg'), dpi=1200, transparent=True,  bbox_inches='tight')


# Make the plot with individual overlay for each tank, to show variability across tanks, and mean ± SD for each condition

# Define bin edges for consistent binning across all conditions
# Use high resolution bins to avoid quantization gaps from pixel-based coordinate data
# Raw coordinates are integers, scaled by ~0.25 mm/pxl, so velocities are quantized
bin_edges = np.linspace(-ylim_viol_main, ylim_viol_main, num_bins + 1)  # 3x more bins to capture quantization
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

kernel = gaussian_kernel(smooth_kernel_size, smooth_sigma)

conditions = ['still', 'breeze', 'stormy']
# colors = sns.color_palette("Blues", 4)[1:]
colors = ["#0072B2","#B20000", "#E69F00", "#009E73", "#CC79A7"][:len(tanks)]

# Create figure with cloud and boxplot for each condition in separate subplots, and polar histogram with swimming angle distribution on top of each subplot # Add some extra space for the cloud and boxplot
cloud_width = xlim_viol * 0.7  # Max width of the density cloud (80% of xlim)
curve_width = xlim_viol * f_curve_overlay  # Max width of the density curve (80% of cloud width)
cloud_offset = 0.1 * xlim_viol
box_width = cloud_width * 0.4  # Boxplot width as a fraction of cloud width
box_linewidth = 1

max_points = len(data) # subsample per treatment for plotting only
points_scatter = 3000

cloud_x = cloud_width - cloud_offset

fig = plt.figure(figsize=(mm_to_inch(w_main_mm), mm_to_inch(w_main_mm) * f_main_height))
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection='polar') for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

y_grid = np.linspace(-ylim_viol_main, ylim_viol_main, 600)

for ax, condition in zip(axes_pol, conditions):
    # Define treatment

    d = df[df['Condition'] == condition].dropna(subset=['orientation', 'speed'])
    orientation_vals = d['orientation'].values
    speed_vals = d['speed'].values

    # Robust, seam-free polar histogram
    theta = np.mod(orientation_vals, 2*np.pi)
    nbins = 50
    edges = np.linspace(0, 2*np.pi, nbins + 1)  # nbins + 1 edges for nbins bins
    counts, _ = np.histogram(theta, bins=edges, density=True)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align='edge', color='gray', alpha=0.5, edgecolor='none')
    
    for tank in tanks:
        d_tank = d[d['tank'] == tank]
        color = colors[tank-1]  # Assuming tank numbers start at 1 and are consecutive
        orientation_vals_tank = d_tank['orientation'].values
        theta_tank = np.mod(orientation_vals_tank, 2*np.pi)
        counts_tank, _ = np.histogram(theta_tank, bins=edges, density=True)
        ax.plot(edges[:-1] + width/2, counts_tank, color=color, linewidth=1.2, alpha=0.8)

    ax.set_thetamin(0)    
    ax.set_ylim(0, ylim_pol_overlay)
    
    # if condition == 'breeze':
        # ax.set_title(r'Distribution of $\theta$', fontsize=10, pad=5)
    
    # disable polar labels and ticks
    ax.set_yticks([ylim_pol_overlay/3, 2*ylim_pol_overlay/3])
    ax.set_yticklabels(['', ''])
    ax.set_xticklabels([])
    # ax.set_xticklabels([0, '' ,90, '', 180, '', 270, ''])


for ax, condition in zip(axes_viol, conditions):

    if condition != 'still':
        ax.tick_params(axis='y', labelleft=False)
        ax.set_yticklabels([])
                
    data = df[df['Condition'] == condition]['dy'].dropna().values
    rand_ind = np.random.choice(len(data), size=min(100, len(data)), replace=False)
    ### Density cloud
    if TEST:
        kde = gaussian_kde(data[rand_ind])
    else:
        kde = gaussian_kde(data)
        
        
    density = kde(y_grid)

    # Normalize the area under the density curve to 1, then scale to fit the subplot width
    area = np.trapezoid(density, y_grid)
    density = density / area
    
    # Scale density to fit subplot width
    density = density * curve_width 

    ### Plot density curves and filled area
    # ax.plot(density, y_grid, color='black', linewidth=1, zorder=3)
    ax.fill_betweenx(y_grid, 0, density, color='gray', alpha=0.7, zorder=2)
    
    ### Plot overlay density curves for each tank
    for tank in tanks:
        d_tank = df[(df['Condition'] == condition) & (df['tank'] == tank)].dropna(subset=['dy'])
        data_tank = d_tank['dy'].values
        
        if len(data_tank) > 0:
            kde_tank = gaussian_kde(data_tank)
            density_tank = kde_tank(y_grid)
            area_tank = np.trapezoid(density_tank, y_grid)
            density_tank = density_tank / area_tank * curve_width
            color = colors[tank-1]  # Assuming tank numbers start at 1 and are consecutive
            ax.plot(density_tank, y_grid, color=color, linewidth=1, zorder=3, alpha=0.8)
            
    x_jitter = -np.random.uniform(cloud_offset, cloud_width-cloud_offset, size=points_scatter)
    ax.scatter(x_jitter, np.random.choice(data, size=points_scatter, replace=False), s=1.5, color='grey', alpha=0.12, linewidth=0, zorder=4)

    ### Boxplot
    # Tank-level mean dy points

    x_points = np.linspace(-cloud_width*0.85, -cloud_width*0.35, len(tanks))
    for x, tank in zip(x_points, tanks):
        d_tank = df[(df['Condition'] == condition) & (df['tank'] == tank)]
        tank_mean = np.mean(d_tank['dy'].dropna().values)
        tank_sd = np.std(d_tank['dy'].dropna().values, ddof=1)

        ax.scatter(x, tank_mean, s=10, color=colors[tank-1], edgecolor='none', linewidth=0.4, zorder=6)
        ax.errorbar(x, tank_mean, yerr=tank_sd, fmt='o', color=colors[tank-1], markersize=3, capsize=2, linewidth=1, zorder=7)



    # ax.errorbar(-cloud_width*0.5, mean_tank,
    #             yerr=sd_tank,
    #             fmt='o',
    #             color=colors[tank-1],
    #             markersize=3,
    #             capsize=2,
    #             linewidth=1,
    #             zorder=7)

    ### Mean ± SD
    # mean_val = np.mean(data)
    # std_val = np.std(data)

    ### Styling
    # if condition == 'breeze':
    #     ax.set_title('Distribution of dy', fontsize=10, pad=5)
    ax.set_xlabel(condition, fontsize=10)

    ax.tick_params(axis='y', labelsize=8, length=0, rotation=60)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis='y', zorder=0)
    ax.axvline(x=0, color='gray', alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main - 0.5, ylim_viol_main + 0.5])

    for spine in ax.spines.values():
        spine.set_visible(False)


# Add outer rectangle, same as your current plot
ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor='black',
    linewidth=1,
    zorder=100
)

ax_bg = fig.add_artist(rect)

if not TEST:
    fig.savefig(os.path.join(path_to_figure_savelocation_overlay, f'{measurement}_full_per_species.png'), dpi=1200, transparent=True, bbox_inches='tight')
    fig.savefig(os.path.join(path_to_figure_savelocation_overlay, f'{measurement}_full_per_species.svg'), dpi=1200, transparent=True, bbox_inches='tight')

#%% Create a flat legend for the tanks
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Tank {tank}') for i, tank in enumerate(tanks)]

fig_legend = plt.figure(figsize=(2, 0.1))
legend_ax = fig_legend.add_subplot(111)
legend_ax.legend(handles=legend_elements, ncol = len(tanks), fontsize=8)
legend_ax.axis('off')
fig_legend.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.png'), dpi=1200, transparent=True, bbox_inches='tight')
fig_legend.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.svg'), dpi=1200, transparent=True, bbox_inches='tight')

print("saved legend at", os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.png'))
print("saved legend at", os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.svg'))

#%% Plot control measurements for all species: no overlays + tank overlays

w_control_mm = 210/2*0.9
factor_control_height = 1.2

control_measurements = [
    "Artemia_0805",
    "Cladocerans_2605",
    "BarnacleLarvae_2102",
]

species_labels = {
    "BarnacleLarvae_2102": "Barnacle larvae",
    "Artemia_0805": "Artemia larvae",
    "Cladocerans_2605": "Water fleas",
}

control_plot_settings = {
    "ylim_pol_main": 0.5,
    "ylim_pol_overlay": 0.5,

    "ylim_viol_main": 6,
    "ytick_list": [-6, -4, -2, 0, 2, 4, 6],
    "xlim_viol": 0.25,

    "f_curve": 2,
    "f_curve_overlay": 2.2,

    "num_bins": 71,
    "points_scatter": 3000,
}
# Load control data
dfs_control = []

for measurement_i in control_measurements:
    path_to_trajectories_i = os.path.join(path_to_tracking_data, measurement_i, "trajectories_scaled")
    df_i = pickle.load(open(os.path.join(path_to_trajectories_i, "traj_nothing_scaled.pkl"), "rb"))

    df_i = df_i.copy()
    df_i["Species"] = species_labels[measurement_i]
    df_i["Measurement"] = measurement_i

    dfs_control.append(df_i)

df_control_all = pd.concat(dfs_control, ignore_index=True)

# Normalize number of observations per species/tank
numobs = []
for measurement_i in control_measurements:
    tanks_i = [1, 2, 3, 4, 5] if measurement_i == "Cladocerans_2605" else [1, 2, 3]
    species_i = species_labels[measurement_i]

    for tank in tanks_i:
        n = len(df_control_all[(df_control_all["Species"] == species_i) & (df_control_all["tank"] == tank)])
        if n > 0:
            numobs.append(n)

min_numobs = min(numobs)

df_control_norm = []
for measurement_i in control_measurements:
    tanks_i = [1, 2, 3, 4, 5] if measurement_i == "Cladocerans_2605" else [1, 2, 3]
    species_i = species_labels[measurement_i]

    for tank in tanks_i:
        df_tank = df_control_all[(df_control_all["Species"] == species_i) & (df_control_all["tank"] == tank)]
        if len(df_tank) > min_numobs:
            df_tank = df_tank.sample(n=min_numobs, random_state=42)
        df_control_norm.append(df_tank)

df_control_plot = pd.concat(df_control_norm, ignore_index=True)

ylim_pol_main = control_plot_settings["ylim_pol_main"]
ylim_pol_overlay = control_plot_settings["ylim_pol_overlay"]

ylim_viol_main = control_plot_settings["ylim_viol_main"]
ytick_list = control_plot_settings["ytick_list"]
xlim_viol = control_plot_settings["xlim_viol"]

f_curve = control_plot_settings["f_curve"]
f_curve_overlay = control_plot_settings["f_curve_overlay"]

num_bins = control_plot_settings["num_bins"]
points_scatter = control_plot_settings["points_scatter"]

bin_edges = np.linspace(-ylim_viol_main, ylim_viol_main, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

species_order = [species_labels[m] for m in control_measurements]
blue_colors = sns.color_palette("Blues", 4)[1:]

# color control is slightly darker than stormy
color_control = sns.color_palette("Blues", 4)[2]
tank_colors = ["#0072B2", "#B20000", "#E69F00", "#009E73", "#CC79A7"]

y_grid = np.linspace(-ylim_viol_main, ylim_viol_main, 600)

cloud_width = xlim_viol * 0.7
cloud_offset = 0.1 * xlim_viol
box_width = cloud_width * 0.4
box_linewidth = 1

# plot control, no overlays

curve_width = xlim_viol * f_curve

fig = plt.figure(figsize=(
    mm_to_inch(w_control_mm),
    mm_to_inch(w_control_mm) * factor_control_height
))

gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection="polar") for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

for ax, species in zip(axes_pol, species_order):
    d = df_control_plot[df_control_plot["Species"] == species].dropna(subset=["orientation", "speed"])

    theta = np.mod(d["orientation"].values, 2 * np.pi)
    nbins = 50
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    counts, _ = np.histogram(theta, bins=edges, density=True)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align="edge", color=color_control, alpha=1, edgecolor="none")

    ax.set_thetamin(0)
    ax.set_ylim(0, ylim_pol_main)
    ax.set_yticks([ylim_pol_main / 3, 2 * ylim_pol_main / 3])
    ax.set_yticklabels(["", ""])
    ax.set_xticklabels([])

for ax, species in zip(axes_viol, species_order):
    if species != species_order[0]:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])

    data = df_control_plot[df_control_plot["Species"] == species]["dy"].dropna().values

    kde = gaussian_kde(data)
    
    # normalize with respect to max peak height, to show differences in distribution shape rather than absolute density (which is affected by number of observations and variability across tanks)
    density = kde(y_grid)
    area = np.trapezoid(density, y_grid)
    peak = np.max(density)
    density = density / peak
    density = density * curve_width * 0.4

    ax.plot(density, y_grid, color="black", linewidth=1, zorder=3)
    ax.fill_betweenx(y_grid, 0, density, color=color_control, alpha=0.7, zorder=2)

    n_scatter = min(points_scatter, len(data))
    x_jitter = -np.random.uniform(cloud_offset, cloud_width - cloud_offset, size=n_scatter)
    ax.scatter(
        x_jitter,
        np.random.choice(data, size=n_scatter, replace=False),
        s=1.5,
        color="grey",
        alpha=0.12,
        linewidth=0,
        zorder=4,
        rasterized=True,
    )

    ax.boxplot(
        data,
        vert=True,
        positions=[-cloud_width / 2],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=box_linewidth, alpha=0.8),
        medianprops=dict(color="black", linewidth=box_linewidth),
        whiskerprops=dict(color="black", linewidth=box_linewidth),
        capprops=dict(color="black", linewidth=box_linewidth),
        zorder=5,
    )

    ax.set_xlabel(species, fontsize=10)
    ax.tick_params(axis="y", labelsize=8, length=0, rotation=60)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis="y", zorder=0)
    ax.axvline(x=0, color="gray", alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main - 0.5, ylim_viol_main + 0.5])

    for spine in ax.spines.values():
        spine.set_visible(False)

ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor="black",
    linewidth=1,
    zorder=100,
)

fig.add_artist(rect)

fig.savefig(
    os.path.join(save_control_main, "control_all_species.png"),
    dpi=1200,
    bbox_inches="tight",
    transparent=True,
)
fig.savefig(
    os.path.join(save_control_main, "control_all_species.svg"),
    dpi=1200,
    bbox_inches="tight",
    transparent=True,
)

# Plot control, tank overlays

curve_width = xlim_viol * f_curve_overlay

fig = plt.figure(figsize=(
    mm_to_inch(w_control_mm),
    mm_to_inch(w_control_mm) * factor_control_height
))

gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection="polar") for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

for ax, species, measurement_i in zip(axes_pol, species_order, control_measurements):
    d = df_control_plot[df_control_plot["Species"] == species].dropna(subset=["orientation", "speed"])

    theta = np.mod(d["orientation"].values, 2 * np.pi)
    nbins = 50
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    counts, _ = np.histogram(theta, bins=edges, density=True)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align="edge", color="gray", alpha=0.4, edgecolor="none")

    tanks_i = sorted(d["tank"].dropna().astype(int).unique())

    peak = max(counts)
    for tank in tanks_i:
        d_tank = d[d["tank"] == tank]
        color = tank_colors[tank - 1]

        theta_tank = np.mod(d_tank["orientation"].values, 2 * np.pi)
        counts_tank, _ = np.histogram(theta_tank, bins=edges, density=True)
        if max(counts_tank) > peak:
            peak = max(counts_tank)
        ax.plot(edges[:-1] + width / 2, counts_tank, color=color, linewidth=1.2, alpha=0.8)

    ylim_pol_overlay = peak * 1.2
    
    ax.set_thetamin(0)
    ax.set_ylim(0, ylim_pol_overlay)
    ax.set_yticks([ylim_pol_overlay / 3, 2 * ylim_pol_overlay / 3])
    ax.set_yticklabels(["", ""])
    ax.set_xticklabels([])

for ax, species in zip(axes_viol, species_order):
    if species != species_order[0]:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])

    data = df_control_plot[df_control_plot["Species"] == species]["dy"].dropna().values

    kde = gaussian_kde(data)
    density = kde(y_grid)
    
    peak = np.max(density)
    density = density / peak
    density = density * curve_width * 0.35
    area_full = np.trapezoid(density, y_grid)

    ax.fill_betweenx(y_grid, 0, density, color="gray", alpha=0.4, zorder=2)

    tanks_i = sorted(df_control_plot[df_control_plot["Species"] == species]["tank"].dropna().astype(int).unique())

    for tank in tanks_i:
        d_tank = df_control_plot[
            (df_control_plot["Species"] == species) &
            (df_control_plot["tank"] == tank)
        ].dropna(subset=["dy"])

        data_tank = d_tank["dy"].values

        if len(data_tank) > 1:
            kde_tank = gaussian_kde(data_tank)
                    
            density_tank = kde_tank(y_grid)
            
            area_tank = np.trapezoid(density_tank, y_grid)
            
            density_tank = density_tank / area_tank
            density_tank = density_tank * area_full

            color = tank_colors[tank - 1]
            ax.plot(density_tank, y_grid, color=color, linewidth=1, zorder=3, alpha=0.8)

    n_scatter = min(points_scatter, len(data))
    x_jitter = -np.random.uniform(cloud_offset, cloud_width - cloud_offset, size=n_scatter)
    ax.scatter(
        x_jitter,
        np.random.choice(data, size=n_scatter, replace=False),
        s=1.5,
        color="grey",
        alpha=0.12,
        linewidth=0,
        zorder=4,
        rasterized=True,
    )

    x_points = np.linspace(-cloud_width * 0.85, -cloud_width * 0.35, len(tanks_i))

    for x, tank in zip(x_points, tanks_i):
        d_tank = df_control_plot[
            (df_control_plot["Species"] == species) &
            (df_control_plot["tank"] == tank)
        ]

        tank_mean = np.mean(d_tank["dy"].dropna().values)
        tank_sd = np.std(d_tank["dy"].dropna().values, ddof=1)

        color = tank_colors[tank - 1]
        ax.errorbar(
            x,
            tank_mean,
            yerr=tank_sd,
            fmt="o",
            color=color,
            markersize=3,
            capsize=2,
            linewidth=1,
            zorder=7,
        )

    ax.set_xlabel(species, fontsize=10)
    ax.tick_params(axis="y", labelsize=8, length=0, rotation=60)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis="y", zorder=0)
    ax.axvline(x=0, color="gray", alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main - 0.5, ylim_viol_main + 0.5])

    for spine in ax.spines.values():
        spine.set_visible(False)

ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor="black",
    linewidth=1,
    zorder=100,
)

fig.add_artist(rect)

fig.savefig(
    os.path.join(save_control_overlay, "control_all_species_overlay.png"),
    dpi=1200,
    bbox_inches="tight",
    transparent=True,
)
fig.savefig(
    os.path.join(save_control_overlay, "control_all_species_overlay.svg"),
    dpi=1200,
    bbox_inches="tight",
    transparent=True,
)

print("Control means ± SD:")
for species in species_order:
    data = df_control_plot[df_control_plot["Species"] == species]["dy"].dropna().values
    print(f"{species}: mean = {np.mean(data):.2f}, std = {np.std(data):.2f}, n = {len(data):,}")
