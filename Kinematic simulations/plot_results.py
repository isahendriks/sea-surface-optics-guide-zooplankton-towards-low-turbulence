#%% Import necessary libraries
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
})

color_still, color_breeze, color_stormy = '#94c4df', '#4a98c9', '#1764ab'

DATA_DIR = Path(__file__).parent / "data"
NU = 1e-6  # kinematic viscosity [m^2/s], same as in the simulations

USE_WEIGHTED_V = True  # False: plain circular variance V, True: speed-weighted V_w
if USE_WEIGHTED_V:
    cv_mean, cv_std = "mean_weighted_circular_variances", "std_weighted_circular_variances"
    stem = 'speed_and_weighted_circular_variance_by_species_panels'
else:
    cv_mean, cv_std = "mean_circular_variances", "std_circular_variances"
    stem = 'speed_and_circular_variance_by_species_panels'

WIND_X_AXIS = False  # False: log epsilon on the bottom axis (wind on top), True: linear wind speed on the bottom axis (epsilon on top)
if WIND_X_AXIS:
    stem += '_wind_axis'

#%% Load simulation results
artemia_still = pickle.load(open("plankton_turbulence_results_Artemia_0805_still.pkl", "rb"))
artemia_breeze = pickle.load(open("plankton_turbulence_results_Artemia_0805_breeze.pkl", "rb"))
artemia_stormy = pickle.load(open("plankton_turbulence_results_Artemia_0805_stormy.pkl", "rb"))

cladocerans_still = pickle.load(open("plankton_turbulence_results_Cladocerans_2605_still.pkl", "rb"))
cladocerans_breeze = pickle.load(open("plankton_turbulence_results_Cladocerans_2605_breeze.pkl", "rb"))
cladocerans_stormy = pickle.load(open("plankton_turbulence_results_Cladocerans_2605_stormy.pkl", "rb"))

barnacle_still = pickle.load(open("plankton_turbulence_results_BarnacleLarvae_2102_still.pkl", "rb"))
barnacle_breeze = pickle.load(open("plankton_turbulence_results_BarnacleLarvae_2102_breeze.pkl", "rb"))
barnacle_stormy = pickle.load(open("plankton_turbulence_results_BarnacleLarvae_2102_stormy.pkl", "rb"))

# Same epsilon values in all runs, the first one is the eps = 0 control
eps = np.array(artemia_still["epsilon_values"])

#%% Mean swim speed U [mm/s] from the tracked trajectories, and the epsilon where the swimming number is 1
# (Kolmogorov velocity (eps*nu)^(1/4) equals U, so eps = U^4 / nu)
U_artemia_still = pd.read_pickle(DATA_DIR / "Artemia_0805" / "traj_still_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()
U_artemia_breeze = pd.read_pickle(DATA_DIR / "Artemia_0805" / "traj_breeze_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()
U_artemia_stormy = pd.read_pickle(DATA_DIR / "Artemia_0805" / "traj_stormy_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()

U_cladocerans_still = pd.read_pickle(DATA_DIR / "Cladocerans_2605" / "traj_still_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()
U_cladocerans_breeze = pd.read_pickle(DATA_DIR / "Cladocerans_2605" / "traj_breeze_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()
U_cladocerans_stormy = pd.read_pickle(DATA_DIR / "Cladocerans_2605" / "traj_stormy_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()

U_barnacle_still = pd.read_pickle(DATA_DIR / "BarnacleLarvae_2102" / "traj_still_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()
U_barnacle_breeze = pd.read_pickle(DATA_DIR / "BarnacleLarvae_2102" / "traj_breeze_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()
U_barnacle_stormy = pd.read_pickle(DATA_DIR / "BarnacleLarvae_2102" / "traj_stormy_scaled.pkl").dropna(subset=["speed", "orientation"])["speed"].mean()

eps_phi1_artemia_still = (U_artemia_still * 1e-3) ** 4 / NU
eps_phi1_artemia_breeze = (U_artemia_breeze * 1e-3) ** 4 / NU
eps_phi1_artemia_stormy = (U_artemia_stormy * 1e-3) ** 4 / NU

eps_phi1_cladocerans_still = (U_cladocerans_still * 1e-3) ** 4 / NU
eps_phi1_cladocerans_breeze = (U_cladocerans_breeze * 1e-3) ** 4 / NU
eps_phi1_cladocerans_stormy = (U_cladocerans_stormy * 1e-3) ** 4 / NU

eps_phi1_barnacle_still = (U_barnacle_still * 1e-3) ** 4 / NU
eps_phi1_barnacle_breeze = (U_barnacle_breeze * 1e-3) ** 4 / NU
eps_phi1_barnacle_stormy = (U_barnacle_stormy * 1e-3) ** 4 / NU

#%% Equivalent 10 m wind speed for each epsilon, from the law of the wall
# epsilon = u*_w^3 / (kappa * z)  (law of the wall; e.g. Thorpe 2007, Terray et al. 1996),
# u*_w = sqrt(rho_air / rho_water) * u*_air  (stress continuity across the air-sea interface),
# u*_air = sqrt(Cd) * U10, with Cd from Large & Pond (1981). Same as epsilon_from_weather in turbulence_model.py.
Z_TANK = 0.09   # depth at which epsilon is evaluated [m] (tank depth)
KAPPA = 0.4     # von Karman constant
RHO_AIR, RHO_WATER = 1.225, 1025.0  # [kg/m^3]

def eps_from_wind(U10, z=Z_TANK):
    """Dissipation rate [m^2/s^3] at depth z [m] for 10 m wind speed U10 [m/s]."""
    U10 = np.asarray(U10, dtype=float)
    Cd = np.where(U10 < 11, 1.2e-3, (0.49 + 0.065 * U10) * 1e-3)
    u_star_water = np.sqrt(Cd) * U10 * np.sqrt(RHO_AIR / RHO_WATER)
    return u_star_water**3 / (KAPPA * z)

# Cd depends on U10, so invert numerically on a fine log grid
_U10_grid = np.logspace(-5, 2, 4000)
_log_eps_grid = np.log10(eps_from_wind(_U10_grid))

def wind_from_eps(epsilon):
    """10 m wind speed [m/s] that gives dissipation rate epsilon [m^2/s^3] at depth Z_TANK."""
    epsilon = np.clip(np.asarray(epsilon, dtype=float), 1e-300, None)
    return 10 ** np.interp(np.log10(epsilon), _log_eps_grid, np.log10(_U10_grid))

print("Equivalent U10 [m/s] per simulated epsilon:")
for e in eps[1:]:
    print(f"  eps = {e:.2e} m^2/s^3  ->  U10 = {wind_from_eps(e):.3g} m/s")

#%% x-axis set-up, depending on WIND_X_AXIS
EPS_LABEL = r'$\varepsilon$ [m²/s³]'
WIND_LABEL = r'$v_{\mathrm{wind}}$ [m/s]'

if WIND_X_AXIS:
    xf = wind_from_eps   # plot everything against wind speed
    XLABEL = WIND_LABEL
else:
    xf = lambda e: e     # plot everything against epsilon
    XLABEL = EPS_LABEL
x = xf(eps)

def vline(ax, epsilon, **kwargs):
    """Vertical line at dissipation rate epsilon, in whatever the x-axis units are."""
    ax.axvline(xf(epsilon), **kwargs)

def format_x_axis(ax):
    if WIND_X_AXIS:
        ax.set_xlim(0, 13)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    else:
        ax.set_xscale('log')
        ax.set_xlim(10**-13.5, 1e-4)

def add_top_axis(ax):
    """Top axis showing the other quantity: wind speed when x is epsilon, epsilon when x is wind speed."""
    if WIND_X_AXIS:
        sec = ax.secondary_xaxis('top', functions=(eps_from_wind, wind_from_eps))
        sec.set_xticks([1e-6, 1e-5, 1e-4], labels=[r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$'])
        sec.set_xlabel(EPS_LABEL)
    else:
        sec = ax.secondary_xaxis('top', functions=(wind_from_eps, eps_from_wind))
        sec.set_xticks([0.01, 0.1, 1, 10], labels=['0.01', '0.1', '1', '10'])
        sec.set_xlabel(WIND_LABEL)
    return sec

#%% Speed (top) and circular variance (bottom) per species
fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 4.6), sharex=True, sharey='row')

### Artemia, speed
ax = axes[0, 0]
ax.set_title('Artemia', fontstyle='italic')

mean, std = np.array(artemia_still["mean_upward_velocities"]), np.array(artemia_still["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_still, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_still, alpha=0.2)
ax.axhline(mean[0], color=color_still, linestyle='--')
vline(ax, eps_phi1_artemia_still, color=color_still, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(artemia_breeze["mean_upward_velocities"]), np.array(artemia_breeze["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_breeze, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_breeze, alpha=0.2)
ax.axhline(mean[0], color=color_breeze, linestyle='--')
vline(ax, eps_phi1_artemia_breeze, color=color_breeze, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(artemia_stormy["mean_upward_velocities"]), np.array(artemia_stormy["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_stormy, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_stormy, alpha=0.2)
ax.axhline(mean[0], color=color_stormy, linestyle='--')
vline(ax, eps_phi1_artemia_stormy, color=color_stormy, linestyle='-.', linewidth=1.4, alpha=0.9)

format_x_axis(ax)
ax.set_box_aspect(1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylabel('Ascent rate [mm/s]')

### Cladocerans, speed
ax = axes[0, 1]
ax.set_title('Cladocerans')

mean, std = np.array(cladocerans_still["mean_upward_velocities"]), np.array(cladocerans_still["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_still, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_still, alpha=0.2)
ax.axhline(mean[0], color=color_still, linestyle='--')
vline(ax, eps_phi1_cladocerans_still, color=color_still, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(cladocerans_breeze["mean_upward_velocities"]), np.array(cladocerans_breeze["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_breeze, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_breeze, alpha=0.2)
ax.axhline(mean[0], color=color_breeze, linestyle='--')
vline(ax, eps_phi1_cladocerans_breeze, color=color_breeze, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(cladocerans_stormy["mean_upward_velocities"]), np.array(cladocerans_stormy["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_stormy, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_stormy, alpha=0.2)
ax.axhline(mean[0], color=color_stormy, linestyle='--')
vline(ax, eps_phi1_cladocerans_stormy, color=color_stormy, linestyle='-.', linewidth=1.4, alpha=0.9)

format_x_axis(ax)
ax.set_box_aspect(1)
ax.grid(True, linestyle='--', alpha=0.7)

### Barnacle larvae, speed
ax = axes[0, 2]
ax.set_title('Barnacle larvae')

mean, std = np.array(barnacle_still["mean_upward_velocities"]), np.array(barnacle_still["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_still, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_still, alpha=0.2)
ax.axhline(mean[0], color=color_still, linestyle='--')
vline(ax, eps_phi1_barnacle_still, color=color_still, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(barnacle_breeze["mean_upward_velocities"]), np.array(barnacle_breeze["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_breeze, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_breeze, alpha=0.2)
ax.axhline(mean[0], color=color_breeze, linestyle='--')
vline(ax, eps_phi1_barnacle_breeze, color=color_breeze, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(barnacle_stormy["mean_upward_velocities"]), np.array(barnacle_stormy["std_upward_velocities"])
ax.plot(x[1:], mean[1:], 'o', color=color_stormy, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_stormy, alpha=0.2)
ax.axhline(mean[0], color=color_stormy, linestyle='--')
vline(ax, eps_phi1_barnacle_stormy, color=color_stormy, linestyle='-.', linewidth=1.4, alpha=0.9)

format_x_axis(ax)
ax.set_box_aspect(1)
ax.grid(True, linestyle='--', alpha=0.7)

for ax in axes[0]:
    add_top_axis(ax)

### Artemia, circular variance
ax = axes[1, 0]

mean, std = np.array(artemia_still[cv_mean]), np.array(artemia_still[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_still, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_still, alpha=0.2)
ax.axhline(mean[0], color=color_still, linestyle='--')
vline(ax, eps_phi1_artemia_still, color=color_still, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(artemia_breeze[cv_mean]), np.array(artemia_breeze[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_breeze, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_breeze, alpha=0.2)
ax.axhline(mean[0], color=color_breeze, linestyle='--')
vline(ax, eps_phi1_artemia_breeze, color=color_breeze, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(artemia_stormy[cv_mean]), np.array(artemia_stormy[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_stormy, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_stormy, alpha=0.2)
ax.axhline(mean[0], color=color_stormy, linestyle='--')
vline(ax, eps_phi1_artemia_stormy, color=color_stormy, linestyle='-.', linewidth=1.4, alpha=0.9)

format_x_axis(ax)
ax.set_box_aspect(1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylabel('Circular variance')
ax.set_xlabel(XLABEL)

### Cladocerans, circular variance
ax = axes[1, 1]

mean, std = np.array(cladocerans_still[cv_mean]), np.array(cladocerans_still[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_still, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_still, alpha=0.2)
ax.axhline(mean[0], color=color_still, linestyle='--')
vline(ax, eps_phi1_cladocerans_still, color=color_still, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(cladocerans_breeze[cv_mean]), np.array(cladocerans_breeze[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_breeze, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_breeze, alpha=0.2)
ax.axhline(mean[0], color=color_breeze, linestyle='--')
vline(ax, eps_phi1_cladocerans_breeze, color=color_breeze, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(cladocerans_stormy[cv_mean]), np.array(cladocerans_stormy[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_stormy, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_stormy, alpha=0.2)
ax.axhline(mean[0], color=color_stormy, linestyle='--')
vline(ax, eps_phi1_cladocerans_stormy, color=color_stormy, linestyle='-.', linewidth=1.4, alpha=0.9)

format_x_axis(ax)
ax.set_box_aspect(1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel(XLABEL)

### Barnacle larvae, circular variance
ax = axes[1, 2]

mean, std = np.array(barnacle_still[cv_mean]), np.array(barnacle_still[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_still, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_still, alpha=0.2)
ax.axhline(mean[0], color=color_still, linestyle='--')
vline(ax, eps_phi1_barnacle_still, color=color_still, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(barnacle_breeze[cv_mean]), np.array(barnacle_breeze[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_breeze, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_breeze, alpha=0.2)
ax.axhline(mean[0], color=color_breeze, linestyle='--')
vline(ax, eps_phi1_barnacle_breeze, color=color_breeze, linestyle='-.', linewidth=1.4, alpha=0.9)

mean, std = np.array(barnacle_stormy[cv_mean]), np.array(barnacle_stormy[cv_std])
ax.plot(x[1:], mean[1:], 'o', color=color_stormy, markersize=3.5)
ax.fill_between(x[1:], mean[1:] - std[1:], mean[1:] + std[1:], color=color_stormy, alpha=0.2)
ax.axhline(mean[0], color=color_stormy, linestyle='--')
vline(ax, eps_phi1_barnacle_stormy, color=color_stormy, linestyle='-.', linewidth=1.4, alpha=0.9)

format_x_axis(ax)
ax.set_box_aspect(1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel(XLABEL)

fig.tight_layout(w_pad=2.5, h_pad=0.8)
fig.savefig(f'{stem}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{stem}.pdf', bbox_inches='tight')
fig.savefig(f'{stem}.svg', bbox_inches='tight')

#%% Legends, saved as separate files so they can be placed freely in the layout
condition_handles = [
    Line2D([0], [0], marker='o', color=color_still, linestyle='none', markersize=3.5, label='Still'),
    Line2D([0], [0], marker='o', color=color_breeze, linestyle='none', markersize=3.5, label='Breeze'),
    Line2D([0], [0], marker='o', color=color_stormy, linestyle='none', markersize=3.5, label='Stormy'),
]
style_handles = [
    Line2D([0], [0], marker='o', color='gray', linestyle='none', markersize=3.5, label='Measured data (+/- 1 SD)'),
    Line2D([0], [0], color='gray', linestyle='-.', label='Swimming number = 1'),
    Line2D([0], [0], color='gray', linestyle='--', label='No-turbulence control'),
]

legend_fig = plt.figure(figsize=(fig.get_figwidth(), 0.4))
legend_fig.legend(handles=condition_handles, loc='center', ncol=3, fontsize=9, frameon=False, columnspacing=1.5, handletextpad=0.5)
legend_fig.savefig(f'{stem}_legend_identity.png', dpi=300, bbox_inches='tight')
legend_fig.savefig(f'{stem}_legend_identity.pdf', bbox_inches='tight')
legend_fig.savefig(f'{stem}_legend_identity.svg', bbox_inches='tight')

legend_fig = plt.figure(figsize=(fig.get_figwidth(), 0.4))
legend_fig.legend(handles=style_handles, loc='center', ncol=3, fontsize=9, frameon=False, columnspacing=1.5, handletextpad=0.5)
legend_fig.savefig(f'{stem}_legend_styles.png', dpi=300, bbox_inches='tight')
legend_fig.savefig(f'{stem}_legend_styles.pdf', bbox_inches='tight')
legend_fig.savefig(f'{stem}_legend_styles.svg', bbox_inches='tight')

# %%
