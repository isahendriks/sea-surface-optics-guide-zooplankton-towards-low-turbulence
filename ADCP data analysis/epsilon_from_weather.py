"""
Turbulent kinetic energy dissipation rate (epsilon) at the field site, estimated from the
wind speed and significant wave height of the weather station, to compare with the
dissipation rates used in the kinematic simulations.

Two estimates are computed from the water-side friction velocity u*_w:
  - wave-affected surface layer: epsilon = ENHANCEMENT_FACTOR * u*_w^3 / (kappa * 0.6 Hs)
  - law of the wall at depth z (no wave enhancement): epsilon = u*_w^3 / (kappa * z)

References
  Large, W.G. & Pond, S. (1981) Open ocean momentum flux measurements in moderate to
      strong winds. J. Phys. Oceanogr., 11, 324-336.
  Craig, P.D. & Banner, M.L. (1994) Modeling wave-enhanced turbulence in the ocean
      surface layer. J. Phys. Oceanogr., 24, 2546-2559.
  Terray, E.A. et al. (1996) Estimates of kinetic energy dissipation under breaking
      waves. J. Phys. Oceanogr., 26, 792-807.

Input:  data/r_weather.csv (time in UTC)
Output: output/epsilon_from_weather.csv, output/epsilon_from_weather.png
"""
#%% Imports and constants
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

RHO_AIR = 1.225             # air density [kg/m^3]
RHO_WATER = 1025.0          # sea water density [kg/m^3] (coastal Skagerrak estimate)
NU_WATER = 1.3e-6           # kinematic viscosity of sea water at ~15 C [m^2/s]
KAPPA = 0.4                 # von Karman constant
WAVE_LAYER_FRACTION = 0.6   # depth of the wave-affected layer = 0.6 * Hs (Terray et al. 1996)
ENHANCEMENT_FACTOR = 100.0  # enhancement of dissipation by breaking waves over the law of the wall
                            # (1-2 orders of magnitude: Agrawal et al. 1992; Craig & Banner 1994; Terray et al. 1996)
LOG_LAW_DEPTHS = (5, 10, 20)  # depths for the law-of-the-wall estimate [m]

#%% Functions

def drag_coefficient(u10):
    """Neutral 10 m drag coefficient of Large & Pond (1981)."""
    u10 = np.asarray(u10, dtype=float)
    return np.where(u10 < 11.0, 1.2e-3, (0.49 + 0.065 * u10) * 1e-3)


def water_side_friction_velocity(u10):
    """Friction velocity in the water [m/s] from the 10 m wind speed [m/s]."""
    tau = RHO_AIR * drag_coefficient(u10) * u10 ** 2  # wind stress [Pa]
    u_star_air = np.sqrt(tau / RHO_AIR)
    return u_star_air * np.sqrt(RHO_AIR / RHO_WATER)  # stress continuity across the surface


def epsilon_wave_layer(u10, hs):
    """Dissipation rate within the wave-affected surface layer [m^2/s^3]."""
    u_star_w = water_side_friction_velocity(u10)
    d = np.clip(WAVE_LAYER_FRACTION * np.asarray(hs, dtype=float), 0.05, None)
    return ENHANCEMENT_FACTOR * u_star_w ** 3 / (KAPPA * d)


def epsilon_log_law(u10, z):
    """Law-of-the-wall dissipation rate (no wave enhancement) at depth z [m], in m^2/s^3."""
    u_star_w = water_side_friction_velocity(u10)
    z = np.clip(np.asarray(z, dtype=float), 0.05, None)
    return u_star_w ** 3 / (KAPPA * z)

#%% Load the weather data and estimate epsilon
weather = pd.read_csv(DATA_DIR / "r_weather.csv", sep=";", decimal=",")
weather["date"] = pd.to_datetime(weather["date"], format="%Y.%m.%d")
weather["time_utc"] = pd.to_timedelta(weather["time_utc"].astype(str))
weather["datetime"] = weather["date"] + weather["time_utc"] + pd.Timedelta(hours=2)  # UTC -> CEST

u10 = weather["wind_speed_m_s"].to_numpy()
hs = weather["OS2_sign_wave_m"].to_numpy()

weather["u_star_water_m_s"] = water_side_friction_velocity(u10)
weather["epsilon_wave_layer_m2_s3"] = epsilon_wave_layer(u10, hs)
for z in LOG_LAW_DEPTHS:
    weather[f"epsilon_loglaw_z{z}m_m2_s3"] = epsilon_log_law(u10, z)

eps_wave = weather["epsilon_wave_layer_m2_s3"]
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

print("=== epsilon in the wave-affected surface layer (m^2/s^3 = W/kg) ===")
print(eps_wave.describe(percentiles=percentiles))

print("\n=== same, converted to mm^2/s^3 (units of the simulations) ===")
print((eps_wave * 1e6).describe(percentiles=percentiles))

eta_mm = (NU_WATER ** 3 / eps_wave.clip(lower=1e-12)) ** 0.25 * 1e3  # Kolmogorov length scale [mm]
print("\n=== derived Kolmogorov length scale eta [mm] ===")
print(eta_mm.describe(percentiles=percentiles))

OUTPUT_DIR.mkdir(exist_ok=True)
weather.to_csv(OUTPUT_DIR / "epsilon_from_weather.csv", index=False)

#%% Plot wind, waves and epsilon over time
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(weather["datetime"], u10, color="steelblue", linewidth=0.8)
axes[0].set_ylabel("Wind speed [m/s]")
axes[0].grid(alpha=0.4)

axes[1].plot(weather["datetime"], hs, color="black", linewidth=0.8)
axes[1].set_ylabel("Sig. wave height [m]")
axes[1].grid(alpha=0.4)

axes[2].semilogy(weather["datetime"], eps_wave, color="darkorange", linewidth=0.8, label="wave layer")
for z in LOG_LAW_DEPTHS:
    axes[2].semilogy(weather["datetime"], weather[f"epsilon_loglaw_z{z}m_m2_s3"], linewidth=0.6, alpha=0.6, label=f"log-law z={z} m")
axes[2].set_ylabel("epsilon [W/kg]")
axes[2].legend(fontsize=7)
axes[2].grid(alpha=0.4, which="both")
axes[2].set_xlabel("Date")

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "epsilon_from_weather.png", dpi=150)
print(f"\nSaved epsilon_from_weather.csv and epsilon_from_weather.png in {OUTPUT_DIR}")
