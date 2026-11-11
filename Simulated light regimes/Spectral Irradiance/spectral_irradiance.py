"""
Spectral irradiance of the three simulated light regimes (LED matrix), compared with the
lunar spectrum.

The spectra of the LED matrix were measured with an Ocean Optics QE Pro spectrometer
(3 s integration time) for the calm, moderate and rough light regime and with the LED
matrix off (background). Counts are converted to spectral irradiance with the
spectrometer's calibration file, integrated over the wavelength range relevant to
zooplankton vision (350-600 nm), and plotted together with the LIME lunar irradiance.

Input (in data/)
  still.txt, breeze.txt, stormy.txt   spectra of the calm, moderate and rough regime
  black.txt                           background spectrum (LED matrix off)
  calibration.txt                     spectrometer calibration
  LIME_lunar_irradiance.csv           lunar spectral irradiance from the LIME toolbox
Output (in output/)
  spectral_intensity.png, spectral_intensity.svg
"""
#%% Imports and settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

WL_MIN, WL_MAX = 300, 750                # plotted wavelength range [nm]
WL_MIN_ZOOPLANKTON, WL_MAX_ZOOPLANKTON = 350, 600  # range relevant to zooplankton vision [nm]
INTEGRATION_TIME = 3                     # spectrometer integration time [s]
AREA = 0.1 * 0.1                         # visible area of the LED matrix [m^2]

#%% Functions

def read_spectrum(filename):
    """Wavelength [nm] and counts of a spectrometer export file (data start after 13 header lines)."""
    lines = pd.read_csv(filename, header=None).values[13:1058]
    data = np.array([list(map(float, row[0].split())) for row in lines])
    return data[:, 0], data[:, 1]


def read_calibration(filename):
    """Columns of the spectrometer calibration file (data start after 8 header lines)."""
    lines = pd.read_csv(filename, header=None).values[8:1053]
    return np.array([list(map(float, row[0].split())) for row in lines])


def integrate_range(E, wl, wl_min, wl_max):
    """Integral of the spectrum E over wl_min <= wl <= wl_max."""
    in_range = (wl >= wl_min) & (wl <= wl_max)
    return np.trapezoid(E[in_range], x=wl[in_range])

#%% Load the spectra
wl, _ = read_spectrum(DATA_DIR / "still.txt")
imin = np.abs(wl - WL_MIN).argmin()
imax = np.abs(wl - WL_MAX).argmin()
wl = wl[imin:imax]

counts = {regime: read_spectrum(DATA_DIR / f"{regime}.txt")[1][imin:imax]
          for regime in ["still", "breeze", "stormy", "black"]}
cal = read_calibration(DATA_DIR / "calibration.txt")[imin:imax, 0]

lime = pd.read_csv(DATA_DIR / "LIME_lunar_irradiance.csv", skiprows=22)
lime.columns = ["Wavelength", "Irradiance", "Uncertainty"]
lime = lime[(lime["Wavelength"] >= WL_MIN) & (lime["Wavelength"] <= WL_MAX)]
wl_lunar = lime["Wavelength"].values
E_lunar = lime["Irradiance"].values * 1e6  # W m^-2 nm^-1 -> µW m^-2 nm^-1

#%% Spectral irradiance and the irradiance in the zooplankton range
# Subtract the background and convert counts to irradiance (µW m^-2 nm^-1)
regimes = {"still": "Calm", "breeze": "Moderate", "stormy": "Rough"}
E = {regime: (counts[regime] - counts["black"]) * cal / INTEGRATION_TIME * AREA for regime in regimes}

I_zooplankton = {regime: integrate_range(E[regime], wl, WL_MIN_ZOOPLANKTON, WL_MAX_ZOOPLANKTON) for regime in regimes}
I_zooplankton_lunar = integrate_range(E_lunar, wl_lunar, WL_MIN_ZOOPLANKTON, WL_MAX_ZOOPLANKTON)

print(f"Total light intensity in the zooplankton relevant range ({WL_MIN_ZOOPLANKTON}-{WL_MAX_ZOOPLANKTON} nm):\n",
      f"Calm: {I_zooplankton['still']*1e-3:.2f} mW/m2\n",
      f"Moderate: {I_zooplankton['breeze']*1e-3:.2f} mW/m2\n",
      f"Rough: {I_zooplankton['stormy']*1e-3:.2f} mW/m2\n",
      f"Lunar: {I_zooplankton_lunar*1e-3:.2f} mW/m2")

#%% Plot
colors = sns.color_palette("Blues", 4)[1:4]

plt.figure(figsize=(7, 4))
for (regime, label), color in zip(regimes.items(), colors):
    plt.plot(wl, E[regime], color=color, linewidth=1, alpha=0.8, label=label)
plt.plot(wl_lunar, E_lunar, color='gray', linewidth=1, alpha=0.8, label="Lunar Spectrum")
plt.axvspan(WL_MIN_ZOOPLANKTON, WL_MAX_ZOOPLANKTON, color='grey', alpha=0.1, label="Zooplankton relevant range")

plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel(r"Spectral Irradiance ($\mu$W m$^{-2}$ nm$^{-1}$)", fontsize=12)
plt.xlim(WL_MIN, WL_MAX)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=10)
plt.legend(loc="upper left", fontsize=10)
plt.tight_layout()

OUTPUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_DIR / "spectral_intensity.png", dpi=1200, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "spectral_intensity.svg", bbox_inches="tight")
plt.show()
