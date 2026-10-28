#%%
#%matplotlib widget
#%matplotlib inline 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Define the functions
def read_data(filename):
    # Takes filename and returns wavelength vector (wl), Intensity vector (I) and title which contains light regime
    dat_pd = pd.read_csv(filename, header=None)
    dat_np = np.array([list(map(float, row[0].split())) for row in dat_pd.values[13:1058]])
    wl = dat_np[:, 0]
    I = dat_np[:, 1]
    title = dat_pd.iloc[12, 0].split()[-1] + ' ' + dat_pd.iloc[13, 0].split()[-1]
    return [wl, I, title]

def E_to_umol(E, wl):
    """
    Convert energy flux from µJ s^-1 m^-2 to photon flux in µmol s^-1 m^-2.

    Parameters:
    E: Energy flux in µJ s^-1 m^-2 (array or scalar).
    wl: Wavelength in nm (array or scalar).

    Returns:
    float: Photon flux in µmol s^-1 m^-2.
    """
    # Constants
    h = 6.626e-34  # Planck's constant, J*s
    c = 3.00e8      # Speed of light, m/s
    avogadro = 6.022e23  # Avogadro's number, mol^-1

    # Calculate energy per photon (J/photon)
    E_photon = h * c / (wl * 1e-9)

    # Calculate photon flux (photons/s/m^2)
    flux = (E*1e-6) / E_photon

    # Convert photon flux to µmol (photons -> µmol)
    flux_umol = flux / avogadro * 1e6

    return flux_umol

def E_to_lux(E, wl, v_lambda):
    """
    Convert energy flux to illuminance in lux.

    Parameters:
    E: Energy flux in µJ s^-1 m^-2 (array or scalar).
    wl: Wavelength bins [nm].
    v_lambda: Photopic luminous efficiency function.
    """

    # lux
    lux = 683.0 * np.trapz(E * 1e-6 * v_lambda, wl)

    return lux

# Load CIE photopic luminous efficiency function
cie = np.loadtxt("CIE_sle_photopic.csv", delimiter=",", skiprows=1)

cie_wavelengths = cie[:, 0]
cie_v_lambda = cie[:, 1]

# Load LIME moon spectrum
lime = pd.read_csv("LIME_lunar_irradiance.csv", skiprows=22)

lime.columns = ["Wavelength", "Irradiance", "Uncertainty"]

#%% Read data test
wl= read_data('LEDmatrix_spectraldata/still.txt')[0]
wlmin = 300 #min(wl)
wlmax = 750 #max(wl)
imin = np.abs(wl-wlmin).argmin()
imax = np.abs(wl-wlmax).argmin()
wl = wl[imin:imax]

bin_width = wl[1] - wl[0]  # assuming uniform bin width

v_lambda = np.interp(
    wl,
    cie_wavelengths,
    cie_v_lambda,
    left=0,
    right=0,
)

# Load lunar spectrum
lime_vis = lime[(lime["Wavelength"] >= wlmin) & (lime["Wavelength"] <= wlmax)]

wl_lunar = lime_vis["Wavelength"].values
E_lunar = lime_vis["Irradiance"].values * 1e6 # convert from W m^-2 nm^-1 to µW m^-2 nm^-1
uncertainty_lunar = lime_vis["Uncertainty"].values * 1e6 # convert from W m^-2 nm^-1 to µW m^-2 nm^-1

# read measurements
still = read_data('LEDmatrix_spectraldata/still.txt')[1][imin:imax]
stormy = read_data('LEDmatrix_spectraldata/stormy.txt')[1][imin:imax]
breeze = read_data('LEDmatrix_spectraldata/breeze.txt')[1][imin:imax]
black = read_data('LEDmatrix_spectraldata/black.txt')[1][imin:imax]

# read calibration file
cal_pd = pd.read_csv('LEDmatrix_spectraldata/calibration.txt', header=None)
cal_np = np.array([list(map(float, row[0].split())) for row in cal_pd.values[8:1053]])
cal = cal_np[imin:imax,0]

# Process measurements (subtract bkg (black), use calibration file, divide by 3 since measurement took 3 seconds)
A = 0.1 * 0.1 # visible area of the LED matrix [m^2]

still_counts = (still - black)
stormy_counts = (stormy - black)
breeze_counts = (breeze - black)

colors = sns.color_palette("Blues", 4)[1:]

# %% Plot spectra
plt.plot(wl, still_counts, label = 'calm', color=colors[0])
plt.plot(wl, breeze_counts, label = 'moderate', color=colors[1])
plt.plot(wl, stormy_counts, label = 'rough', color=colors[2])
plt.xlabel("Wavelength [nm]", fontsize=18)
plt.ylabel("Intensity [$\mu$J]", fontsize=18)
plt.grid(True)
plt.legend()
plt.tick_params(axis='both', labelsize=12)
plt.xlim(wlmin, wlmax)

# %% Calculate total light intensity

# convert counts to energy flux (µJ s^-1 m^-2)/(µW m^-2) 
E_still = still_counts * cal / 3 * A
E_stormy = stormy_counts * cal / 3 * A 
E_breeze = breeze_counts * cal / 3 * A 

# convert to PPFD (µmol m^-2 s^-1)
still_spec_PPDF = E_to_umol(E_still, wl)
stormy_spec_PPDF = E_to_umol(E_stormy, wl)
breeze_spec_PPDF = E_to_umol(E_breeze, wl)

# calculate illuminance (lux) using the photopic luminous efficiency function
I_still_lux = E_to_lux(E_still, wl, v_lambda)
I_breeze_lux = E_to_lux(E_breeze, wl, v_lambda)
I_stormy_lux = E_to_lux(E_stormy, wl, v_lambda)

# Calculate total light intensity by integrating the PPFD spectra over the wavelength range
I_still_PPDF = np.trapz(still_spec_PPDF, x=wl)
I_breeze_PPDF = np.trapz(breeze_spec_PPDF, x=wl)
I_stormy_PPDF = np.trapz(stormy_spec_PPDF, x=wl)

# Calculate total light intensity by integrating the energy flux spectra over the wavelength range
wl_min_zooplanton = 350
wl_max_zooplanton = 600

# Calculate intensity for the zooplankton relevant range
I_still_E = np.trapz(E_still[(wl >= wl_min_zooplanton) & (wl <= wl_max_zooplanton)], x=wl[(wl >= wl_min_zooplanton) & (wl <= wl_max_zooplanton)])
I_breeze_E = np.trapz(E_breeze[(wl >= wl_min_zooplanton) & (wl <= wl_max_zooplanton)], x=wl[(wl >= wl_min_zooplanton) & (wl <= wl_max_zooplanton)])
I_stormy_E = np.trapz(E_stormy[(wl >= wl_min_zooplanton) & (wl <= wl_max_zooplanton)], x=wl[(wl >= wl_min_zooplanton) & (wl <= wl_max_zooplanton)])
I_lunar_E = np.trapz(E_lunar[(wl_lunar >= wl_min_zooplanton) & (wl_lunar <= wl_max_zooplanton)], x=wl_lunar[(wl_lunar >= wl_min_zooplanton) & (wl_lunar <= wl_max_zooplanton)])

# print(f"I_still: {I_still_lux} lux, I_breeze: {I_breeze_lux} lux, I_stormy: {I_stormy_lux} lux")
# print(f"I_still_PPDF: {I_still_PPDF} µmol m^-2 s^-1, I_breeze_PPDF: {I_breeze_PPDF} µmol m^-2 s^-1, I_stormy_PPDF: {I_stormy_PPDF} µmol m^-2 s^-1")

# plt.figure(figsize=(7, 4))

# plt.plot(wl, still_spec_PPDF*1e3, color=colors[0], linewidth=1, alpha=0.8, label=fr"Calm ({I_still_lux:.2f} lux)")
# plt.plot(wl, breeze_spec_PPDF*1e3, color=colors[1], linewidth=1, alpha=0.8, label=fr"Moderate ({I_breeze_lux:.2f} lux)")
# plt.plot(wl, stormy_spec_PPDF*1e3, color=colors[2], linewidth=1, alpha=0.8, label=fr"Rough ({I_stormy_lux:.2f} lux)")

# plt.xlabel("Wavelength (nm)", fontsize=12)
# plt.ylabel(r"Photon Flux Density  (nmol m$^{-2}$ s$^{-1}$ nm$^{-1}$)", fontsize=12)
# plt.xlim(wlmin, wlmax)
# plt.grid(True, alpha=0.3)
# plt.tick_params(axis='both', labelsize=10)
# plt.legend(loc="upper left", fontsize=10)
# plt.tight_layout()
# plt.savefig("LEDmatrix_spectraldata/spectral_intensity.png", dpi=600, bbox_inches="tight")
# plt.savefig("LEDmatrix_spectraldata/spectral_intensity.svg", bbox_inches="tight")
# plt.show()


plt.figure(figsize=(7, 4))

plt.plot(wl, E_still, color=colors[0], linewidth=1, alpha=0.8, label=fr"Calm")
plt.plot(wl, E_breeze, color=colors[1], linewidth=1, alpha=0.8, label=fr"Moderate")
plt.plot(wl, E_stormy, color=colors[2], linewidth=1, alpha=0.8, label=fr"Rough")

plt.plot(wl_lunar, E_lunar, color='gray', linewidth=1, alpha=0.8, label=fr"Lunar Spectrum")
# plt.fill_between(wl_lunar, E_lunar - uncertainty_lunar, E_lunar + uncertainty_lunar, color='gray', alpha=0.2, label=fr"Uncertainty")

# plot rectangle for zooplankton relevant range
plt.axvspan(wl_min_zooplanton, wl_max_zooplanton, color='grey', alpha=0.1, label=fr"Zooplankton relevant range")

plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel(r"Spectral Irradiance ($\mu$W m$^{-2}$ nm$^{-1}$)", fontsize=12)
plt.xlim(wlmin, wlmax)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=10)
plt.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig("LEDmatrix_spectraldata/spectral_intensity.png", dpi=1200, bbox_inches="tight")
plt.savefig("LEDmatrix_spectraldata/spectral_intensity.svg", bbox_inches="tight")
plt.show()

print(f"Total light intensity in the zooplankton relevant range ({wl_min_zooplanton}-{wl_max_zooplanton} nm):\n",
      f"Calm: {I_still_E*1e-3:.2f} mW/m2\n",
      f"Moderate: {I_breeze_E*1e-3:.2f} mW/m2\n",
      f"Rough: {I_stormy_E*1e-3:.2f} mW/m2\n",
      f"Lunar: {I_lunar_E*1e-3:.2f} mW/m2")

# %%
