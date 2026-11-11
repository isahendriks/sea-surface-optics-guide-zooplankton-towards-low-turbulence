# Sea surface optics guide zooplankton towards low turbulence

Analysis and simulation code accompanying the manuscript. The repository contains four
independent parts, each in its own folder:

| Folder | Script | What it does |
|---|---|---|
| `ADCP data analysis` | `ADCP_data_analysis_and_plot.py` | Diel vertical migration in the field from ADCP backscatter: ascent rate and migration magnitude per day, regressed on wave height and wind speed; makes the ADCP figure |
| | `epsilon_from_weather.py` | Turbulent dissipation rate at the field site estimated from wind speed and wave height |
| `Simulated light regimes` | `Arduino Control Scripts/{calm,moderate,rough}/*.ino` | Arduino sketches that drive the 32 x 32 LED matrix for the three light regimes |
| | `Spectral Irradiance/spectral_irradiance.py` | Spectral irradiance of the three light regimes compared with the lunar spectrum |
| `Trajectory Analysis` | `load_and_save.py` | Loads and cleans the tracked trajectories, saves them, and tests the effect of the light regime on vertical velocity and swimming direction (ANOVA + Tukey's HSD over tanks) |
| | `plot_full_per_species.py` | Figures of swimming direction and vertical velocity per species and light regime, and of the controls, with the statistics shown in them |
| | `post_processing_functions.py` | Functions used by `load_and_save.py` |
| `Kinematic simulations` | `coupled_simulation.py` | Simulates plankton swimming with the measured behaviour in kinematic turbulence over a range of dissipation rates |
| | `turbulence_model.py`, `swimming_behavior.py` | The turbulence model and the empirical swimming behaviour (functions only) |
| | `validate_turbulence_model.py` | Checks the turbulence model (incompressibility, -5/3 spectrum, isotropy, dissipation rate) |
| | `demo_swimming_behavior.py` | Behaviour-only simulation and animation, to check the swimming behaviour |

## Installation

The code was run with Python 3.12 and the package versions in `requirements.txt`:

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The scripts are divided into cells (`#%%`), so they can be run as a whole or cell by cell in
VS Code or Spyder. Run every script from its own folder.

## Data

The data are not part of this repository. Put them in a `data/` folder next to the scripts
that use them:

| Folder | Expected input |
|---|---|
| `ADCP data analysis/data/` | `Trubadur_ADCP_beam5_m30sec.mat`, `r_weather.csv`, `w_truba.csv` |
| `Simulated light regimes/Spectral Irradiance/data/` | `still.txt`, `breeze.txt`, `stormy.txt`, `black.txt`, `calibration.txt`, `LIME_lunar_irradiance.csv` |
| `Trajectory Analysis/data/` | `<species>/tracking_results/<regime><tank>.xml` (TrackMate tracks) |
| `Kinematic simulations/data/` | `<species>/traj_<regime>_scaled.pkl` (written by `load_and_save.py`) |

The species are `Artemia_0805`, `BarnacleLarvae_2102` and `Cladocerans_2605`; the light
regimes are `still` (calm), `breeze` (moderate), `stormy` (rough) and `nothing` (control,
no light). Figures and other results are written to an `output/` folder next to the script
(the simulation results to the `Kinematic simulations` folder).

## Order of the trajectory analysis and simulations

1. `Trajectory Analysis/load_and_save.py` cleans the tracks and writes `traj_<regime>_scaled.pkl`.
2. `Trajectory Analysis/plot_full_per_species.py` reads these files and makes the figures.
3. Copy the `traj_*_scaled.pkl` files to `Kinematic simulations/data/<species>/` and run
   `Kinematic simulations/coupled_simulation.py`.

The simulations and the plotted samples of frames use a fixed random seed, so they can be
reproduced exactly.

## License

MIT, see `LICENSE`.
