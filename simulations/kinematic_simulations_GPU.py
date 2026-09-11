# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from IPython.display import Image

def velocity_at(positions, t):
    # positions: (M, 2) array of (x, y); t: time in seconds
    phase = positions @ k_n.T + omega_n[None, :] * t   # (M, N)
    u = np.cos(phase) @ A_n + np.sin(phase) @ B_n        # (M, 2)
    return u[:, 0], u[:, 1]

# %% Set parameters for simulation

### Simulation parameters
tank_size = 90
grid_size = tank_size + 1  # 1 mm grid spacing, only used for plotting/visualization
fps = 10 # temporal resolution [fps]
t_simulation = 60 # total length of simulation [s]
N_plankton = 100 # Number of plankters

### Parameters for velocity field (in SI units * 1e6 to convert to mm)
epsilon = 1e-4 * 1e6 # Dissipation rate , ranges from 1e-4 (rough sea) to 1e-14 (calm sea) - *1e-6 to convert to mm
nu = 1e-6 * 1e6 # Kinematic viscosity of sea water depends on salinity and temperature. Ranges from 1e-6 to 1.8e-6, times 1e-6 to convert to mm
N = 50 # Total number of wave numbers sampled
L = tank_size / 5 # Maximum length scale of turbulence [mm]
eta = (nu**3 / epsilon) ** (1/4)   # Define eta (Kolmogorov length scale)
dx_physics = eta/2 # physical/turbulence modelling resolution [mm] - smallest eddy the Fourier modes can represent should be <= eta/2

### Plankton parameters
set_turbulence = False

### Starting conditions of plankton (based on real swimmers)
velocity = 5 #2.4456 # mean upward velocity [mm/s]
angle = 154.12 # in degrees (plus or minus around the target angle 90 degrees)
Dt = 2.1989e-1 # random swimming component from control [mm^2/s]
reorientation_rate = 2.0 # mean number of heading-reorientation events per second [s^-1] 

### Video parameters
stride = 6 # number of grid points to skip when plotting the quiver plot (for clarity)
step = 1 # number of frames to skip when creating the video (for speed)

### Create grid for the simulation
x = np.linspace(0, tank_size, grid_size)
y = np.linspace(0, tank_size, grid_size)
X, Y = np.meshgrid(x, y)

### Calculate timesteps for the animation
timesteps = t_simulation * fps
delta_t = 1/fps
frame_indices = range(0, timesteps, step)

### Set up the meshgrid for plotting the velocity field
Xs = X[::stride, ::stride]
Ys = Y[::stride, ::stride]

# %% ### 1. Setting up the Kolmogorov energy speectrum

# Calculate E0 (amplitude of energy spectrum)
E0 = 1.5 * epsilon**(2/3) * (1 - (eta/L)**(4/3))**(-1) 

# Generate wavevectors k_n using the geometric series in ascending order
k_min = 2 * np.pi / L
k_max_numerical = np.pi / dx_physics       # Nyquist limit set by the modelling resolution (independent of the plotting grid)
k_max_resolved = min(2 * np.pi / eta, k_max_numerical)   # don't sample modes finer than dx_physics or the physical dissipation scale (eta) can resolve

k_values = k_min * (k_max_resolved / k_min) ** (np.arange(N) / (N - 1))

# Energy spectrum
E_k = E0 * (k_values ** (-5 / 3))

# Delta k values
delta_k0 = (k_values[1] - k_values[0]) / 2
delta_kN = (k_values[-1] - k_values[-2]) / 2
delta_k_ = (k_values[2:] - k_values[:-2]) / 2
delta_k = np.concatenate(([delta_k0], delta_k_, [delta_kN]))

# Plot the Kolmogorov energy specutrm in the intertial subrange
plt.figure(figsize=(10, 6))
plt.loglog(k_values, E_k, marker='o', label=r'$E(k) = E_0 \cdot k^{-5/3}$', color="darkviolet")
plt.xlabel('Wave number $k$')
plt.ylabel('Energy spectrum $E(k)$')
plt.title("Kolmogorov Energy Spectrum in the Inertial Range")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# %% ### 2. **Kinematic simulation of velocity fields**

# Amplitudes of the Fourier modes
a_n = b_n = np.sqrt(2 * E_k * delta_k)

# Temporal frequencies
omega_n = 0.4 * np.sqrt((k_values ** 3) * E_k)

# Random phases for amplitudes and wavevectors
angles = 2 * np.pi * np.random.rand(N)

# Define A_n, B_n, and k_n according to the incompressibility constraints (Shape: (N, 2)), where N = number of Fourier modes
A_n = np.array([a_n * np.cos(angles), -a_n * np.sin(angles)]).T
B_n = np.array([-b_n * np.cos(angles), b_n * np.sin(angles)]).T
k_n = np.array([k_values * np.sin(angles), k_values * np.cos(angles)]).T

# Define the spatial grid (x, y)
positions = np.stack([X.ravel(), Y.ravel()], axis=-1)  # Flattened grid positions for efficiency (Shape: (grid_size * grid_size, 2))

# Initialize an array to store the velocity field at each time step
velocity_field = np.zeros((timesteps, grid_size, grid_size, 2))

# Compute the velocity field for each time step
for t in range(timesteps):
    print(f"\r Simulating velocity field for timestep {t}/{timesteps} ...", end='\r', flush=True)
    # Initialize the velocity at each point to zero
    u = np.zeros((grid_size * grid_size, 2))

    # Loop over each Fourier mode
    for n in range(N):
        # Compute the phase shift for each mode
        phase = np.dot(positions, k_n[n]) + omega_n[n] * (t/fps)

        # Reshape A_n[n] and B_n[n] to be (1, 2) for broadcasting (for multiplication to all points in the grid)
        A_n_n = A_n[n].reshape(1, 2)
        B_n_n = B_n[n].reshape(1, 2)

        # Add the contribution of this mode to the velocity field
        u += (A_n_n * np.cos(phase)[:, None] + B_n_n * np.sin(phase)[:, None])


    # Reshape and store the velocity field for this time step
    velocity_field[t, ..., 0] = u[:, 0].reshape(grid_size, grid_size)  # x-component
    velocity_field[t, ..., 1] = u[:, 1].reshape(grid_size, grid_size)  # y-component

print("finished")
# velocity_field now contains the velocity vectors at each point on the grid for each time step


# %% # Plot quiver plot of velocity field at single time step

# Select a time step
t = 1  # Choose a specific time step (e.g., t=0 for the initial field)

Us = velocity_field[t, ::stride, ::stride, 0]  # x-component of the velocity
Vs = velocity_field[t, ::stride, ::stride, 1]  # y-component of the velocity

# Plot the velocity field using quiver
plt.figure(figsize=(12, 12))
plt.quiver(Xs, Ys, Us, Vs, scale=500, pivot='mid', color='blue')
plt.title(f"Velocity Field at Time Step {t}")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.grid()
plt.show()

U = velocity_field[t, ..., 0]  # x-component of the velocity
V = velocity_field[t, ..., 1]  # y-component of the velocity

plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, U, V, density=1.5, color=np.sqrt(U**2 + V**2), cmap='viridis')
plt.colorbar(label="Velocity Magnitude")
plt.title(f"Streamline Plot of Velocity Field at Time Step {t}")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

# %% # Plot quiver plot of velocity field as gif animation over time
# Set up the figure and axis for the quiver plot
fig, ax = plt.subplots(figsize=(8, 8))

### Plot quiver plot simulation
U0, V0 = velocity_field[0, ::stride, ::stride, 0], velocity_field[0, ::stride, ::stride, 1]

fig, ax = plt.subplots(figsize=(8, 8))
quiver = ax.quiver(Xs, Ys, U0, V0, scale=500, pivot='mid', color='blue')
title = ax.set_title("Velocity Field at T = 0 s")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()

def update_quiver(frame):
    print(f"\rProcessing frame {frame}/{timesteps} ...", end='', flush=True)
    U = velocity_field[frame, ::stride, ::stride, 0]
    V = velocity_field[frame, ::stride, ::stride, 1]
    quiver.set_UVC(U, V)
    title.set_text(f"Velocity Field at T = {frame/fps:.2f} s")
    return quiver, title

anim = FuncAnimation(fig, update_quiver, frames=frame_indices, interval=50, blit=True)

gif_path = "velocityfield_quiver.gif"
anim.save(gif_path, writer=PillowWriter(fps=fps/step))
plt.close(fig)
Image(filename=gif_path)

# %% Verify interpolation of the velocity field using RegularGridInterpolator
# Select a timestep for testing
timestep = 0
u_interpolator = RegularGridInterpolator((x, y), velocity_field[timestep, ..., 0])  # x-component of velocity
v_interpolator = RegularGridInterpolator((x, y), velocity_field[timestep, ..., 1])  # y-component of velocity

# Check interpolation at a specific grid point by using coordinates, not indices
index = 3  # Example grid index (4th row and 4th column in zero-based index)
point = [x[index], y[index]]  # Use actual physical coordinates

# Interpolated velocity at this physical point
u_point = u_interpolator(point)
v_point = v_interpolator(point)

# Compare with the actual velocity at the closest grid point using indices
u_grid = velocity_field[timestep, index, index, 0]
v_grid = velocity_field[timestep, index, index, 1]

if not (np.isclose(u_point, u_grid) and np.isclose(v_point, v_grid)):
    print(f"Interpolation mismatch at point {point}: interpolated=({u_point}, {v_point}), grid=({u_grid}, {v_grid})")
else:
    print("Interpolation check passed.")


# %% Extract parameters from experimental data to set plankton parameters
### Read the data from experiments
# measurement = "Cladocerans_2605"
# path_to_still = r"R:\\LU24A1047-PLS\\TrackingData\\" + measurement + r"\\trajectories_scaled\\traj_still_scaled.pkl"
# path_to_control = r"R:\\LU24A1047-PLS\\TrackingData\\" + measurement + r"\\trajectories_scaled\\traj_nothing_scaled.pkl"

# df_still = pd.read_pickle(path_to_still)
# df_control = pd.read_pickle(path_to_control)


# df_still = df_still.dropna(subset=["speed", "orientation"])
# df_control = df_control.dropna(subset=["speed", "orientation"])

# # velocity: mean directed swimming speed from the stimulus condition [mm/s]
# velocity_measurement = df_still["speed"].mean()

# # response_angle: half-width of a uniform heading spread around target_angle (90 deg)
# # that reproduces the measured resultant length R of the still condition
# target_angle = np.pi / 2
# R_still = np.sqrt(np.mean(np.cos(df_still["orientation"])) ** 2 + np.mean(np.sin(df_still["orientation"])) ** 2)
# response_angle_measurement = brentq(lambda a: np.sinc(a / np.pi) - R_still, 1e-6, np.pi - 1e-6)

# # Dt: random swimming component from the no-stimulus control condition [mm^2/s]
# # matches mean-squared displacement over one frame interval (delta_t) to the model's 2*Dt*delta_t
# delta_t_meas = 0.1  # video frame interval [s]
# Dt_measurement = np.mean(df_control["speed"] ** 2) * delta_t_meas / 4

# # reorientation_rate: mean number of heading-randomization events per second [1/s], estimated
# # from the decay of the heading autocorrelation function over time lag tau.
# # Model: the heading holds steady and resets to a fresh iid draw at Poisson rate lambda
# # (a renewal process - same model used for `phi` in the plankton simulation loop). For such
# # a process, C(tau) = E[cos(phi(t+tau) - phi(t))] = R^2 + (1 - R^2) * exp(-lambda * tau),
# # where R is the resultant length of the heading distribution (R_still, computed above).
# # Fitting the exponential decay of the measured C(tau) therefore gives lambda directly, in
# # physical units of 1/s - independent of the video frame rate (unlike a raw per-frame turning
# # rate), which is what makes it usable as `reorientation_rate` at any simulation fps.
# max_lag_frames = 30
# lags = np.arange(1, max_lag_frames + 1)
# cos_dphi_by_lag = {lag: [] for lag in lags}

# for _, track in df_still.groupby("particle"):
#     t_to_phi = dict(zip(track["t"], track["orientation"]))
#     for t0, phi0 in t_to_phi.items():
#         for lag in lags:
#             phi1 = t_to_phi.get(t0 + lag)  # skips gaps: only exact lag-frame-apart pairs are used
#             if phi1 is not None:
#                 cos_dphi_by_lag[lag].append(np.cos(phi1 - phi0))

# C_tau = np.array([np.mean(cos_dphi_by_lag[lag]) if cos_dphi_by_lag[lag] else np.nan for lag in lags])
# tau_seconds = lags * delta_t_meas

# # Only the part of the decay still above the R_still^2 asymptote is usable (log needs a positive argument)
# valid = ~np.isnan(C_tau) & (C_tau > R_still ** 2)
# slope, _ = np.polyfit(tau_seconds[valid], np.log(C_tau[valid] - R_still ** 2), 1)
# reorientation_rate_measurement = -slope

# print(f"velocity = {velocity_measurement:.4f} mm/s")
# print(f"response_angle = {np.degrees(response_angle_measurement):.2f} deg")
# print(f"Dt = {Dt_measurement:.4e} mm^2/s")
# print(f"reorientation_rate = {reorientation_rate_measurement:.4f} 1/s")

# 
# Artemia data: 
# velocity = 2.4456 mm/s
# response_angle = 154.12 deg
# Dt = 2.1989e-01 mm^2/s
# 
# Barnacle Larvae:
# velocity = 1.1411 mm/s
# response_angle = 179.12 deg
# Dt = 9.2954e-02 mm^2/s
# 
# Cladocerans: 
# velocity = 2.0006 mm/s
# response_angle = 152.66 deg
# Dt = 1.5735e-01 mm^2/s
# 
# 

# %% Set plankton parameters based on experimental data (Artemia)

# %% Coupling the velocity field with the plankton swimming and random noise
target_angle = np.pi / 2
response_angle = angle * (np.pi / 180)

# Define velocity interpolators
x_grid = np.linspace(0, tank_size, grid_size)
y_grid = np.linspace(0, tank_size, grid_size)

# Store the positions
stored_positions = np.zeros((N_plankton, 2, timesteps))

# Intialize the plankton positions and orientations
# x_pos = np.random.rand(n_planktons) * L
# y_pos = np.random.rand(n_planktons) * L
x_pos = np.random.uniform(0, tank_size, N_plankton)
y_pos = np.random.uniform(0, tank_size, N_plankton)
phi = target_angle + response_angle * (2 * np.random.rand(N_plankton) - 1)

if reorientation_rate * delta_t > 0.5:
    print(f"Warning: reorientation_rate * delta_t = {reorientation_rate * delta_t:.2f} is not << 1; "
          f"increase fps so individual reorientation events are resolved.")

for t in range(timesteps):
    print(f"\r Simulating plankton movement for timestep {t}/{timesteps} ...", end='', flush=True)

    # Reorient a subset of plankton this step (Poisson-process approximation at rate reorientation_rate
    # [1/s]); everyone else keeps their previous heading. This ties the reorientation correlation time to
    # a physical rate instead of the simulation timestep, so results converge as fps increases.
    reorient = np.random.rand(N_plankton) < reorientation_rate * delta_t
    phi[reorient] = target_angle + response_angle * (2 * np.random.rand(int(reorient.sum())) - 1)

    # turbulence velocities
    positions = np.stack([x_pos, y_pos], axis=-1)
    vx_turb, vy_turb = velocity_at(positions, t/fps)

    if set_turbulence is False:
        vx_turb = 0
        vy_turb = 0

    # behavioral velocities
    vx_behav = velocity * np.cos(phi)
    vy_behav = velocity * np.sin(phi)

    # update positions
    x_pos = x_pos + (vx_turb + vx_behav) * delta_t + np.sqrt(2 * Dt * delta_t) * np.random.randn(N_plankton)
    y_pos = y_pos + (vy_turb + vy_behav) * delta_t + np.sqrt(2 * Dt * delta_t) * np.random.randn(N_plankton)

    # reflect at the boundaries
    x_pos[x_pos > tank_size] = 2 * tank_size - x_pos[x_pos > tank_size]
    x_pos[x_pos < 0] = -x_pos[x_pos < 0]
    y_pos[y_pos > tank_size] = 2 * tank_size - y_pos[y_pos > tank_size]
    y_pos[y_pos < 0] = -y_pos[y_pos < 0]

    # store the positions
    stored_positions[:, 0, t] = x_pos
    stored_positions[:, 1, t] = y_pos

# %% Create animation of velocity field with plankton swimming through it
# Video parameters
trail_length = 20  # Number of previous positions to show in the trail

# Set up the figure and axis for the quiver plot
fig, ax = plt.subplots(figsize=(8, 8))

Xs = X[::stride, ::stride]
Ys = Y[::stride, ::stride]

U = velocity_field[0, ::stride, ::stride, 0]
V = velocity_field[0, ::stride, ::stride, 1]

# Initialize the quiver plot with the first time step
quiver = ax.quiver(Xs, Ys, U, V, scale=1000, pivot='mid', color='blue', alpha=0.2, label='Velocity Field')
title = ax.set_title("T = 0 s")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")

# Add the plankton positions
scat = ax.scatter(stored_positions[:, 0, 0], stored_positions[:, 1, 0], c='black', s=30, alpha=0.5, label='Plankton')
trails = [ax.plot([], [], '-', linewidth=1, color="black", alpha=0.5)[0] for _ in range(N_plankton)]
ax.set_xlim(0, tank_size)
ax.set_ylim(0, tank_size)
ax.legend(loc='upper right')  # created once, not touched again

# Function to update the plot for each frame
def update(frame):
    print(f"\rProcessing frame {frame}/{timesteps} ...", end='', flush=True)
    U = velocity_field[frame, ::stride, ::stride, 0]
    V = velocity_field[frame, ::stride, ::stride, 1]
    quiver.set_UVC(U, V)
    title.set_text(f"T = {frame/fps:.2f} s")
    scat.set_offsets(np.c_[stored_positions[:, 0, frame], stored_positions[:, 1, frame]])

    start = max(0, frame - trail_length)
    for i in range(N_plankton):
        trails[i].set_data(stored_positions[i, 0, start:frame+1], stored_positions[i, 1, start:frame+1])
    return (quiver, scat, title, *trails)

anim = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)

gif_path = "velocityfield_plankton_quiver.gif"
anim.save(gif_path, writer=PillowWriter(fps=fps/step))
plt.close(fig)
Image(filename=gif_path)

# %% Print all the parameters in this simulation

print("Simulation Parameters:")
print(f"Tank size: {tank_size} mm")
print(f"Grid size: {grid_size} (plotting only)")
print(f"Resolution: {tank_size/(grid_size-1):.2f} mm/grid point (plotting only)")
print(f"Physics modelling resolution (dx_physics): {dx_physics} mm -> k_max_numerical = {k_max_numerical:.2f} rad/mm")
print(f"Kolmogorov microscale (eta): {eta:.3f} mm -> k_max_eta = {2*np.pi/eta:.2f} rad/mm")
print(f"Resolved k_max: {k_max_resolved:.2f} rad/mm ({'eta-limited' if 2*np.pi/eta <= k_max_numerical else 'dx_physics-limited'})")
print(f"Number of plankton: {N_plankton}")
print(f"Time steps: {timesteps}")
print(f"Frame rate: {fps} fps")
print(f"Reorientation rate: {reorientation_rate} 1/s (reorientation_rate * delta_t = {reorientation_rate * delta_t:.3f})")
print(f"Step size of video: {step}")
print(f"Stride for quiver plot: {stride}")
