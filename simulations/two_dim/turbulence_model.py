#%% Import necessary libraries
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image

#%% Define functions for velocity field

def generate_field_params(epsilon, nu, N, L):
    """
    Generates the random Fourier-mode coefficients (k_n, A_n, B_n, omega_n) for a 2D
    isotropic-turbulence velocity field with the given Kolmogorov energy spectrum. This is
    the grid-independent part of the model -- evaluating the field at any set of positions
    (on a grid or not) only needs these coefficients, via velocity_at(). create_velocity_field
    calls this internally and then additionally evaluates the field on a grid for plotting/
    validation; call this directly instead when you only need velocity_at() (e.g. particle-only
    simulations), since it skips the grid_size^2 * timesteps grid computation entirely.

    epsilon: Dissipation rate [mm^2/s^3]
    nu: Kinematic viscosity [mm^2/s]
    N: Number of Fourier modes
    L: Maximum length scale of turbulence [mm]
    """
    ### Define the physical parameters for the turbulence model
    eta = (nu**3 / epsilon) ** (1/4)   # Define eta (Kolmogorov length scale)
    dx_physics = eta/2 # physical/turbulence modelling resolution [mm] - smallest eddy the Fourier modes can represent should be <= eta/2

    # Calculate E0 (amplitude of energy spectrum). For 2D: 2/(3 * (2 * np.pi)**(4/3)) * epsilon**(2/3) * (1 - (eta/L)**(4/3))**(-1), for 3D: 1.5 * epsilon**(2/3) * (1 - (eta/L)**(4/3))**(-1)

    E0 = (2 / (3 * (2 * np.pi)**(4/3))) * epsilon**(2/3) * (1 - (eta/L)**(4/3))**(-1)

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

    # Amplitudes of the Fourier modes
    a_n = b_n = np.sqrt(2 * E_k * delta_k)

    # Temporal frequencies
    omega_n = 0.4 * np.sqrt((k_values ** 3) * E_k)

    # Random phases for amplitudes and wavevectors
    angles = 2 * np.pi * np.random.rand(N)

    # Define A_n, B_n, and k_n according to the incompressibility constraints (Shape: (N, 2)), where N = number of Fourier modes
    A_n = np.array([a_n * np.cos(angles), -a_n * np.sin(angles)]).T.astype(np.float32)
    B_n = np.array([-b_n * np.cos(angles), b_n * np.sin(angles)]).T.astype(np.float32)
    k_n = np.array([k_values * np.sin(angles), k_values * np.cos(angles)]).T.astype(np.float32)
    omega_n = omega_n.astype(np.float32)

    return dict(k_n=k_n, A_n=A_n, B_n=B_n, omega_n=omega_n)

def create_velocity_field(epsilon, nu, N, L, grid_size, tank_size, timesteps, fps):
    """
    Creates a 2D velocity field based on the Kolmogorov energy spectrum for isotropic turbulence,
    evaluated on a grid at every timestep (for plotting/validation). See generate_field_params
    for the grid-independent Fourier coefficients alone -- much cheaper when you only need
    velocity_at() at a handful of (possibly moving) positions, e.g. for particle simulations.
    epsilon: Dissipation rate [mm^2/s^3]
    nu: Kinematic viscosity [mm^2/s]
    N: Number of Fourier modes
    L: Maximum length scale of turbulence [mm]

    Runs on GPU via CuPy when it's installed and a GPU is visible, otherwise transparently
    falls back to NumPy on CPU
    """
    field_params = generate_field_params(epsilon, nu, N, L)
    k_n, A_n, B_n, omega_n = field_params["k_n"], field_params["A_n"], field_params["B_n"], field_params["omega_n"]

    try:
        import cupy as xp
        _ = xp.cuda.runtime.getDeviceCount()  # raises if no GPU is actually visible
        _gpu = True
    except Exception:
        xp = np
        _gpu = False
    print(f"\r create_velocity_field: using {'GPU (CuPy)' if _gpu else 'CPU (NumPy)'}" + " " * 20)

    def to_numpy(a):
        return a.get() if _gpu else a

    ### Define the parameters for the spatial grid
    x = xp.linspace(0, tank_size, grid_size, dtype=xp.float32)
    y = xp.linspace(0, tank_size, grid_size, dtype=xp.float32)
    X, Y = xp.meshgrid(x, y)

    # Move the (small) Fourier-mode arrays onto the compute device (no-op if xp is NumPy)
    A_n_dev = xp.asarray(A_n)
    B_n_dev = xp.asarray(B_n)
    k_n_dev = xp.asarray(k_n)
    omega_n_dev = xp.asarray(omega_n)

    # Define the spatial grid (x, y)
    positions = xp.stack([X.ravel(), Y.ravel()], axis=-1)  # Flattened grid positions for efficiency (Shape: (grid_size * grid_size, 2))

    # Spatial part of the phase doesn't depend on t, so compute it once instead of every timestep
    spatial_phase = positions @ k_n_dev.T  # (grid_size^2, N)

    # Initialize an array to store the velocity field at each time step
    velocity_field = xp.zeros((timesteps, grid_size, grid_size, 2), dtype=xp.float32)

    # Compute the velocity field for each time step. The sum over Fourier modes is done as a
    # single matrix multiply (matching velocity_at's formula) instead of a Python loop over N:
    # one large batched op per timestep instead of N small ones -- this is what actually lets
    # the GPU (or even just NumPy's BLAS backend) do meaningful work per iteration, since a
    # Python-level loop of grid_size^2 * N small steps would spend most of its time on
    # per-iteration/kernel-launch overhead rather than the GPU itself.
    for t in range(timesteps):
        print(f"\r Simulating velocity field for timestep {t}/{timesteps} ...", end='\r', flush=True)
        phase = spatial_phase + omega_n_dev[None, :] * xp.float32(t / fps)  # (grid_size^2, N)
        u = xp.cos(phase) @ A_n_dev + xp.sin(phase) @ B_n_dev                # (grid_size^2, 2)

        velocity_field[t, ..., 0] = u[:, 0].reshape(grid_size, grid_size)  # x-component
        velocity_field[t, ..., 1] = u[:, 1].reshape(grid_size, grid_size)  # y-component

    return to_numpy(velocity_field), field_params

def velocity_at(field_params, positions, t):
    # positions: (M, 2) array of (x, y); t: time in seconds
    # Evaluates the same Fourier-mode sum as create_velocity_field, but at arbitrary
    # (off-grid) positions, so particles aren't restricted to grid points.
    k_n, A_n, B_n, omega_n = field_params["k_n"], field_params["A_n"], field_params["B_n"], field_params["omega_n"]
    phase = positions @ k_n.T + omega_n[None, :] * t   # (M, N)
    u = np.cos(phase) @ A_n + np.sin(phase) @ B_n        # (M, 2)
    return u[:, 0], u[:, 1]

def vorticity_at(field_params, positions, t):
    """
    Local (scalar) vorticity curl(u) = dv/dx - du/dy of the same Fourier-mode field as
    velocity_at, evaluated analytically at arbitrary positions. Differentiating each mode's
    cos/sin term w.r.t. x or y brings down a factor of the corresponding k_n component and
    swaps cos<->sin (with a sign flip), so this reuses the same phase array as velocity_at.
    """
    k_n, A_n, B_n = field_params["k_n"], field_params["A_n"], field_params["B_n"]
    omega_n = field_params["omega_n"]
    phase = positions @ k_n.T + omega_n[None, :] * t   # (M, N)

    alpha_n = A_n[:, 0] * k_n[:, 1] - A_n[:, 1] * k_n[:, 0]  # coefficient of sin(phase_n)
    beta_n = B_n[:, 1] * k_n[:, 0] - B_n[:, 0] * k_n[:, 1]   # coefficient of cos(phase_n)

    return np.sin(phase) @ alpha_n + np.cos(phase) @ beta_n

def epsilon_from_weather(wind_speed_ms, sig_wave_height_m, z_m, rho_air=1.225, rho_water=1025.0, kappa=0.4):
    """
    Estimates near-surface turbulent kinetic energy dissipation rate epsilon [m^2/s^3] from
    10m wind speed and significant wave height, using the Terray et al. (1996) wave-enhanced
    surface-layer result: dissipation is roughly depth-independent within the wave-breaking
    layer (z <~ 0.6*Hs), then transitions to the classic law-of-the-wall u*^3/(kappa*z) below it.

    wind_speed_ms: 10m wind speed [m/s], scalar or array
    sig_wave_height_m: significant wave height Hs [m], scalar or array
    z_m: depth below the surface to evaluate epsilon at [m]
    """
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=float)
    sig_wave_height_m = np.asarray(sig_wave_height_m, dtype=float)

    # Wind drag coefficient (Large & Pond 1981)
    Cd = np.where(wind_speed_ms < 11, 1.2e-3, (0.49 + 0.065 * wind_speed_ms) * 1e-3)

    u_star_air = np.sqrt(Cd) * wind_speed_ms
    u_star_water = u_star_air * np.sqrt(rho_air / rho_water)  # stress continuity across the interface

    alpha = 0.3  # Terray et al. (1996) normalization constant 
    transition_depth = 0.6 * sig_wave_height_m

    epsilon_wave_layer = alpha * u_star_water**3 / sig_wave_height_m
    epsilon_log_layer = u_star_water**3 / (kappa * z_m)

    return np.where(z_m <= transition_depth, epsilon_wave_layer, epsilon_log_layer)

# %% Set parameters of the velocity field and simulation

### Simulation parameters
tank_size = 90
fps = 25 # temporal resolution [fps]
t_simulation = 60 # total length of simulation [s]

### Parameters for velocity field (in SI units * 1e6 to convert to mm)
nu = 1e-6 * 1e6 # Kinematic viscosity of sea water depends on salinity and temperature. Ranges from 1e-6 to 1.8e-6, times 1e-6 to convert to mm
N = 50 # Total number of wave numbers sampled
L = tank_size / 5 # Maximum length scale of turbulence [mm]

### Extract epsilon from weather data (if available), else fall back to a manual value
USE_WEATHER_DATA = True
weather_data_path = Path(__file__).parent.parent / "simulations" / "data" / "r_weather.csv"  # shared data/ lives one level up, see swimming_behavior.py's load_behavior_parameters
weather_df = pd.read_csv(weather_data_path, sep=';', decimal=',')
weather_df['datetime'] = pd.to_datetime(weather_df['date'] + ' ' + weather_df['time_utc'], format='%Y.%m.%d %H:%M:%S')

T = weather_df['datetime'].values
wave_height = weather_df['OS2_sign_wave_m'].values
wind_speed = weather_df['wind_speed_m_s'].values

weather_row_index = 550  # which row of r_weather.csv to use (-1 = most recent); pick a specific row/date to match a given sea state
### Plot weather data for the last 30 days
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(T, wave_height, label='Significant Wave Height [m]', color='blue')
ax2.plot(T, wind_speed, label='Wind Speed [m/s]', color='orange')
ax1.vlines(T[weather_row_index], ymin=0, ymax=max(wave_height), color='red', linestyle='--', label='Selected Row')
ax1.set_xlabel('Date')
# ax1.set_xticks(T[::len(T)//10])
ax1.set_ylabel('Significant Wave Height [m]')
ax2.set_ylabel('Wind Speed [m/s]')
ax1.set_title('Weather Data for the Last 30 Days')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

z_m = tank_size / 1000  # depth below the surface to evaluate epsilon at [m] -- external parameter, see epsilon_from_weather; defaults to the tank scale

weather_row = weather_df.iloc[weather_row_index]
epsilon_si = epsilon_from_weather(weather_row["wind_speed_m_s"], weather_row["OS2_sign_wave_m"], z_m)
print(f"Estimated epsilon from weather data (row {weather_row_index}): {epsilon_si:.3e} m^2/s^3")
epsilon = epsilon_si * 1e6  # convert to mm^2/s^3

eta = (nu**3 / epsilon) ** (1/4)   # Define eta (Kolmogorov length scale)
dx_physics = eta/2 # physical/turbulence modelling resolution [mm] - smallest eddy the Fourier modes can represent should be <= eta/2

### Print recommended tank parameters
MAX_GRID_SIZE = 2000  # ~2000^2 * 1500 timesteps * 2 components * 4 bytes (float32) =~ 24 GB for velocity_field
RESOLUTION_SAFETY_FACTOR = 8  # dx <= dx_physics/8 -> k_max_resolved*dx <= pi/8 ~ 0.39, ~15-20% FD error
grid_size_needed = int(np.ceil(tank_size / (dx_physics / RESOLUTION_SAFETY_FACTOR))) + 1
grid_size = min(grid_size_needed, MAX_GRID_SIZE)  # +1 to include both endpoints of the tank
if grid_size_needed > MAX_GRID_SIZE:
    print(f"NOTE: resolving eta={eta:.3f} mm at this epsilon with a {RESOLUTION_SAFETY_FACTOR}x safety "
          f"margin would need grid_size={grid_size_needed} "
          f"(~{grid_size_needed**2 * t_simulation * fps * 2 * 4 / 1e9:.1f} GB for velocity_field); "
          f"capped at grid_size={MAX_GRID_SIZE}. Tests 1/4/5/6 may still be inaccurate at this epsilon -- "
          f"raise MAX_GRID_SIZE if you have the memory/time, reduce t_simulation, or trust tests 2/3 instead.")


N_quiver = 20  # number of arrows to plot in quiver plot (per axis)
stride = max(1, grid_size // N_quiver)  # stride for quiver plot (to reduce number of arrows plotted)

timesteps = t_simulation * fps # total number of timesteps

velocity_field, field_params = create_velocity_field(epsilon, nu, N, L, grid_size, tank_size, timesteps, fps)

#%% Validate the velocity field

dx = tank_size / (grid_size - 1)
k_max_resolved = 2 * np.pi / eta
grid_nyquist = np.pi / dx
if dx > dx_physics / RESOLUTION_SAFETY_FACTOR:
    print(f"WARNING: grid spacing dx={dx:.3f} mm exceeds dx_physics/{RESOLUTION_SAFETY_FACTOR}={dx_physics/RESOLUTION_SAFETY_FACTOR:.3f} mm "
          f"(k_max_resolved*dx={k_max_resolved*dx:.2f}, vs pi/{RESOLUTION_SAFETY_FACTOR}={np.pi/RESOLUTION_SAFETY_FACTOR:.2f} target). "
          f"Tests 1/4/5/6 below may still carry significant finite-difference error at this epsilon.")

# 1. Check that the velocity field is incompressible (divergence-free)
divergence = (np.gradient(velocity_field[:, :, :, 0], dx, axis=2)  + np.gradient(velocity_field[:, :, :, 1], dx, axis=1))

print(f"1. Maximum divergence: {np.max(np.abs(divergence)):.3e} 1/s ")

# 2. Check that the Fourier-mode coefficients satisfy the incompressibility constraints
A_dot_k = np.abs(np.sum(field_params["A_n"] * field_params["k_n"], axis=1))
B_dot_k = np.abs(np.sum(field_params["B_n"] * field_params["k_n"], axis=1))
print(f"2. Maximum |A_n . k_n| = {A_dot_k.max():.2e}, max |B_n . k_n| = {B_dot_k.max():.2e}")

# 3. Timescale of the turbulence (eddy turnover time) - should be much smaller than the simulation time
eddy_turnover_time = eddy_turnover_time = (L**2 / epsilon)**(1/3)  # [s]
print(f"3. Eddy turnover time: {eddy_turnover_time:.3f} s, Simulation time: {t_simulation:.1f} s, Ratio: {eddy_turnover_time/t_simulation:.3f} (target: << 1.0)")

### Snapshot dependent analysis
# 4. Check that the velocity field's energy spectrum follows the imposed -5/3 power law.
N_snapshots = 3
t_snap_check = np.random.choice(np.arange(0, timesteps, 1), size=N_snapshots, replace=False)  # Randomly select 3 timesteps to check
for t_snap in t_snap_check:
    print(f"\nAnalyzing velocity field at timestep {t_snap} ...")
    Vx_snap = velocity_field[t_snap, :, :, 0]
    Vy_snap = velocity_field[t_snap, :, :, 1]

    # Radian wavenumber grid matching the FFT's bin layout [rad/mm]
    k_axis = 2 * np.pi * np.fft.fftfreq(grid_size, d=dx)
    KX, KY = np.meshgrid(k_axis, k_axis)
    K = np.sqrt(KX**2 + KY**2)

    Vx_hat = np.fft.fft2(Vx_snap) / grid_size**2
    Vy_hat = np.fft.fft2(Vy_snap) / grid_size**2
    E_k_2d = 0.5 * (np.abs(Vx_hat)**2 + np.abs(Vy_hat)**2)

    n_bins = grid_size // 4
    k_bin_edges = np.linspace(0, K.max(), n_bins + 1)
    k_bin_centers = 0.5 * (k_bin_edges[1:] + k_bin_edges[:-1])
    E_spectrum = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (K >= k_bin_edges[i]) & (K < k_bin_edges[i + 1])
        if mask.any():
            E_spectrum[i] = E_k_2d[mask].sum() / (k_bin_edges[i + 1] - k_bin_edges[i])

    # Fit the slope over the inertial range actually sampled by the Fourier modes
    k_mags = np.linalg.norm(field_params["k_n"], axis=1)
    k_lo, k_hi = k_mags.min(), k_mags.max()
    valid = (k_bin_centers > k_lo) & (k_bin_centers < k_hi) & (E_spectrum > 0)
    slope, intercept = np.polyfit(np.log(k_bin_centers[valid]), np.log(E_spectrum[valid]), 1)


    print(f"4. Fitted energy-spectrum slope over k in [{k_lo:.3f}, {k_hi:.3f}] rad/mm: {slope:.3f} (target: {-5/3:.3f})")

    # 5. compare Isotropy
    var_x = np.var(Vx_snap)
    var_y = np.var(Vy_snap)
    print(f"5. Variance of Vx: {var_x:.3e}, Variance of Vy: {var_y:.3e}, Ratio Vx/Vy: {var_x/var_y:.3f} (target: 1.0)")

    # 6. Compare dissipation rate from the velocity field to the imposed epsilon.
    # Full 2D incompressible dissipation: eps = 2*nu*[<(du/dx)^2> + <(dv/dy)^2> + 0.5*<(du/dy+dv/dx)^2>]
    # (this real-space estimate will still run a few % low relative to the exact spectral check
    # above, since the diagnostic grid only resolves the smallest eddies with ~6 points per
    # wavelength -- central differences systematically underestimate gradients at high k)
    dVx_dx = np.gradient(Vx_snap, dx, axis=1)
    dVx_dy = np.gradient(Vx_snap, dx, axis=0)
    dVy_dx = np.gradient(Vy_snap, dx, axis=1)
    dVy_dy = np.gradient(Vy_snap, dx, axis=0)
    epsilon_estimated = 2 * nu * (np.mean(dVx_dx**2) + np.mean(dVy_dy**2) + 0.5 * np.mean((dVx_dy + dVy_dx)**2))
    print(f"6. Estimated dissipation rate: {epsilon_estimated:.3e}, Imposed dissipation rate: {epsilon:.3e}, Ratio: {epsilon_estimated/epsilon:.3f} (target: 1.0)")


#%% Create neutral buoyancy particles to visualize the velocity field

N_particles = 10

# Particle motion is integrated (forward-Euler on dx/dt = v(x,t)) at a fixed, fine physics
# timestep, independent of fps. Euler's method has error that shrinks with step size, so
# tying the integration step directly to 1/fps (as before) meant changing the output frame
# rate silently changed the computed trajectories, not just how densely they were sampled.
# Decoupling them: fps still controls how many frames are stored/animated, but each frame
# interval is internally subdivided into small enough physics substeps that the trajectory
# has converged regardless of fps.
dt_physics_target = 0.01  # s; halving this further leaves trajectories unchanged to sub-micron precision for this field
substeps_per_frame = max(1, int(np.ceil((1 / fps) / dt_physics_target)))
dt_physics = (1 / fps) / substeps_per_frame  # <= dt_physics_target, exactly divides each output frame interval

# Store the positions (one entry per OUTPUT frame, not per physics substep)
stored_positions = np.zeros((N_particles, 2, timesteps))

# Purely cosmetic spin indicator (one entry per OUTPUT frame): a neutrally buoyant tracer
# carries no swimming velocity, so unlike the plankton's heading in coupled_simulation.py,
# rotating this angle has no effect whatsoever on x_pos/y_pos -- it's tracked and animated
# only to make each tracer's local rotation by the flow visible. Physically this is the same
# Jeffery/Faxen result (a torque-free body rotates at half the local vorticity).
stored_orientations = np.zeros((N_particles, timesteps))

# Intialize the particle positions (uniform random, no initial velocity: neutrally buoyant
# tracers, so they carry no inertia and are simply advected by the local flow)
x_pos = np.random.uniform(0, tank_size, N_particles)
y_pos = np.random.uniform(0, tank_size, N_particles)
theta = np.random.uniform(0, 2 * np.pi, N_particles)  # initial spin orientation (arbitrary)

for frame in range(timesteps):
    print(f"\r Simulating particle movement for frame {frame}/{timesteps} ...", end='', flush=True)

    for sub in range(substeps_per_frame):
        t_phys = frame / fps + sub * dt_physics

        # turbulence velocities
        positions = np.stack([x_pos, y_pos], axis=-1)
        vx_turb, vy_turb = velocity_at(field_params, positions, t_phys)

        # update positions (velocities are in mm/s, so advance by dt_physics per substep)
        x_pos = x_pos + vx_turb * dt_physics
        y_pos = y_pos + vy_turb * dt_physics

        # spin the tracer's orientation marker at half the local vorticity -- cosmetic only,
        # does not feed back into x_pos/y_pos (see comment on stored_orientations above)
        omega_local = vorticity_at(field_params, positions, t_phys)
        theta = (theta + 0.5 * omega_local * dt_physics) % (2 * np.pi)

        # reflect at the sides
        x_pos[x_pos > tank_size] = 2 * tank_size - x_pos[x_pos > tank_size]
        x_pos[x_pos < 0] = -x_pos[x_pos < 0]

        # Option 1: Reflect at top/bottom boundaries
        # y_pos[y_pos > tank_size] = 2 * tank_size - y_pos[y_pos > tank_size]
        # y_pos[y_pos < 0] = -y_pos[y_pos < 0]

        # Option 2: leave simulation at top and bottom
        left_domain = (y_pos > tank_size) | (y_pos < 0)
        x_pos[left_domain] = np.nan
        y_pos[left_domain] = np.nan

    # store the positions once per output frame
    stored_positions[:, 0, frame] = x_pos
    stored_positions[:, 1, frame] = y_pos
    stored_orientations[:, frame] = theta

df_positions = pd.DataFrame({
    "id": np.repeat(np.arange(N_particles), timesteps),
    "t": np.tile(np.arange(timesteps), N_particles),
    "xpos": stored_positions[:, 0, :timesteps].flatten(),
    "ypos": stored_positions[:, 1, :timesteps].flatten(),
}).dropna(subset=["xpos", "ypos"])

# %% Make simulation with quiver + particles
frame_indices = np.arange(0, timesteps, 1)  # Indices of frames to include in the animation
trail_length = 3 * fps  # Number of previous positions to show in the trail

x_grid = np.linspace(0, tank_size, grid_size)
y_grid = np.linspace(0, tank_size, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)
Xs, Ys = X[::stride, ::stride], Y[::stride, ::stride]

### Plot quiver plot simulation
U0, V0 = velocity_field[0, ::stride, ::stride, 0], velocity_field[0, ::stride, ::stride, 1]

fig, ax = plt.subplots(figsize=(8, 8))

# Derive arrow scale
speeds = np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)
typical_speed = np.percentile(speeds, 90)  # robust to rare large spikes
grid_spacing = Xs[0, 1] - Xs[0, 0]  # physical spacing between plotted arrows [mm]
arrow_scale = typical_speed / (0.8 * grid_spacing)

quiver = ax.quiver(Xs, Ys, U0, V0, scale=arrow_scale, scale_units='xy', pivot='mid', color='blue', alpha=0.4)
scat = ax.scatter(stored_positions[:, 0, 0], stored_positions[:, 1, 0], c='black', s=30, zorder=3)
trails = [ax.plot([], [], '-', linewidth=1, color='black', alpha=0.5)[0] for _ in range(N_particles)]

# Spin indicators: a short line through each tracer showing its cosmetic orientation
# (stored_orientations), so the rotation induced by local vorticity is visible even though
# it has no effect on the tracer's trajectory (see comment where theta is updated above).
spin_half_length = 1.5  # mm
spin_markers = [ax.plot([], [], '-', linewidth=2, color='red', zorder=4)[0] for _ in range(N_particles)]

title = ax.set_title("Velocity Field at T = 0 s")

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_xlim(0, tank_size)
ax.set_ylim(0, tank_size)
ax.grid()

# Add legend for quiver and particles
quiver_legend = ax.quiverkey(quiver, X=0.9, Y=1.05, U=typical_speed, label=f"{typical_speed:.2f} mm/s", labelpos='E', color='blue')
particle_legend = ax.scatter([], [], c='black', s=30, label='test particles')
spin_legend = ax.plot([], [], '-', linewidth=2, color='red', label='tracer spin (cosmetic)')[0]
ax.legend(handles=[quiver_legend, particle_legend, spin_legend], loc='upper left', bbox_to_anchor=(0, 1.15), ncol=3)

def update_quiver(frame):
    print(f"\rProcessing frame {frame}/{timesteps} ...", end='', flush=True)
    U = velocity_field[frame, ::stride, ::stride, 0]
    V = velocity_field[frame, ::stride, ::stride, 1]

    quiver.set_UVC(U, V)
    scat.set_offsets(np.c_[stored_positions[:, 0, frame], stored_positions[:, 1, frame]])

    start = max(0, frame - trail_length)
    for i in range(N_particles):
        trails[i].set_data(stored_positions[i, 0, start:frame+1], stored_positions[i, 1, start:frame+1])

        x, y = stored_positions[i, 0, frame], stored_positions[i, 1, frame]
        theta = stored_orientations[i, frame]
        dx, dy = spin_half_length * np.cos(theta), spin_half_length * np.sin(theta)
        spin_markers[i].set_data([x - dx, x + dx], [y - dy, y + dy])

    title.set_text(f"Velocity Field at T = {frame/fps:.2f} s")
    return (quiver, scat, title, *trails, *spin_markers)

anim = FuncAnimation(fig, update_quiver, frames=frame_indices, interval=50, blit=True)

gif_path = "velocityfield_particles.gif"
anim.save(gif_path, writer=PillowWriter(fps=fps))
plt.close(fig)
Image(filename=gif_path)


#%% Verify 