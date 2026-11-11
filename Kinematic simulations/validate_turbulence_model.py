"""
Validation of the kinematic turbulence model (turbulence_model.py).

Generates one turbulence realisation on a grid and checks that it
  1. is divergence-free on the grid,
  2. has Fourier coefficients that satisfy the incompressibility constraint,
  3. has an eddy turnover time much shorter than the simulated time,
  4. follows the imposed -5/3 energy spectrum,
  5. is isotropic (equal variance of u_x and u_y),
  6. dissipates energy at the imposed rate epsilon.
Finally it advects neutrally buoyant tracers through the field and saves an animation
(velocityfield_particles.gif).
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from turbulence_model import create_velocity_field, epsilon_from_weather, velocity_at, vorticity_at

np.random.seed(0)  # fixed seed so the run can be reproduced exactly

#%% Range of epsilon expected from the weather in the field
depth_m = 0.09  # depth at which epsilon is evaluated [m]: 9 cm, the depth of the experimental tank

epsilon_min = epsilon_from_weather(0.01, 0.01, z_m=depth_m)  # calm: 0.01 m/s wind, 0.01 m waves
epsilon_max = epsilon_from_weather(12.0, 3.0, z_m=depth_m)   # rough: 12 m/s wind, 3 m waves
print(f"Estimated epsilon range from weather: {epsilon_min:.2e} to {epsilon_max:.2e} m^2/s^3")

#%% Parameters of the velocity field and the simulation
tank_size = 90      # [mm]
fps = 25            # time steps per second
t_simulation = 60   # simulated time [s]
timesteps = t_simulation * fps

nu = 1e-6 * 1e6     # kinematic viscosity [mm^2/s] (sea water ~1e-6 to 1.8e-6 m^2/s)
N = 50              # number of Fourier modes
L = tank_size / 5   # largest length scale of the turbulence [mm]
epsilon_si = 1e-6   # dissipation rate [m^2/s^3]
epsilon = epsilon_si * 1e6  # [mm^2/s^3]

eta = (nu**3 / epsilon) ** (1 / 4)  # Kolmogorov length scale [mm]
dx_physics = eta / 2                # smallest eddy the Fourier modes need to represent [mm]

# Grid resolution: dx <= dx_physics / 8 keeps the finite-difference error of the
# gradient-based checks (1, 4, 6) at ~15-20%. Capped to limit memory use.
MAX_GRID_SIZE = 2000
RESOLUTION_SAFETY_FACTOR = 8
grid_size_needed = int(np.ceil(tank_size / (dx_physics / RESOLUTION_SAFETY_FACTOR))) + 1  # +1: both tank edges
grid_size = min(grid_size_needed, MAX_GRID_SIZE)
if grid_size_needed > MAX_GRID_SIZE:
    print(f"NOTE: resolving eta={eta:.3f} mm at this epsilon with a {RESOLUTION_SAFETY_FACTOR}x safety "
          f"margin would need grid_size={grid_size_needed} "
          f"(~{grid_size_needed**2 * t_simulation * fps * 2 * 4 / 1e9:.1f} GB for velocity_field); "
          f"capped at grid_size={MAX_GRID_SIZE}. Tests 1/4/5/6 may still be inaccurate at this epsilon -- "
          f"raise MAX_GRID_SIZE if you have the memory/time, reduce t_simulation, or trust tests 2/3 instead.")

velocity_field, field_params = create_velocity_field(epsilon, nu, N, L, grid_size, tank_size, timesteps, fps)

#%% Validate the velocity field
dx = tank_size / (grid_size - 1)
k_max_resolved = 2 * np.pi / eta
if dx > dx_physics / RESOLUTION_SAFETY_FACTOR:
    print(f"WARNING: grid spacing dx={dx:.3f} mm exceeds dx_physics/{RESOLUTION_SAFETY_FACTOR}={dx_physics/RESOLUTION_SAFETY_FACTOR:.3f} mm "
          f"(k_max_resolved*dx={k_max_resolved*dx:.2f}, vs pi/{RESOLUTION_SAFETY_FACTOR}={np.pi/RESOLUTION_SAFETY_FACTOR:.2f} target). "
          f"Tests 1/4/5/6 below may still carry significant finite-difference error at this epsilon.")

# 1. Divergence on the grid (should be ~0)
divergence = np.gradient(velocity_field[:, :, :, 0], dx, axis=2) + np.gradient(velocity_field[:, :, :, 1], dx, axis=1)
print(f"1. Maximum divergence: {np.max(np.abs(divergence)):.3e} 1/s ")

# 2. Incompressibility of the Fourier coefficients: A_n . k_n = B_n . k_n = 0
A_dot_k = np.abs(np.sum(field_params["A_n"] * field_params["k_n"], axis=1))
B_dot_k = np.abs(np.sum(field_params["B_n"] * field_params["k_n"], axis=1))
print(f"2. Maximum |A_n . k_n| = {A_dot_k.max():.2e}, max |B_n . k_n| = {B_dot_k.max():.2e}")

# 3. Eddy turnover time of the largest eddies, should be << simulated time
eddy_turnover_time = (L**2 / epsilon) ** (1 / 3)  # [s]
print(f"3. Eddy turnover time: {eddy_turnover_time:.3f} s, Simulation time: {t_simulation:.1f} s, Ratio: {eddy_turnover_time/t_simulation:.3f} (target: << 1.0)")

# 4-6 on three random snapshots
N_snapshots = 3
t_snap_check = np.random.choice(np.arange(0, timesteps, 1), size=N_snapshots, replace=False)
for t_snap in t_snap_check:
    print(f"\nAnalyzing velocity field at timestep {t_snap} ...")
    Vx_snap = velocity_field[t_snap, :, :, 0]
    Vy_snap = velocity_field[t_snap, :, :, 1]

    # 4. Energy spectrum, radially binned, slope fitted over the range of the Fourier modes
    k_axis = 2 * np.pi * np.fft.fftfreq(grid_size, d=dx)  # [rad/mm]
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

    k_mags = np.linalg.norm(field_params["k_n"], axis=1)
    k_lo, k_hi = k_mags.min(), k_mags.max()
    valid = (k_bin_centers > k_lo) & (k_bin_centers < k_hi) & (E_spectrum > 0)
    slope, intercept = np.polyfit(np.log(k_bin_centers[valid]), np.log(E_spectrum[valid]), 1)
    print(f"4. Fitted energy-spectrum slope over k in [{k_lo:.3f}, {k_hi:.3f}] rad/mm: {slope:.3f} (target: {-5/3:.3f})")

    # 5. Isotropy
    var_x = np.var(Vx_snap)
    var_y = np.var(Vy_snap)
    print(f"5. Variance of Vx: {var_x:.3e}, Variance of Vy: {var_y:.3e}, Ratio Vx/Vy: {var_x/var_y:.3f} (target: 1.0)")

    # 6. Dissipation rate from the velocity gradients:
    #    eps = 2*nu*[<(du/dx)^2> + <(dv/dy)^2> + 0.5*<(du/dy + dv/dx)^2>]
    #    (runs a few % low: central differences underestimate gradients of the smallest eddies)
    dVx_dx = np.gradient(Vx_snap, dx, axis=1)
    dVx_dy = np.gradient(Vx_snap, dx, axis=0)
    dVy_dx = np.gradient(Vy_snap, dx, axis=1)
    dVy_dy = np.gradient(Vy_snap, dx, axis=0)
    epsilon_estimated = 2 * nu * (np.mean(dVx_dx**2) + np.mean(dVy_dy**2) + 0.5 * np.mean((dVx_dy + dVy_dx)**2))
    print(f"6. Estimated dissipation rate: {epsilon_estimated:.3e}, Imposed dissipation rate: {epsilon:.3e}, Ratio: {epsilon_estimated/epsilon:.3f} (target: 1.0)")

#%% Advect neutrally buoyant tracers through the field
N_particles = 10

# Sub-step the advection at <= 0.01 s (smaller steps leave the trajectories unchanged)
dt_physics_target = 0.01
substeps_per_frame = max(1, int(np.ceil((1 / fps) / dt_physics_target)))
dt_physics = (1 / fps) / substeps_per_frame

stored_positions = np.zeros((N_particles, 2, timesteps))
stored_orientations = np.zeros((N_particles, timesteps))

x_pos = np.random.uniform(0, tank_size, N_particles)
y_pos = np.random.uniform(0, tank_size, N_particles)
theta = np.random.uniform(0, 2 * np.pi, N_particles)  # orientation marker, for the animation only

for frame in range(timesteps):
    print(f"\r Simulating particle movement for frame {frame}/{timesteps} ...", end='', flush=True)

    for sub in range(substeps_per_frame):
        t_phys = frame / fps + sub * dt_physics

        positions = np.stack([x_pos, y_pos], axis=-1)
        vx_turb, vy_turb = velocity_at(field_params, positions, t_phys)
        x_pos = x_pos + vx_turb * dt_physics
        y_pos = y_pos + vy_turb * dt_physics

        # Rotate the orientation marker at half the local vorticity (does not affect the movement)
        omega_local = vorticity_at(field_params, positions, t_phys)
        theta = (theta + 0.5 * omega_local * dt_physics) % (2 * np.pi)

        # Reflect at the side walls; tracers that leave through the top or bottom are removed
        x_pos[x_pos > tank_size] = 2 * tank_size - x_pos[x_pos > tank_size]
        x_pos[x_pos < 0] = -x_pos[x_pos < 0]
        left_domain = (y_pos > tank_size) | (y_pos < 0)
        x_pos[left_domain] = np.nan
        y_pos[left_domain] = np.nan

    stored_positions[:, 0, frame] = x_pos
    stored_positions[:, 1, frame] = y_pos
    stored_orientations[:, frame] = theta

#%% Animation: velocity field (arrows) with the tracers
trail_length = 3 * fps  # number of previous positions shown as a trail
N_quiver = 20           # arrows per axis
stride = max(1, grid_size // N_quiver)

x_grid = np.linspace(0, tank_size, grid_size)
y_grid = np.linspace(0, tank_size, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)
Xs, Ys = X[::stride, ::stride], Y[::stride, ::stride]
U0, V0 = velocity_field[0, ::stride, ::stride, 0], velocity_field[0, ::stride, ::stride, 1]

# Arrow scale: the 90th percentile speed spans 0.8 of the arrow spacing
speeds = np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)
typical_speed = np.percentile(speeds, 90)
grid_spacing = Xs[0, 1] - Xs[0, 0]
arrow_scale = typical_speed / (0.8 * grid_spacing)

fig, ax = plt.subplots(figsize=(8, 8))
quiver = ax.quiver(Xs, Ys, U0, V0, scale=arrow_scale, scale_units='xy', pivot='mid', color='blue', alpha=0.4)
scat = ax.scatter(stored_positions[:, 0, 0], stored_positions[:, 1, 0], c='black', s=30, zorder=3)
trails = [ax.plot([], [], '-', linewidth=1, color='black', alpha=0.5)[0] for _ in range(N_particles)]

spin_half_length = 1.5  # [mm]
spin_markers = [ax.plot([], [], '-', linewidth=2, color='red', zorder=4)[0] for _ in range(N_particles)]

title = ax.set_title("Velocity Field at T = 0 s")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_xlim(0, tank_size)
ax.set_ylim(0, tank_size)
ax.grid()

quiver_legend = ax.quiverkey(quiver, X=0.9, Y=1.05, U=typical_speed, label=f"{typical_speed:.2f} mm/s", labelpos='E', color='blue')
particle_legend = ax.scatter([], [], c='black', s=30, label='test particles')
spin_legend = ax.plot([], [], '-', linewidth=2, color='red', label='tracer spin (cosmetic)')[0]
ax.legend(handles=[quiver_legend, particle_legend, spin_legend], loc='upper left', bbox_to_anchor=(0, 1.15), ncol=3)


def update_quiver(frame):
    print(f"\rProcessing frame {frame}/{timesteps} ...", end='', flush=True)
    quiver.set_UVC(velocity_field[frame, ::stride, ::stride, 0], velocity_field[frame, ::stride, ::stride, 1])
    scat.set_offsets(np.c_[stored_positions[:, 0, frame], stored_positions[:, 1, frame]])

    start = max(0, frame - trail_length)
    for i in range(N_particles):
        trails[i].set_data(stored_positions[i, 0, start:frame+1], stored_positions[i, 1, start:frame+1])

        x, y = stored_positions[i, 0, frame], stored_positions[i, 1, frame]
        orientation = stored_orientations[i, frame]
        dx_spin, dy_spin = spin_half_length * np.cos(orientation), spin_half_length * np.sin(orientation)
        spin_markers[i].set_data([x - dx_spin, x + dx_spin], [y - dy_spin, y + dy_spin])

    title.set_text(f"Velocity Field at T = {frame/fps:.2f} s")
    return (quiver, scat, title, *trails, *spin_markers)


anim = FuncAnimation(fig, update_quiver, frames=np.arange(timesteps), interval=50, blit=True)
anim.save("velocityfield_particles.gif", writer=PillowWriter(fps=fps))
plt.close(fig)
