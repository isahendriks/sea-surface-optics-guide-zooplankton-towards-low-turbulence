#%% Import necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from IPython.display import Image

#%% Define functions for velocity field and plankton movement

def create_velocity_field(epsilon, nu, N, L, grid_size, tank_size, timesteps, fps):
    """
    Creates a 2D velocity field based on the Kolmogorov energy spectrum for isotropic turbulence.
    epsilon: Dissipation rate [mm^2/s^3]
    nu: Kinematic viscosity [mm^2/s]
    N: Number of Fourier modes
    L: Maximum length scale of turbulence [mm]
    """
    ### Define the parameters for the spatial grid
    x = np.linspace(0, tank_size, grid_size)
    y = np.linspace(0, tank_size, grid_size)
    X, Y = np.meshgrid(x, y)

    ### Define the physical parameters for the turbulence model
    eta = (nu**3 / epsilon) ** (1/4)   # Define eta (Kolmogorov length scale)
    dx_physics = eta/2 # physical/turbulence modelling resolution [mm] - smallest eddy the Fourier modes can represent should be <= eta/2

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

    return velocity_field

def couple_plankton(velocity_field, angle, velocity, reorientation_rate, Dt, tank_size, grid_size, timesteps, fps, N_plankton, set_turbulence):
    target_angle = np.pi / 2
    response_angle = angle * (np.pi / 180)

    # Extract the velocity field components

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

    # Track which plankton are still in the tank; those that reach the top escape and disappear
    alive = np.ones(N_plankton, dtype=bool)

    delta_t = 1.0 / fps
    if reorientation_rate * delta_t > 0.5:
        print(f"Warning: reorientation_rate * delta_t = {reorientation_rate * delta_t:.2f} is not << 1; "
            f"increase fps so individual reorientation events are resolved.")

    for t in range(timesteps):
        print(f"\r Simulating plankton movement for timestep {t}/{timesteps} ...", end='', flush=True)

        # Reorient a subset of plankton this step (Poisson-process approximation at rate reorientation_rate [s^-1]); 
        reorient = np.random.rand(N_plankton) < reorientation_rate * delta_t
        phi[reorient] = target_angle + response_angle * (2 * np.random.rand(int(reorient.sum())) - 1)

        # turbulence velocities
        positions = np.stack([x_pos, y_pos], axis=-1)

        # Extract the velocity field components at the current time step
        vx_field = velocity_field[t, :, :, 0]
        vy_field = velocity_field[t, :, :, 1]

        # Interpolate the velocity field at the plankton positions
        vx_interpolator = RegularGridInterpolator((x_grid, y_grid), vx_field, bounds_error=False, fill_value=None)
        vy_interpolator = RegularGridInterpolator((x_grid, y_grid), vy_field, bounds_error=False, fill_value=None)
        vx_turb = vx_interpolator(positions)
        vy_turb = vy_interpolator(positions)

        if set_turbulence is False:
            vx_turb = 0
            vy_turb = 0

        # behavioral velocities
        vx_behav = velocity * np.cos(phi)
        vy_behav = velocity * np.sin(phi)

        # update positions
        x_new = x_pos + (vx_turb + vx_behav) * delta_t + np.sqrt(2 * Dt * delta_t) * np.random.randn(N_plankton)
        y_new = y_pos + (vy_turb + vy_behav) * delta_t + np.sqrt(2 * Dt * delta_t) * np.random.randn(N_plankton)

        # reflect at the sides
        x_new[x_new > tank_size] = 2 * tank_size - x_new[x_new > tank_size]
        x_new[x_new < 0] = -x_new[x_new < 0]


        # plankton that reach the top or bottom escape the tank and disappear from the simulation
        alive &= (y_new <= tank_size) & (y_new >= 0)

        # freeze plankton that have disappeared at their last position so they stop moving
        x_pos = np.where(alive, x_new, x_pos)
        y_pos = np.where(alive, y_new, y_pos)

        # store the positions (NaN marks plankton that have disappeared, so they drop out of the dataframe/animation)
        stored_positions[:, 0, t] = np.where(alive, x_pos, np.nan)
        stored_positions[:, 1, t] = np.where(alive, y_pos, np.nan)

    return stored_positions

#%% Set parameters for test simulation

### Simulation parameters
tank_size = 90
grid_size = tank_size + 1  # 1 mm grid spacing, only used for plotting/visualization
fps = 10 # temporal resolution [fps]
t_simulation = 60 # total length of simulation [s]
N_plankton = 500 # Number of plankters

### Parameters for velocity field (in SI units * 1e6 to convert to mm)
epsilon = 1e-8 * 1e6 # Dissipation rate , ranges from 1e-4 (rough sea) to 1e-14 (calm sea) - *1e-6 to convert to mm
nu = 1e-6 * 1e6 # Kinematic viscosity of sea water depends on salinity and temperature. Ranges from 1e-6 to 1.8e-6, times 1e-6 to convert to mm
N = 50 # Total number of wave numbers sampled
L = tank_size/3 # Maximum length scale of turbulence [mm]
eta = (nu**3 / epsilon) ** (1/4)   # Define eta (Kolmogorov length scale)
dx_physics = eta/2 # physical/turbulence modelling resolution [mm] - smallest eddy the Fourier modes can represent should be <= eta/2

### Plankton parameters
set_turbulence = True

### Starting conditions of plankton (based on real swimmers)
velocity = 2.4456 #2.4456 # mean upward velocity [mm/s]
angle = 154.12 / 2 # in degrees (plus or minus around the target angle 90 degrees)
Dt = 2.1989e-1 # random swimming component from control [mm^2/s]
reorientation_rate = 0.1628 # mean number of heading-reorientation events per second [s^-1] 

### Video parameters
N_quivers = 30
stride = int(grid_size / N_quivers) # number of grid points to skip when plotting the quiver plot (for clarity)
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

#%% Simulate Animation of Plankton Swimming Through Velocity Field
epsilon_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14] # Dissipation rates to test
mean_upward_velocities = []

for epsilon in epsilon_values:
    print(f"\nSimulating for epsilon = {epsilon:.1e} ...")

    epsilon *= 1e6 # Convert to mm^2/s^3 for the simulation

    # Create the velocity field
    velocity_field = create_velocity_field(epsilon, nu, N, L, grid_size, tank_size, timesteps, fps)

    # Couple plankton to the velocity field and simulate their movement
    stored_positions = couple_plankton(velocity_field, angle, velocity, reorientation_rate, Dt, tank_size, grid_size, timesteps, fps, N_plankton, set_turbulence)

    mean_upward_velocity = np.nanmean(np.diff(stored_positions[:, 1, :], axis=1) / delta_t)

    print(f"Mean upward velocity of plankton for epsilon = {epsilon:.1e}: {mean_upward_velocity:.4f} mm/s")
    mean_upward_velocities.append(mean_upward_velocity)

#%%
plt.scatter(epsilon_values, mean_upward_velocities, marker='o')
plt.xscale('log')
plt.xlabel('Dissipation Rate (epsilon) [mm^2/s^3]')
plt.ylabel('Mean Upward Velocity of Plankton [mm/s]')
plt.title('Mean Upward Velocity of Plankton vs Dissipation Rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#%% Create animation of velocity field with plankton swimming through it

epsilon_sim_vid = [1e-4, 1e-8, 1e-12] # Dissipation rates to create videos for

for epsilon in epsilon_sim_vid:
    print(f"\nCreating animation for epsilon = {epsilon:.1e} ...")
    epsilon *= 1e6 # Convert to mm^2/s^3 for the simulation

    # Simulate the velocity field
    velocity_field = create_velocity_field(epsilon, nu, N, L, grid_size, tank_size, timesteps, fps)

    # Couple plankton to the velocity field and simulate their movement
    stored_positions = couple_plankton(velocity_field, angle, velocity, reorientation_rate, Dt, tank_size, grid_size, timesteps, fps, N_plankton, set_turbulence)

    # Calculate the mean upward velocity of plankton
    mean_upward_velocity = np.nanmean(np.diff(stored_positions[:, 1, :], axis=1) / delta_t)

    trail_length = 20  # Number of previous positions to show in the trail

    # Set up the figure and axis for the quiver plot
    fig, ax = plt.subplots(figsize=(6, 6))

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

        if set_turbulence:
            quiver.set_UVC(U, V)

        else:
            quiver.set_UVC(np.zeros_like(U), np.zeros_like(V))
        
        title.set_text(f"epsilon = {epsilon:.2e} mm^2/s^3, mean dy = {mean_upward_velocity:.2e} mm/s, T = {frame/fps:.2f} s")
        scat.set_offsets(np.c_[stored_positions[:, 0, frame], stored_positions[:, 1, frame]])

        start = max(0, frame - trail_length)
        for i in range(N_plankton):
            trails[i].set_data(stored_positions[i, 0, start:frame+1], stored_positions[i, 1, start:frame+1])
        return (quiver, scat, title, *trails)

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)

    gif_path = "simulation_epsilon_{epsilon:.2e}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps/step))
    plt.close(fig)
    Image(filename=gif_path)

