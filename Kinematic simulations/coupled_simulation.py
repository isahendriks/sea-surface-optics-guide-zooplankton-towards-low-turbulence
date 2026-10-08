#%% Import necessary libraries
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_THIS_DIR = Path(__file__).parent

#%% Load functions from turbulence model and swimming model script
def _load_functions(module_name, stop_before):
    """Import only the functions from the scripts without executing the rest of the code."""
    module_path = _THIS_DIR / module_name
    source = module_path.read_text(encoding="utf-8").split(stop_before)[0]
    ns = {"__file__": str(module_path)}  # so e.g. swimming_behavior's own __file__-relative paths resolve correctly
    exec(compile(source, module_name, 'exec'), ns)
    return ns

_tm = _load_functions('turbulence_model.py', '# %% Set parameters of the velocity field and simulation')
_sb = _load_functions('swimming_behavior.py', '# %%\nmeasurement = "Artemia_0805"')

generate_field_params = _tm['generate_field_params']
velocity_at = _tm['velocity_at']
vorticity_at = _tm['vorticity_at']
load_behavior_parameters = _sb['load_behavior_parameters']
sample_turning_kernel = _sb['sample_turning_kernel']
wrap_angle = _sb['wrap_angle']
DELTA_T_MEAS = _sb['DELTA_T_MEAS']

#%% Define functions for plankton movement

def couple_plankton(field_params, turning_kernel, initial_headings, tank_size, timesteps, fps,
                     N_plankton, set_turbulence=True, couple_orientation=True, dt_physics_target=0.01):
    """
    Simulate the coupled movement of plankton in a tank, considering both their swimming behavior and the effects of turbulence.
    
    Important note: 
        Make sure to run the scripts "turbulence_model.py" and "swimming_behavior.py" first with the parameters that are to be used. 
        These scripts validate the turbuelence field and swimming behavior, and generate the necessary data for this simulation.    
    Input parameters:
        field_params: Fourier-mode coefficients from generate_field_params/create_velocity_field
        turning_kernel, initial_headings: from load_behavior_parameters
        timesteps, fps: fps controls how many positions get STORED (for output/animation) and is
            independent of the turning kernel's decision cadence -- a new behavioral sample is
            always drawn at the kernel's true native cadence (DELTA_T_MEAS), regardless of fps,
            so raising or lowering fps changes trajectory resolution, not the underlying
            turning-angle/speed statistics the kernel encodes.
        set_turbulence: False runs the behavior-only control (no flow contribution at all)
        couple_orientation: False reproduces the pure kinematic-superposition behavior (turbulence
            only advects position, never affects heading); ignored when set_turbulence=False
        dt_physics_target: turbulence advection is sub-stepped at (approximately) this timestep,
            independent of both fps and the behavioral cadence, for accurate integration
   
    
    Output:
        stored_positions: numpy array of shape (N_plankton, 2, timesteps) with the x and y
            positions of each plankton at each stored frame
        T_end: the number of frames actually stored (may be less than timesteps if all plankton
            have left the tank before the end of the simulation)
    """

    target_angle = np.pi / 2
    output_dt = 1.0 / fps # spacing between STORED frames
    behavioral_dt = DELTA_T_MEAS  # fixed: the kernel's true native decision cadence, independent of fps

    substeps_per_behavior = max(1, int(np.ceil(behavioral_dt / dt_physics_target)))
    dt_physics = behavioral_dt / substeps_per_behavior             # <= dt_physics_target
    physics_steps_per_output = max(1, round(output_dt / dt_physics))  # when to store a frame

    stored_positions = np.zeros((N_plankton, 2, timesteps))

    x_pos = np.random.uniform(0, tank_size, N_plankton)
    y_pos = np.random.uniform(0, tank_size, N_plankton)
    phi = np.random.choice(initial_headings, size=N_plankton, replace=True)

    # +1 as a rounding safety margin so we never run short of the last requested output frame
    n_behavior_frames = int(np.ceil(timesteps * output_dt / behavioral_dt)) + 1
    output_idx = 0
    physics_step = 0

    for b in range(n_behavior_frames):
        if output_idx >= timesteps:
            break
        if np.isnan(x_pos).all():
            print(f"\nAll plankton have left the simulation domain at output frame {output_idx}. Ending simulation.")
            break

        print(f"\r Simulating plankton movement: behavioral frame {b}/{n_behavior_frames} "
              f"(output frame {output_idx}/{timesteps}) ...", end='', flush=True)

        # One behavioral decision per true DELTA_T_MEAS interval, regardless of fps -- the
        # animal's intended swim is independent of the flow (that's the Option A assumption).
        dphi, speed = sample_turning_kernel(turning_kernel, phi, target_angle)
        phi = wrap_angle(phi + dphi)

        # Sub-step turbulence advection within this behavioral interval for accurate
        # integration, independent of fps (mirrors the fix in turbulence_model.py's particle loop).
        for sub in range(substeps_per_behavior):
            if set_turbulence:
                t_phys = physics_step * dt_physics
                positions = np.stack([x_pos, y_pos], axis=-1)
                vx_turb, vy_turb = velocity_at(field_params, positions, t_phys)

                if couple_orientation:
                    # Passive reorientation by the local flow (Jeffery/Faxen): rotates the
                    # heading at half the local vorticity, on top of the animal's own turning.
                    omega_local = vorticity_at(field_params, positions, t_phys)
                    phi = wrap_angle(phi + 0.5 * omega_local * dt_physics)
            else:
                vx_turb = vy_turb = 0.0

            vx_behav = speed * np.cos(phi)
            vy_behav = speed * np.sin(phi)

            # Kinematic superposition: swim velocity + ambient flow velocity
            x_pos = x_pos + (vx_behav + vx_turb) * dt_physics
            y_pos = y_pos + (vy_behav + vy_turb) * dt_physics

            # reflect at the sides
            x_pos[x_pos > tank_size] = 2 * tank_size - x_pos[x_pos > tank_size]
            x_pos[x_pos < 0] = -x_pos[x_pos < 0]

            # leave the simulation at top and bottom - drop both coordinates together so a
            # departed plankton stops being updated (and reoriented) afterward
            left_domain = (y_pos > tank_size) | (y_pos < 0)
            x_pos[left_domain] = np.nan
            y_pos[left_domain] = np.nan

            physics_step += 1
            if output_idx < timesteps and physics_step % physics_steps_per_output == 0:
                stored_positions[:, 0, output_idx] = x_pos
                stored_positions[:, 1, output_idx] = y_pos
                output_idx += 1
                if output_idx >= timesteps:
                    break

    T_end = output_idx

    return stored_positions, T_end

def animate_plankton(stored_positions, T_end, tank_size, fps, N_plankton, epsilon_si, field_params, N_quiver=20, trail_length=20, gif_name=None):
    """
    Animates the stored plankton trajectories together with the turbulence velocity field and
    saves the result as a .gif. Mirrors swimming_behavior.py's animation (scatter of current
    positions + fading trails) combined with turbulence_model.py's quiver-plot animation.
    """

    frame_indices = np.arange(0, T_end, 1)  # Indices of frames to include in the animation

    fig, ax = plt.subplots(figsize=(6, 6))

    quiver = None
    if field_params is not None:
        x_grid = np.linspace(0, tank_size, N_quiver)
        y_grid = np.linspace(0, tank_size, N_quiver)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_positions = np.stack([X.ravel(), Y.ravel()], axis=-1)

        # Precompute the coarse quiver field for every stored frame (cheap: N_quiver^2 points x T_end)
        U = np.zeros((T_end, N_quiver, N_quiver))
        V = np.zeros((T_end, N_quiver, N_quiver))
        for frame in range(T_end):
            t_phys = frame / fps
            vx, vy = velocity_at(field_params, grid_positions, t_phys)
            U[frame] = vx.reshape(N_quiver, N_quiver)
            V[frame] = vy.reshape(N_quiver, N_quiver)

        speeds = np.sqrt(U**2 + V**2)
        typical_speed = np.percentile(speeds, 90)  # robust to rare large spikes
        grid_spacing = X[0, 1] - X[0, 0]
        arrow_scale = typical_speed / (0.8 * grid_spacing)

        quiver = ax.quiver(X, Y, U[0], V[0], scale=arrow_scale, scale_units='xy', pivot='mid', color='blue', alpha=0.4)
        quiver_legend = ax.quiverkey(quiver, X=0.9, Y=1.05, U=typical_speed, label=f"{typical_speed:.2f} mm/s", labelpos='E', color='blue')

    title = ax.set_title("T = 0 s")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    scat = ax.scatter(stored_positions[:, 0, 0], stored_positions[:, 1, 0], c='black', s=30, alpha=0.5, zorder=3, label='Plankton')
    trails = [ax.plot([], [], '-', linewidth=1, color="black", alpha=0.5)[0] for _ in range(N_plankton)]
    ax.set_xlim(0, tank_size)
    ax.set_ylim(0, tank_size)

    if quiver is not None:
        ax.legend(handles=[quiver_legend, scat], loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2)
    else:
        ax.legend(loc='upper left')

    def update(frame):
        print(f"\rProcessing frame {frame}/{T_end} ...", end='', flush=True)

        title.set_text(f"T = {frame/fps:.2f} s")
        if quiver is not None:
            quiver.set_UVC(U[frame], V[frame])
        scat.set_offsets(np.c_[stored_positions[:, 0, frame], stored_positions[:, 1, frame]])

        start = max(0, frame - trail_length)
        for i in range(N_plankton):
            trails[i].set_data(stored_positions[i, 0, start:frame+1], stored_positions[i, 1, start:frame+1])

        artists = (scat, title, *trails)
        return (quiver, *artists) if quiver is not None else artists

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)

    if gif_name is not None:
        anim.save(gif_name, writer=PillowWriter(fps=fps))

    else:
        gif_path = f"plankton_epsilon_{epsilon_si:.1e}.gif"
        anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"\nAnimation saved as {gif_name if gif_name is not None else gif_path}.")

#%% Simulate plankton movement through different turbulence levels

measurement = "Artemia_0805"  # which behavioral dataset (species/date) in data/ to use for the turning kernel
condition = "still"  # which experimental condition's trajectories to use: "still", "breeze", or "stormy"
behavior = load_behavior_parameters(measurement, condition=condition)  # load_behavior_parameters builds "data/" + measurement itself
turning_kernel = behavior["turning_kernel"]
initial_headings = behavior["initial_headings"]

### Simulation parameters
tank_size = 90
fps = 10 # temporal resolution [fps]
t_simulation = 60 # total length of simulation [s]

### Parameters for velocity field (in SI units * 1e6 to convert to mm)
nu_si = 1e-6 # Kinematic viscosity of sea water depends on salinity and temperature. Ranges from 1e-6 to 1.8e-6, times 1e-6 to convert to mm
N_modes = 100 # Total number of wave numbers sampled

L = 90   # Maximum length scale of turbulence [mm]
timesteps = t_simulation * fps
delta_t = 1 / fps

### Parameters for the plankton in the simulation
N_plankton = 200
couple_orientation = True  # if True, the local flow also rotates the animal's heading 
epsilon_values = [0, 1e-14, 1*10**(-13.5), 1e-13, 1*10**(-12.5),1e-12, 1*10**(-11.5),1e-11, 1*10**(-10.5),1e-10, 1*10**(-9.5), 1e-9, 1*10**(-8.5),1e-8, 1*10**(-7.5), 1e-7,1*10**(-6.5), 1e-6, 1*10**(-5.5), 1e-5, 1*10**(-4.5), 1e-4]  # Dissipation rates to test [m^2/s^3]
N_trials = 15

MAKE_GIF = False  # if True, save one .gif animation from that epsilon's first trial

mean_upward_velocities_full = np.zeros((len(epsilon_values), N_trials))
std_upward_velocities = []
mean_upward_velocities = []
stored_positions_by_epsilon = {}

for epsilon_si in epsilon_values:
    mean_upward_velocity_trials = []
    for trial in range(N_trials):  # run multiple trials for each epsilon to average out stochasticity
        print(f"\nSimulating for epsilon = {epsilon_si:.1e} m^2/s^3 ... trial {trial + 1}/{N_trials}")

        # Convert to mm2/s^3 for the simulation
        epsilon = epsilon_si * 1e6 
        nu = nu_si * 1e6

        if epsilon_si == 0:
            print("Running behavior-only control (no turbulence).")
            set_turbulence = False
            field_params = None  # No turbulence field needed for the control case
        else:
            set_turbulence = True
            field_params = generate_field_params(epsilon, nu, N_modes, L)

        # Run the actual simulation and couple the fields
        stored_positions, T_end = couple_plankton(
            field_params, turning_kernel, initial_headings, tank_size, timesteps, fps,
            N_plankton, set_turbulence=set_turbulence,
        )

        # Calculate mean upward velocity, same parameter as used for the experimental data analysis
        mean_upward_velocity = np.nanmean(np.diff(stored_positions[:, 1, :T_end], axis=1) / delta_t)

        print(f"Mean upward velocity of plankton for epsilon = {epsilon_si:.1e}: {mean_upward_velocity:.4f} mm/s")
        mean_upward_velocity_trials.append(mean_upward_velocity)
        stored_positions_by_epsilon[epsilon_si] = stored_positions

        if MAKE_GIF and trial == 0:  # one representative gif per epsilon, from its first trial
            gif_name = f"coupledsim_{condition}_{epsilon_si:.1e}_{measurement}.gif"
            animate_plankton(stored_positions, T_end, tank_size, fps, N_plankton, epsilon_si, field_params, gif_name)

    # Average the results across all trials for this epsilon
    mean_upward_velocity = np.mean(mean_upward_velocity_trials)
    std_upward_velocity = np.std(mean_upward_velocity_trials)

    print(f"Average mean upward velocity for epsilon = {epsilon_si:.1e}: {mean_upward_velocity:.4f} mm/s ± {std_upward_velocity:.4f} mm/s")
    mean_upward_velocities.append(mean_upward_velocity)
    mean_upward_velocities_full[epsilon_values.index(epsilon_si), :] = mean_upward_velocity_trials
    std_upward_velocities.append(std_upward_velocity)

### Save the results for later analysis or plotting
results = {
    "epsilon_values": epsilon_values,
    "mean_upward_velocities": mean_upward_velocities,
    "std_upward_velocities": std_upward_velocities,
    "mean_upward_velocities_full": mean_upward_velocities_full,
    "stored_positions_by_epsilon": stored_positions_by_epsilon,
}

measurement_name = Path(measurement).name  # e.g. "Artemia_0805"
filename = Path(f"plankton_turbulence_results_{measurement_name}_{condition}.pkl")

if filename.exists():
    print(f"Warning: Overwriting existing results file {filename}.")
with open(filename, "wb") as f:
    pickle.dump(results, f)

#%% Plot mean upward velocity vs epsilon with error bars
epsilon_arr = np.array(epsilon_values)
mean_arr = np.array(mean_upward_velocities)
std_arr = np.array(std_upward_velocities)

is_control = epsilon_arr == 0
eps_turb, mean_turb, std_turb = epsilon_arr[~is_control], mean_arr[~is_control], std_arr[~is_control]

fig, ax = plt.subplots(figsize=(6, 5))

if is_control.any():
    control_mean, control_std = mean_arr[is_control][0], std_arr[is_control][0]
    ax.axhline(control_mean, color='gray', linestyle='--', label='No turbulence (control)')
    ax.axhspan(control_mean - control_std, control_mean + control_std, color='gray', alpha=0.15)

ax.errorbar(eps_turb, mean_turb, yerr=std_turb, fmt='o', capsize=5, color='C0', label='With turbulence')
ax.set_xscale('log')
ax.set_xlabel('Dissipation rate, epsilon [m^2/s^3]')

### Add to xaxis arrow point to still for low epsilon and rough for high epsilon 
ax.annotate('Still', xy=(1e-6, 0), xytext=(1e-5, 0.5),
            arrowprops=dict(arrowstyle='->', color='C0'), fontsize=10)
ax.annotate('Rough', xy=(1e-2, 0), xytext=(1e-1, 0.5),
            arrowprops=dict(arrowstyle='->', color='C1'), fontsize=10)

ax.set_ylabel('Mean upward velocity of plankton [mm/s]')
ax.set_title('Effect of turbulence intensity on net upward swimming speed')
ax.legend()

ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
fig.savefig('upward_velocity_vs_epsilon.png', dpi=150)
plt.show()

#%% Plot all three conditions in one figure for comparison
df_still = pickle.load(open(f"plankton_turbulence_results_{measurement_name}_still.pkl", "rb"))
df_breeze = pickle.load(open(f"plankton_turbulence_results_{measurement_name}_breeze.pkl", "rb"))
df_stormy = pickle.load(open(f"plankton_turbulence_results_{measurement_name}_stormy.pkl", "rb"))

mean_still = np.array(df_still["mean_upward_velocities"])
mean_breeze = np.array(df_breeze["mean_upward_velocities"])
mean_stormy = np.array(df_stormy["mean_upward_velocities"])

std_still = np.array(df_still["std_upward_velocities"])
std_breeze = np.array(df_breeze["std_upward_velocities"])
std_stormy = np.array(df_stormy["std_upward_velocities"])

control_still = mean_still[0]
control_breeze = mean_breeze[0]
control_stormy = mean_stormy[0]

plt.figure(figsize=(6, 5))

# Plot the mean upward velocities with error bars for each condition
plt.errorbar(epsilon_arr[1:], mean_still[1:], yerr=std_still, fmt='o', capsize=5, color='C0', label='Still (control)')
plt.errorbar(epsilon_arr[1:], mean_breeze[1:], yerr=std_breeze, fmt='o', capsize=5, color='C1', label='Breeze')
plt.errorbar(epsilon_arr[1:], mean_stormy[1:], yerr=std_stormy, fmt='o', capsize=5, color='C2', label='Stormy')

# Plot control bands for each condition
plt.axhline(control_still, color='C0', linestyle='--', label='Still (control)')
plt.axhspan(control_still - std_still[0], control_still + std_still[0], color='C0', alpha=0.15)
plt.axhline(control_breeze, color='C1', linestyle='--', label='Breeze (control)')
plt.axhspan(control_breeze - std_breeze[0], control_breeze + std_breeze[0], color='C1', alpha=0.15)
plt.axhline(control_stormy, color='C2', linestyle='--', label='Stormy (control)')
plt.axhspan(control_stormy - std_stormy[0], control_stormy + std_stormy[0], color='C2', alpha=0.15)

plt.xscale('log')
plt.xlabel('Dissipation rate, epsilon [m^2/s^3]')
plt.ylabel('Mean upward velocity of plankton [mm/s]')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Effect of turbulence on mean vertical swimming speed')
plt.legend()
