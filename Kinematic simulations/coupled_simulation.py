#%% Import necessary libraries
import pickle
import time
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
_sb = _load_functions('swimming_behavior.py', '# %% load behavioral parameters and set up simulation parameters')

generate_field_params = _tm['generate_field_params']
velocity_at = _tm['velocity_at']
vorticity_at = _tm['vorticity_at']
load_behavior_parameters = _sb['load_behavior_parameters']
sample_turning_kernel = _sb['sample_turning_kernel']
wrap_angle = _sb['wrap_angle']
DELTA_T_MEAS = _sb['DELTA_T_MEAS']

# function to format the elapsed time and ETA for progress printouts
def _format_duration(seconds):
    """Formats a duration in seconds as e.g. '2h 14m', '3m 05s', or '42s', for progress/ETA printouts."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"

#%% Trajectory statistics, computed exactly as in Trajectory Analysis/load_and_save.py

# Same thresholds as Trajectory Analysis/load_and_save.py, so simulated and measured values are directly comparable. One simulated trial plays the role of one experimental tank.
ARTIFACT_THRESHOLD = 0.0001  # [mm] steps with |dx| or |dy| below this are dropped
MIN_MOVEMENT = 0.3  # [mm] trajectories with a smaller net (start-to-end) displacement are stationary
MIN_FRAMES = 30  # trajectories with fewer frames are dropped
ORIENTATION_BIN_EDGES = np.linspace(0, 2 * np.pi, 50 + 1)  # same 50 bins as the polar histograms in plot_full_per_species.py

def circular_variance(angles):
    """V = 1 - R (R = mean resultant length); identical to V() in Trajectory Analysis/load_and_save.py."""
    angles = angles[~np.isnan(angles)]
    R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
    return 1 - R

def speed_weighted_circular_variance(dx, dy):
    """
    V_w = 1 - R_w, with each step's direction weighted by its speed: R_w = |sum of step velocities|
    / sum of step speeds, i.e. the net-to-gross ratio of the movement. Unlike V, fast and slow steps
    don't count equally -- e.g. in barnacle larvae, slow downward steps no longer cancel fast upward
    ones. dx, dy are per-step velocities (here displacement, so swimming + flow).
    """
    return 1 - np.hypot(np.sum(dx), np.sum(dy)) / np.sum(np.hypot(dx, dy))

def trajectory_statistics(stored_positions, T_end, fps):
    """
    Per-trial statistics from simulated positions (N_plankton, 2, timesteps), following the
    experimental pipeline in Trajectory Analysis/load_and_save.py step by step:
      1. dx, dy per frame in mm/s (y points up, so dy > 0 is upward, as in the scaled tracking data)
      2. drop NaN steps (plankton that left through the top/bottom)
      3. artifact filter: drop steps where |dx| OR |dy| < ARTIFACT_THRESHOLD; like the experimental
         groupby-diff, this also drops each trajectory's first step
      4. drop stationary trajectories (net displacement < MIN_MOVEMENT)
      5. drop trajectories with fewer than MIN_FRAMES frames
      6. orientation = arctan2(dy, dx); mean dy and V = 1 - R pooled over all remaining frames
    The experimental time window (time_start) is not applied: the simulation has no acclimation
    period to cut.

    Returns a dict with the mean upward velocity [mm/s], circular variance (plain and speed-weighted), orientation histogram
    counts (raw counts on ORIENTATION_BIN_EDGES, so trials can be pooled later), and the number of
    steps/trajectories that went into them.
    """
    step_x = np.diff(stored_positions[:, 0, :T_end], axis=1)  # [mm/frame]
    step_y = np.diff(stored_positions[:, 1, :T_end], axis=1)
    dx = step_x * fps  # [mm/s]
    dy = step_y * fps

    # 2. + 3. (NaN comparisons are False, so NaN steps drop out here too)
    valid = (np.abs(step_x) >= ARTIFACT_THRESHOLD) & (np.abs(step_y) >= ARTIFACT_THRESHOLD)
    valid[:, 0] = False

    # 4. net displacement between the end positions of the first and last remaining step
    has_steps = valid.any(axis=1)
    first = np.argmax(valid, axis=1)
    last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    idx = np.arange(valid.shape[0])
    net_displacement = np.hypot(stored_positions[idx, 0, last + 1] - stored_positions[idx, 0, first + 1],
                                stored_positions[idx, 1, last + 1] - stored_positions[idx, 1, first + 1])

    # 5. the simulation has no tracking gaps, so a trajectory's frame count is its number of steps
    keep_particle = has_steps & (net_displacement >= MIN_MOVEMENT) & (valid.sum(axis=1) >= MIN_FRAMES)
    keep = valid & keep_particle[:, None]

    # 6.
    dx_kept, dy_kept = dx[keep], dy[keep]
    orientation = np.arctan2(dy_kept, dx_kept)
    counts, _ = np.histogram(np.mod(orientation, 2 * np.pi), bins=ORIENTATION_BIN_EDGES)

    return {
        "mean_upward_velocity": np.mean(dy_kept) if dy_kept.size else np.nan,
        "circular_variance": circular_variance(orientation) if orientation.size else np.nan,
        "speed_weighted_circular_variance": speed_weighted_circular_variance(dx_kept, dy_kept) if dx_kept.size else np.nan,
        "orientation_counts": counts,
        "n_steps": int(keep.sum()),
        "n_trajectories": int(keep_particle.sum()),
    }

#%% Define functions for plankton movement

def couple_plankton(field_params, turning_kernel, initial_headings, tank_size, timesteps, fps,
                     N_plankton, set_turbulence=True, couple_orientation=True, dt_physics_target=0.01,
                     progress_label="", progress_interval=0.5):
    """
    Couple the simulated swimming behavior of plankton with the turbulence velocity field, and return the stored positions of the plankton over time.

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
        progress_label: short string (e.g. condition/epsilon/trial) prepended to the per-frame
            progress line, so it's identifiable when this is one call among thousands in a big run
        progress_interval: minimum wall-clock seconds between progress prints

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

    # float32 throughout, matching field_params' dtype (see generate_field_params)
    x_pos = np.random.uniform(0, tank_size, N_plankton).astype(np.float32)
    y_pos = np.random.uniform(0, tank_size, N_plankton).astype(np.float32)
    phi = np.random.choice(initial_headings, size=N_plankton, replace=True).astype(np.float32)

    # +1 as a rounding safety margin to avoid storing fewer frames than requested when output_dt is not an integer multiple of behavioral_dt
    n_behavior_frames = int(np.ceil(timesteps * output_dt / behavioral_dt)) + 1
    output_idx = 0
    physics_step = 0
    last_progress_print = 0.0
    prefix = f"{progress_label} " if progress_label else ""

    for b in range(n_behavior_frames):
        if output_idx >= timesteps:
            break
        if np.isnan(x_pos).all():
            print(f"\n{prefix}All plankton have left the simulation domain at output frame {output_idx}. Ending simulation.")
            break

        now = time.time()
        if now - last_progress_print >= progress_interval or output_idx >= timesteps - 1:
            pct = 100 * output_idx / timesteps if timesteps else 100
            print(f"\r{prefix}frame {output_idx}/{timesteps} ({pct:4.1f}%) ...", end='', flush=True)
            last_progress_print = now

        # Sample the turning kernel once per true DELTA_T_MEAS interval, for all plankton at once
        dphi, speed = sample_turning_kernel(turning_kernel, phi, target_angle)
        dphi = dphi.astype(np.float32, copy=False)
        speed = speed.astype(np.float32, copy=False)
        phi = wrap_angle(phi + dphi)

        # Sub-step turbulence advection within this behavioral interval for accurate integration, independent of fps
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

#%% Set the parameters for single simulation run (one species/condition/epsilon/trial) and run it

measurement = "BarnacleLarvae_2102"  # which behavioral dataset (species/date) in data/ to use for the turning kernel

### Simulation parameters
tank_size = 90
fps = 10 # temporal resolution [fps]
t_simulation = 120 # total length of simulation [s]

### Parameters for velocity field (in SI units * 1e6 to convert to mm)
nu_si = 1e-6 # Kinematic viscosity [m^2/s]; sea water ranges ~1e-6 to 1.8e-6 m^2/s. Multiplied by 1e6 below to get mm^2/s
N_modes = 500 # Total number of wave numbers sampled

L = 90   # Maximum length scale of turbulence [mm]
timesteps = t_simulation * fps
delta_t = 1 / fps

### Parameters for the plankton in the simulation
N_plankton = 200
couple_orientation = True  # if True, the local flow also rotates the animal's heading 
epsilon_values = [0, 1*10**(-13.5), 1e-13, 1*10**(-12.5),1e-12, 1*10**(-11.5),1e-11, 1*10**(-10.5),1e-10, 1*10**(-9.5), 1e-9, 1*10**(-8.5),1e-8, 1*10**(-7.5), 1e-7,1*10**(-6.5), 1e-6, 1*10**(-5.5), 1e-5, 1*10**(-4.5), 1e-4]  # Dissipation rates to test [m^2/s^3]
# epsilon_values = [0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,  1e-9, 1e-8,  1e-7, 1e-6,  1e-5,  1e-4]  # Dissipation rates to test [m^2/s^3]

# epsilon_values = [0, 1e-12, 1e-8, 1e-4]

N_trials = 50

MAKE_GIF = False  # if True, save one .gif animation from that epsilon's first trial

#%% Run the simulation for each epsilon value and store the results
mean_upward_velocities_full = np.zeros((len(epsilon_values), N_trials))
circular_variances_full = np.zeros((len(epsilon_values), N_trials))
weighted_circular_variances_full = np.zeros((len(epsilon_values), N_trials))
orientation_counts_full = np.zeros((len(epsilon_values), N_trials, len(ORIENTATION_BIN_EDGES) - 1), dtype=int)
n_steps_full = np.zeros((len(epsilon_values), N_trials), dtype=int)
std_upward_velocities = []
mean_upward_velocities = []
stored_positions_by_epsilon = {}  # trajectories of the first trial per epsilon, as an example (all trials would be ~4 GB)

condition = "breeze"  # which experimental condition's trajectories to use: "still", "breeze", or "stormy"
behavior = load_behavior_parameters(measurement, condition=condition)  # load_behavior_parameters builds "data/" + measurement itself
turning_kernel = behavior["turning_kernel"]
initial_headings = behavior["initial_headings"]

for e_idx, epsilon_si in enumerate(epsilon_values):
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

        # Calculate mean upward velocity and circular variance, same parameters and filtering as used for the experimental data analysis
        stats = trajectory_statistics(stored_positions, T_end, fps)
        mean_upward_velocity = stats["mean_upward_velocity"]
        circular_variances_full[e_idx, trial] = stats["circular_variance"]
        weighted_circular_variances_full[e_idx, trial] = stats["speed_weighted_circular_variance"]
        orientation_counts_full[e_idx, trial] = stats["orientation_counts"]
        n_steps_full[e_idx, trial] = stats["n_steps"]

        print(f"Mean upward velocity of plankton for epsilon = {epsilon_si:.1e}: {mean_upward_velocity:.4f} mm/s, "
              f"V = {stats['circular_variance']:.3f}", end = '\r', flush=True)
        mean_upward_velocity_trials.append(mean_upward_velocity)
        if trial == 0:
            stored_positions_by_epsilon[epsilon_si] = stored_positions

        if MAKE_GIF and trial == 0:  # one representative gif per epsilon, from its first trial
            gif_name = f"coupledsim_{condition}_{epsilon_si:.1e}_{Path(measurement).name}.gif"
            animate_plankton(stored_positions, T_end, tank_size, fps, N_plankton, epsilon_si, field_params, gif_name)

    # Average the results across all trials for this epsilon (ddof=1, as across tanks in Trajectory Analysis)
    mean_upward_velocity = np.mean(mean_upward_velocity_trials)
    std_upward_velocity = np.std(mean_upward_velocity_trials, ddof=1)

    print(f"Average mean upward velocity for epsilon = {epsilon_si:.1e}: {mean_upward_velocity:.4f} mm/s ± {std_upward_velocity:.4f} mm/s, "
          f"V = {np.mean(circular_variances_full[e_idx]):.3f} ± {np.std(circular_variances_full[e_idx], ddof=1):.3f}")
    mean_upward_velocities.append(mean_upward_velocity)
    mean_upward_velocities_full[e_idx, :] = mean_upward_velocity_trials
    std_upward_velocities.append(std_upward_velocity)

### Save the results for later analysis or plotting
results = {
    "epsilon_values": epsilon_values,
    "mean_upward_velocities": mean_upward_velocities,
    "std_upward_velocities": std_upward_velocities,
    "mean_upward_velocities_full": mean_upward_velocities_full,
    "circular_variances_full": circular_variances_full,
    "mean_circular_variances": np.mean(circular_variances_full, axis=1),
    "std_circular_variances": np.std(circular_variances_full, axis=1, ddof=1),
    "weighted_circular_variances_full": weighted_circular_variances_full,
    "mean_weighted_circular_variances": np.mean(weighted_circular_variances_full, axis=1),
    "std_weighted_circular_variances": np.std(weighted_circular_variances_full, axis=1, ddof=1),
    "orientation_counts_full": orientation_counts_full,
    "orientation_bin_edges": ORIENTATION_BIN_EDGES,
    "n_steps_full": n_steps_full,
    "stored_positions_by_epsilon": stored_positions_by_epsilon,
}

measurement_name = Path(measurement).name  # e.g. "Artemia_0805"
filename = Path(f"plankton_turbulence_results_{measurement_name}_{condition}.pkl")

if filename.exists():
    print(f"Warning: Overwriting existing results file {filename}.")
with open(filename, "wb") as f:
    pickle.dump(results, f)

#%% Function: run every epsilon N trials for each condition and save results per condition
def run_and_save_conditions(measurement, conditions, progress=None):
    """
    Runs N_trials simulations per epsilon in epsilon_values for each condition in `conditions`,
    with the turning kernel of `measurement` (e.g. "Artemia_0805", in data/) under that condition,
    and saves plankton_turbulence_results_<species>_<condition>.pkl per condition (overwriting
    any existing file). Uses the simulation parameters set in the parameter cell above.

    progress: optional dict {"start": time.time(), "total": number of trials, "done": 0} shared
        across several calls, so the printed %/elapsed/ETA covers a whole multi-species run;
        created fresh (covering just this call) when omitted. Returned for reuse.
    """
    measurement_name = Path(measurement).name  # e.g. "Artemia_0805"
    if progress is None:
        progress = {"start": time.time(), "total": len(conditions) * len(epsilon_values) * N_trials, "done": 0}

    for c_idx, condition in enumerate(conditions):
        print(f"\n=== {measurement_name} / condition: {condition} ({c_idx + 1}/{len(conditions)}) "
              f"| run elapsed {_format_duration(time.time() - progress['start'])} ===")
        mean_upward_velocities_full = np.zeros((len(epsilon_values), N_trials))
        circular_variances_full = np.zeros((len(epsilon_values), N_trials))
        weighted_circular_variances_full = np.zeros((len(epsilon_values), N_trials))
        orientation_counts_full = np.zeros((len(epsilon_values), N_trials, len(ORIENTATION_BIN_EDGES) - 1), dtype=int)
        n_steps_full = np.zeros((len(epsilon_values), N_trials), dtype=int)
        std_upward_velocities = []
        mean_upward_velocities = []
        stored_positions_by_epsilon = {}  # trajectories of the first trial per epsilon, as an example (all trials would be ~4 GB)

        behavior = load_behavior_parameters(measurement, condition=condition)  # load_behavior_parameters builds "data/" + measurement itself
        turning_kernel = behavior["turning_kernel"]
        initial_headings = behavior["initial_headings"]

        for e_idx, epsilon_si in enumerate(epsilon_values):
            mean_upward_velocity_trials = []

            for trial in range(N_trials):  
                elapsed = time.time() - progress["start"]
                overall_frac = progress["done"] / progress["total"] if progress["total"] else 0
                eta_str = _format_duration(elapsed / overall_frac - elapsed) if overall_frac > 0 else "…"
                label = (f"[{measurement_name} {condition} {c_idx + 1}/{len(conditions)} | eps {e_idx + 1}/{len(epsilon_values)}="
                         f"{epsilon_si:.1e} | trial {trial + 1}/{N_trials}]")
                
                print(f"\n{label} overall {overall_frac * 100:4.1f}% "
                      f"(elapsed {_format_duration(elapsed)}, ETA {eta_str})")

                # Convert to mm2/s^3 for the simulation
                epsilon = epsilon_si * 1e6
                nu = nu_si * 1e6

                if epsilon_si == 0:
                    print(f"{label} Running behavior-only control (no turbulence).")
                    set_turbulence = False
                    field_params = None  # No turbulence field needed for the control case
                else:
                    set_turbulence = True
                    field_params = generate_field_params(epsilon, nu, N_modes, L)

                stored_positions, T_end = couple_plankton(field_params, turning_kernel, initial_headings, tank_size, timesteps, fps, N_plankton, set_turbulence=set_turbulence, progress_label=label)

                stats = trajectory_statistics(stored_positions, T_end, fps)
                mean_upward_velocity = stats["mean_upward_velocity"]
                circular_variances_full[e_idx, trial] = stats["circular_variance"]
                weighted_circular_variances_full[e_idx, trial] = stats["speed_weighted_circular_variance"]
                orientation_counts_full[e_idx, trial] = stats["orientation_counts"]
                n_steps_full[e_idx, trial] = stats["n_steps"]

                print(f"\r{label} Mean upward velocity: {mean_upward_velocity:.4f} mm/s, "
                      f"V = {stats['circular_variance']:.3f}, V_w = {stats['speed_weighted_circular_variance']:.3f}" + " " * 20)
                mean_upward_velocity_trials.append(mean_upward_velocity)
                if trial == 0:
                    stored_positions_by_epsilon[epsilon_si] = stored_positions
                progress["done"] += 1

                if MAKE_GIF and trial == 0:  # one representative gif per epsilon, from its first trial
                    gif_name = f"coupledsim_{condition}_{epsilon_si:.1e}_{measurement_name}.gif"
                    animate_plankton(stored_positions, T_end, tank_size, fps, N_plankton, epsilon_si, field_params, gif_name)

            # Average the results across all trials for this epsilon (ddof=1, as across tanks in Trajectory Analysis)
            mean_upward_velocity = np.mean(mean_upward_velocity_trials)
            std_upward_velocity = np.std(mean_upward_velocity_trials, ddof=1)

            print(f"Average mean upward velocity for epsilon = {epsilon_si:.1e}: {mean_upward_velocity:.4f} mm/s ± {std_upward_velocity:.4f} mm/s, "
                  f"V = {np.mean(circular_variances_full[e_idx]):.3f} ± {np.std(circular_variances_full[e_idx], ddof=1):.3f}, "
                  f"V_w = {np.mean(weighted_circular_variances_full[e_idx]):.3f} ± {np.std(weighted_circular_variances_full[e_idx], ddof=1):.3f}")
            mean_upward_velocities.append(mean_upward_velocity)
            mean_upward_velocities_full[e_idx, :] = mean_upward_velocity_trials
            std_upward_velocities.append(std_upward_velocity)

        ### Save the results for later analysis or plotting
        results = {
            "epsilon_values": epsilon_values,
            "mean_upward_velocities": mean_upward_velocities,
            "std_upward_velocities": std_upward_velocities,
            "mean_upward_velocities_full": mean_upward_velocities_full,
            "circular_variances_full": circular_variances_full,
            "mean_circular_variances": np.mean(circular_variances_full, axis=1),
            "std_circular_variances": np.std(circular_variances_full, axis=1, ddof=1),
            "weighted_circular_variances_full": weighted_circular_variances_full,
            "mean_weighted_circular_variances": np.mean(weighted_circular_variances_full, axis=1),
            "std_weighted_circular_variances": np.std(weighted_circular_variances_full, axis=1, ddof=1),
            "orientation_counts_full": orientation_counts_full,
            "orientation_bin_edges": ORIENTATION_BIN_EDGES,
            "n_steps_full": n_steps_full,
            "stored_positions_by_epsilon": stored_positions_by_epsilon,
            "turning_kernel": turning_kernel,
            "initial_headings": initial_headings,
        }

        filename = Path(f"plankton_turbulence_results_{measurement_name}_{condition}.pkl")
        if filename.exists():
            print(f"Warning: Overwriting existing results file {filename}.")
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {filename}")

    return progress

#%% Rerun ALL simulations

all_measurements = ["Artemia_0805", "Cladocerans_2605", "BarnacleLarvae_2102"]
all_conditions = ["still", "breeze", "stormy"]

progress = {"start": time.time(), "total": len(all_measurements) * len(all_conditions) * len(epsilon_values) * N_trials, "done": 0}

for m_idx, m in enumerate(all_measurements):
    print(f"\n##### Species {m_idx + 1}/{len(all_measurements)}: {Path(m).name} #####")
    run_and_save_conditions(m, all_conditions, progress)

print(f"\nAll simulations done in {_format_duration(time.time() - progress['start'])}.")
