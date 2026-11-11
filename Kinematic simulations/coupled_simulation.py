"""

Coupled simulation of plankton swimming in kinematic turbulence.

Each simulated plankton swims with the empirical turning kernel of one species and light
condition (swimming_behavior.py) and is carried by a kinematic turbulence field
(turbulence_model.py):
  - position: swimming velocity + local flow velocity (kinematic superposition)
  - heading:  own turning (from the kernel) + rotation by half the local vorticity
For every dissipation rate in EPSILON_VALUES, N_TRIALS independent simulations (each with
a new turbulence realisation) are run, and the mean upward velocity and circular variance
of the swimming direction are computed in the same way as for the experimental data
(Trajectory Analysis/load_and_save.py).

Input:  data/<measurement>/traj_<condition>_scaled.pkl (output of load_and_save.py)
Output: plankton_turbulence_results_<measurement>_<condition>.pkl, one per species and condition

"""
#%% Imports
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from swimming_behavior import DELTA_T_MEAS, TARGET_ANGLE, load_behavior_parameters, sample_turning_kernel, wrap_angle
from turbulence_model import generate_field_params, velocity_at, vorticity_at

#%% Simulation parameters
SEED = 0  # random seed, so the simulations can be reproduced exactly

TANK_SIZE = 90       # [mm]
FPS = 10             # stored positions per second
T_SIMULATION = 120   # simulated time [s]
TIMESTEPS = T_SIMULATION * FPS

NU_SI = 1e-6         # kinematic viscosity [m^2/s] (sea water ~1e-6 to 1.8e-6 m^2/s)
N_MODES = 500        # number of Fourier modes of the turbulence field
L = 90               # largest length scale of the turbulence [mm]

N_PLANKTON = 200     # plankton per simulation
N_TRIALS = 50        # simulations per dissipation rate

# Dissipation rates [m^2/s^3]: 0 (behaviour-only control) and 10^-13.5 ... 10^-4 in half decades
EPSILON_VALUES = [0,
                  1*10**(-13.5), 1e-13, 1*10**(-12.5), 1e-12, 1*10**(-11.5), 1e-11,
                  1*10**(-10.5), 1e-10, 1*10**(-9.5), 1e-9, 1*10**(-8.5), 1e-8,
                  1*10**(-7.5), 1e-7, 1*10**(-6.5), 1e-6, 1*10**(-5.5), 1e-5,
                  1*10**(-4.5), 1e-4]

MEASUREMENTS = ["Artemia_0805", "Cladocerans_2605", "BarnacleLarvae_2102"]
CONDITIONS = ["still", "breeze", "stormy"]

MAKE_GIF = False  # True: save an animation of the first trial of every dissipation rate

#%% Trajectory statistics, computed as for the experimental data
# Same thresholds as Trajectory Analysis/load_and_save.py; one simulated trial plays the
# role of one experimental tank.
ARTIFACT_THRESHOLD = 0.0001  # [mm] steps with |dx| or |dy| below this are dropped
MIN_MOVEMENT = 0.3           # [mm] trajectories with a smaller net displacement are stationary
MIN_FRAMES = 30              # trajectories with fewer frames are dropped
ORIENTATION_BIN_EDGES = np.linspace(0, 2 * np.pi, 50 + 1)  # same 50 bins as the polar histograms


def circular_variance(angles):
    """V = 1 - R, with R the mean resultant length of the angles."""
    angles = angles[~np.isnan(angles)]
    R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
    return 1 - R


def speed_weighted_circular_variance(dx, dy):
    """
    V_w = 1 - R_w, with every step direction weighted by its speed:
    R_w = |sum of step velocities| / sum of step speeds (the net-to-gross ratio of the movement).
    """
    return 1 - np.hypot(np.sum(dx), np.sum(dy)) / np.sum(np.hypot(dx, dy))


def trajectory_statistics(stored_positions, T_end, fps):
    """
    Statistics of one simulated trial, following the processing of the experimental data:
      1. dx, dy per frame [mm/s] (y points up, so dy > 0 is upward)
      2. drop NaN steps (plankton that left through the top or bottom)
      3. drop steps with |dx| or |dy| < ARTIFACT_THRESHOLD; like in the experimental
         pipeline this also drops the first step of every trajectory
      4. drop stationary trajectories (net displacement < MIN_MOVEMENT)
      5. drop trajectories with fewer than MIN_FRAMES frames
      6. orientation = arctan2(dy, dx); mean dy and circular variance over all remaining steps
    The experimental acclimation window (time_start) is not applied.

    Returns a dict with the mean upward velocity [mm/s], the circular variance (plain and
    speed weighted), the orientation histogram counts (on ORIENTATION_BIN_EDGES), and the
    number of steps and trajectories used.
    """
    step_x = np.diff(stored_positions[:, 0, :T_end], axis=1)  # [mm/frame]
    step_y = np.diff(stored_positions[:, 1, :T_end], axis=1)
    dx = step_x * fps  # [mm/s]
    dy = step_y * fps

    # 2 + 3 (comparisons with NaN are False, so NaN steps drop out as well)
    valid = (np.abs(step_x) >= ARTIFACT_THRESHOLD) & (np.abs(step_y) >= ARTIFACT_THRESHOLD)
    valid[:, 0] = False

    # 4. net displacement between the end points of the first and the last remaining step
    has_steps = valid.any(axis=1)
    first = np.argmax(valid, axis=1)
    last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    idx = np.arange(valid.shape[0])
    net_displacement = np.hypot(stored_positions[idx, 0, last + 1] - stored_positions[idx, 0, first + 1],
                                stored_positions[idx, 1, last + 1] - stored_positions[idx, 1, first + 1])

    # 5. the simulation has no tracking gaps, so the number of frames is the number of steps
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

#%% Simulation of plankton in turbulence

def couple_plankton(field_params, turning_kernel, initial_headings, tank_size, timesteps, fps,
                    N_plankton, set_turbulence=True, couple_orientation=True, dt_physics_target=0.01,
                    progress_label="", progress_interval=0.5):
    """
    Simulate N_plankton plankton swimming in one turbulence realisation.

    Three time steps are used:
      - behaviour: a new (turning angle, speed) pair is drawn every DELTA_T_MEAS (the frame
        interval of the tracking data the kernel was built from), independent of fps
      - physics:   the flow is integrated in sub-steps of at most dt_physics_target
      - output:    positions are stored fps times per second

    Parameters
        field_params: Fourier-mode coefficients from generate_field_params (None without turbulence)
        turning_kernel, initial_headings: from load_behavior_parameters
        tank_size: [mm]; timesteps: number of stored frames; fps: stored frames per second
        set_turbulence: False runs the behaviour-only control (no flow)
        couple_orientation: False lets the flow only move the plankton, not rotate them
        progress_label, progress_interval: text and minimum time [s] between progress prints

    Returns
        stored_positions: (N_plankton, 2, timesteps) x and y [mm] at every stored frame
        T_end: number of frames actually stored (smaller than timesteps if all plankton
               left the tank before the end)
    """
    output_dt = 1.0 / fps
    behavioral_dt = DELTA_T_MEAS

    substeps_per_behavior = max(1, int(np.ceil(behavioral_dt / dt_physics_target)))
    dt_physics = behavioral_dt / substeps_per_behavior
    physics_steps_per_output = max(1, round(output_dt / dt_physics))

    stored_positions = np.zeros((N_plankton, 2, timesteps))

    # float32, the same precision as the turbulence coefficients
    x_pos = np.random.uniform(0, tank_size, N_plankton).astype(np.float32)
    y_pos = np.random.uniform(0, tank_size, N_plankton).astype(np.float32)
    phi = np.random.choice(initial_headings, size=N_plankton, replace=True).astype(np.float32)

    # +1 so rounding can never leave fewer stored frames than requested
    n_behavior_frames = int(np.ceil(timesteps * output_dt / behavioral_dt)) + 1
    output_idx = 0
    physics_step = 0
    last_progress_print = 0.0
    prefix = f"{progress_label} " if progress_label else ""

    for _ in range(n_behavior_frames):
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

        # Own swimming: draw a measured (turning angle, speed) pair for every plankton
        dphi, speed = sample_turning_kernel(turning_kernel, phi, TARGET_ANGLE)
        dphi = dphi.astype(np.float32, copy=False)
        speed = speed.astype(np.float32, copy=False)
        phi = wrap_angle(phi + dphi)

        for _ in range(substeps_per_behavior):
            if set_turbulence:
                t_phys = physics_step * dt_physics
                positions = np.stack([x_pos, y_pos], axis=-1)
                vx_turb, vy_turb = velocity_at(field_params, positions, t_phys)

                if couple_orientation:
                    # Passive rotation by the flow at half the local vorticity
                    omega_local = vorticity_at(field_params, positions, t_phys)
                    phi = wrap_angle(phi + 0.5 * omega_local * dt_physics)
            else:
                vx_turb = vy_turb = 0.0

            vx_behav = speed * np.cos(phi)
            vy_behav = speed * np.sin(phi)

            # Kinematic superposition: swimming velocity + flow velocity
            x_pos = x_pos + (vx_behav + vx_turb) * dt_physics
            y_pos = y_pos + (vy_behav + vy_turb) * dt_physics

            # Reflect at the side walls; plankton that leave through the top or bottom are removed
            x_pos[x_pos > tank_size] = 2 * tank_size - x_pos[x_pos > tank_size]
            x_pos[x_pos < 0] = -x_pos[x_pos < 0]
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

    return stored_positions, output_idx


def animate_plankton(stored_positions, T_end, tank_size, fps, N_plankton, field_params, gif_name,
                     N_quiver=20, trail_length=20):
    """Save a .gif of the plankton trajectories on top of the flow field (arrows)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    quiver = None
    if field_params is not None:
        x_grid = np.linspace(0, tank_size, N_quiver)
        y_grid = np.linspace(0, tank_size, N_quiver)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_positions = np.stack([X.ravel(), Y.ravel()], axis=-1)

        U = np.zeros((T_end, N_quiver, N_quiver))
        V = np.zeros((T_end, N_quiver, N_quiver))
        for frame in range(T_end):
            vx, vy = velocity_at(field_params, grid_positions, frame / fps)
            U[frame] = vx.reshape(N_quiver, N_quiver)
            V[frame] = vy.reshape(N_quiver, N_quiver)

        # Arrow scale: the 90th percentile speed spans 0.8 of the arrow spacing
        typical_speed = np.percentile(np.sqrt(U**2 + V**2), 90)
        arrow_scale = typical_speed / (0.8 * (X[0, 1] - X[0, 0]))

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

    anim = FuncAnimation(fig, update, frames=np.arange(T_end), interval=50, blit=True)
    anim.save(gif_name, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"\nAnimation saved as {gif_name}.")

#%% Run all dissipation rates for one species and condition

def format_duration(seconds):
    """Format a duration as e.g. '2h 14m', '3m 05s' or '42s'."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_condition(measurement, condition, progress):
    """
    Run N_TRIALS simulations for every dissipation rate in EPSILON_VALUES with the turning
    kernel of one species and light condition, and return the results as a dict.

    progress: dict {"start": time.time(), "total": number of trials, "done": 0}, shared
        between calls so the printed progress and ETA cover the whole run.
    """
    n_eps = len(EPSILON_VALUES)
    mean_upward_velocities_full = np.zeros((n_eps, N_TRIALS))
    circular_variances_full = np.zeros((n_eps, N_TRIALS))
    weighted_circular_variances_full = np.zeros((n_eps, N_TRIALS))
    orientation_counts_full = np.zeros((n_eps, N_TRIALS, len(ORIENTATION_BIN_EDGES) - 1), dtype=int)
    n_steps_full = np.zeros((n_eps, N_TRIALS), dtype=int)
    stored_positions_by_epsilon = {}  # trajectories of the first trial per epsilon, as an example

    behavior = load_behavior_parameters(measurement, condition)
    turning_kernel = behavior["turning_kernel"]
    initial_headings = behavior["initial_headings"]

    for e_idx, epsilon_si in enumerate(EPSILON_VALUES):
        for trial in range(N_TRIALS):
            elapsed = time.time() - progress["start"]
            overall_frac = progress["done"] / progress["total"] if progress["total"] else 0
            eta_str = format_duration(elapsed / overall_frac - elapsed) if overall_frac > 0 else "…"
            label = (f"[{measurement} {condition} | eps {e_idx + 1}/{n_eps}={epsilon_si:.1e} "
                     f"| trial {trial + 1}/{N_TRIALS}]")
            print(f"\n{label} overall {overall_frac * 100:4.1f}% "
                  f"(elapsed {format_duration(elapsed)}, ETA {eta_str})")

            if epsilon_si == 0:
                print(f"{label} Running behavior-only control (no turbulence).")
                field_params = None
            else:
                # Convert to mm for the simulation
                field_params = generate_field_params(epsilon_si * 1e6, NU_SI * 1e6, N_MODES, L)

            stored_positions, T_end = couple_plankton(
                field_params, turning_kernel, initial_headings, TANK_SIZE, TIMESTEPS, FPS,
                N_PLANKTON, set_turbulence=field_params is not None, progress_label=label)

            stats = trajectory_statistics(stored_positions, T_end, FPS)
            mean_upward_velocities_full[e_idx, trial] = stats["mean_upward_velocity"]
            circular_variances_full[e_idx, trial] = stats["circular_variance"]
            weighted_circular_variances_full[e_idx, trial] = stats["speed_weighted_circular_variance"]
            orientation_counts_full[e_idx, trial] = stats["orientation_counts"]
            n_steps_full[e_idx, trial] = stats["n_steps"]
            print(f"\r{label} Mean upward velocity: {stats['mean_upward_velocity']:.4f} mm/s, "
                  f"V = {stats['circular_variance']:.3f}, V_w = {stats['speed_weighted_circular_variance']:.3f}" + " " * 20)

            if trial == 0:
                stored_positions_by_epsilon[epsilon_si] = stored_positions
                if MAKE_GIF:
                    gif_name = f"coupledsim_{condition}_{epsilon_si:.1e}_{measurement}.gif"
                    animate_plankton(stored_positions, T_end, TANK_SIZE, FPS, N_PLANKTON, field_params, gif_name)
            progress["done"] += 1

        # Mean and SD over the trials (ddof=1, as over the tanks in the experiments)
        v = mean_upward_velocities_full[e_idx]
        cv = circular_variances_full[e_idx]
        wcv = weighted_circular_variances_full[e_idx]
        print(f"Average mean upward velocity for epsilon = {epsilon_si:.1e}: {np.mean(v):.4f} mm/s ± {np.std(v, ddof=1):.4f} mm/s, "
              f"V = {np.mean(cv):.3f} ± {np.std(cv, ddof=1):.3f}, "
              f"V_w = {np.mean(wcv):.3f} ± {np.std(wcv, ddof=1):.3f}")

    return {
        "epsilon_values": EPSILON_VALUES,
        "mean_upward_velocities": list(np.mean(mean_upward_velocities_full, axis=1)),
        "std_upward_velocities": list(np.std(mean_upward_velocities_full, axis=1, ddof=1)),
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

#%% Run all species and conditions and save the results
if __name__ == "__main__":
    np.random.seed(SEED)
    progress = {"start": time.time(), "total": len(MEASUREMENTS) * len(CONDITIONS) * len(EPSILON_VALUES) * N_TRIALS, "done": 0}

    for measurement in MEASUREMENTS:
        for condition in CONDITIONS:
            print(f"\n=== {measurement} / {condition} | run elapsed {format_duration(time.time() - progress['start'])} ===")
            results = run_condition(measurement, condition, progress)

            filename = Path(f"plankton_turbulence_results_{measurement}_{condition}.pkl")
            if filename.exists():
                print(f"Warning: Overwriting existing results file {filename}.")
            with open(filename, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved {filename}")

    print(f"\nAll simulations done in {format_duration(time.time() - progress['start'])}.")
