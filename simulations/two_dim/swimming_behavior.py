#%% Import necessary libraries
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image

DELTA_T_MEAS = 0.1  # video frame interval of the tracking data [s] (10 fps)


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


#%% Extract parameters from experimental data to set plankton parameters

def build_turning_kernel(df, target_angle, n_bins=8, min_bin_samples=200):
    """Empirical, deviation-conditioned turning-angle/speed kernel from real trajectories.

    For each bin of "current heading deviation from target_angle", stores the
    (turning_angle, speed) pairs actually observed one frame later in that bin. Sampling
    from this kernel reproduces the real turning-angle distribution (shape, persistence,
    heavy tails), the real speed distribution, their correlation, and the (weak,
    deviation-dependent) restoring bias toward target_angle - all directly from data,
    with no parametric model (uniform-arc heading, Poisson reorientation, isotropic
    diffusion) imposed on top.

    Bins with too few observed transitions (< min_bin_samples) fall back to the full,
    unconditional pool of transitions so the kernel stays well-sampled everywhere.
    """
    df = df.dropna(subset=["speed", "orientation"]).sort_values(["tank", "particle", "t"])
    g = df.groupby(["tank", "particle"])

    dphi = g["orientation"].diff().apply(wrap_angle)
    consecutive = g["t"].diff() == 1  # only use frame-to-frame pairs, skip gaps
    valid = dphi.notna() & consecutive

    dev_before = wrap_angle(target_angle - df["orientation"]).shift(1)

    dphi_v = dphi[valid].to_numpy()
    speed_v = df["speed"][valid].to_numpy()
    dev_v = dev_before[valid].to_numpy()

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(dev_v, bin_edges) - 1, 0, n_bins - 1)

    kernel = []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() < min_bin_samples:
            kernel.append((dphi_v, speed_v))
        else:
            kernel.append((dphi_v[sel], speed_v[sel]))

    return {"bin_edges": bin_edges, "kernel": kernel}


def sample_turning_kernel(turning_kernel, phi, target_angle):
    """Draw one (turning_angle, speed) pair per heading in phi from the empirical kernel,
    conditioned on each heading's current deviation from target_angle."""
    bin_edges = turning_kernel["bin_edges"]
    kernel = turning_kernel["kernel"]
    n_bins = len(kernel)

    dev = wrap_angle(target_angle - phi)
    bin_idx = np.clip(np.digitize(dev, bin_edges) - 1, 0, n_bins - 1)

    dphi_sample = np.zeros_like(phi)
    speed_sample = np.zeros_like(phi)
    for b in range(n_bins):
        sel = bin_idx == b
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        dphi_pool, speed_pool = kernel[b]
        draw = np.random.randint(0, len(dphi_pool), size=n_sel)
        dphi_sample[sel] = dphi_pool[draw]
        speed_sample[sel] = speed_pool[draw]

    return dphi_sample, speed_sample


def load_behavior_parameters(measurement="Artemia_0805", condition="still"):
    """
    condition: which experimental condition's trajectories to build the turning kernel from,
        e.g. "still" (default, no wind/wave stimulus), "breeze", or "stormy" -- selects
        traj_{condition}_scaled.pkl within the measurement's data folder.
    """
    ### Read the data from experiments
    # data/ is shared across model variants (two_dim/, three_dim/, ...) and lives one level up
    # from this file, so resolve it relative to this file's own location rather than the
    # process's current working directory -- otherwise this only works by accident when
    # launched from exactly the right directory.
    data_directory = Path(__file__).parent.parent / "data" / measurement
    path_to_traj = data_directory / f"traj_{condition}_scaled.pkl"

    df_traj = pd.read_pickle(path_to_traj)
    df_traj = df_traj.dropna(subset=["speed", "orientation"])

    target_angle = np.pi / 2

    # Empirical, deviation-conditioned (turning-angle, speed) kernel - see build_turning_kernel.
    # This is what drives the simulation.
    turning_kernel = build_turning_kernel(df_traj, target_angle)

    print(f"turning kernel: {len(turning_kernel['kernel'])} bins, "
          f"{[len(pool[0]) for pool in turning_kernel['kernel']]} samples/bin")

    return {
        "turning_kernel": turning_kernel,
        "initial_headings": df_traj["orientation"].to_numpy(),
    }

# %%
measurement = "Artemia_0805"
condition = "still"  # "still", "breeze", or "stormy"
path_to_measurement = Path("data") / measurement
behavioral_params = load_behavior_parameters(measurement=path_to_measurement, condition=condition)

tank_size = 90  # mm
T_sim = 120
fps = 10
delta_t = 1/fps
timesteps = T_sim * fps

if not np.isclose(delta_t, DELTA_T_MEAS):
    print(f"Warning: fps={fps} (delta_t={delta_t:.3f}s) does not match the data's native "
          f"cadence ({1/DELTA_T_MEAS:.0f} fps, delta_t={DELTA_T_MEAS:.3f}s); the turning kernel "
          "was built from single-frame transitions at that cadence, so resampling it at a "
          "different fps changes the effective turning-angle and speed statistics.")

### Plankton parameters
N_plankton = 100
turning_kernel = behavioral_params["turning_kernel"]

# %% Coupling the empirical turning kernel with plankton swimming
target_angle = np.pi / 2

# Store the positions
stored_positions = np.zeros((N_plankton, 2, timesteps))

# Initialize plankton positions uniformly, and headings by bootstrapping real observed
# orientations - a realistic starting distribution rather than an arbitrary arc/uniform draw
x_pos = np.random.uniform(0, tank_size, N_plankton)
y_pos = np.random.uniform(0, tank_size, N_plankton)
phi = np.random.choice(behavioral_params["initial_headings"], size=N_plankton, replace=True)

T_end = timesteps

for t in range(timesteps):
    print(f"\r Simulating plankton movement for timestep {t}/{timesteps} ...", end='', flush=True)

    # If all plankton have left the simulation domain, break the loop
    if np.isnan(x_pos).all():
        print(f"\nAll plankton have left the simulation domain at timestep {t}. Ending simulation.")
        T_end = t
        break

    # Draw a real (turning_angle, speed) transition per plankton from the empirical kernel,
    # conditioned on each plankton's current deviation from target_angle
    dphi, speed = sample_turning_kernel(turning_kernel, phi, target_angle)
    phi = wrap_angle(phi + dphi)

    # update positions using the sampled speed directly as the velocity magnitude
    x_pos = x_pos + speed * np.cos(phi) * delta_t
    y_pos = y_pos + speed * np.sin(phi) * delta_t

    # reflect at the sides
    x_pos[x_pos > tank_size] = 2 * tank_size - x_pos[x_pos > tank_size]
    x_pos[x_pos < 0] = -x_pos[x_pos < 0]

    # leave simulation at top and bottom - drop both coordinates together so a departed
    # plankton stops being updated (and reoriented) in all subsequent timesteps
    left_domain = (y_pos > tank_size) | (y_pos < 0)
    x_pos[left_domain] = np.nan
    y_pos[left_domain] = np.nan

    # store the positions
    stored_positions[:, 0, t] = x_pos
    stored_positions[:, 1, t] = y_pos

### Save stored positions into a dataframe for later analysis
# Each plankton contributes rows only up to the timestep it left the tank (dropna removes the
# rest), rather than truncating every plankton's trajectory at the first departure.
df_positions = pd.DataFrame({
    "id": np.repeat(np.arange(N_plankton), T_end),
    "t": np.tile(np.arange(T_end), N_plankton),
    "xpos": stored_positions[:, 0, :T_end].flatten(),
    "ypos": stored_positions[:, 1, :T_end].flatten(),
}).dropna(subset=["xpos", "ypos"])

# %% Create animation of velocity field with plankton swimming through it
# Video parameters
trail_length = 20  # Number of previous positions to show in the trail

# Set up the figure and axis for the quiver plot
frame_indices = np.arange(0, T_end, 1)  # Indices of frames to include in the animation

fig, ax = plt.subplots(figsize=(6, 6))

# Initialize the plot with the first time step
title = ax.set_title("T = 0 s")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")

# Add the plankton positions
scat = ax.scatter(stored_positions[:, 0, 0], stored_positions[:, 1, 0], c='black', s=30, alpha=0.5, label='Plankton')
trails = [ax.plot([], [], '-', linewidth=1, color="black", alpha=0.5)[0] for _ in range(N_plankton)]
ax.set_xlim(0, tank_size)
ax.set_ylim(0, tank_size)

# Function to update the plot for each frame
def update(frame):
    print(f"\rProcessing frame {frame}/{T_end} ...", end='', flush=True)

    title.set_text(f"T = {frame/fps:.2f} s")
    scat.set_offsets(np.c_[stored_positions[:, 0, frame], stored_positions[:, 1, frame]])

    start = max(0, frame - trail_length)
    for i in range(N_plankton):
        trails[i].set_data(stored_positions[i, 0, start:frame+1], stored_positions[i, 1, start:frame+1])
    return (scat, title, *trails)

anim = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)

gif_path = "plankton_swimming.gif"
anim.save(gif_path, writer=PillowWriter(fps=fps))
plt.close(fig)
Image(filename=gif_path)


# %%
