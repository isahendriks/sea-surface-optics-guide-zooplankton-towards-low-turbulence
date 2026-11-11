"""
Behaviour-only simulation (no turbulence): plankton swim with the empirical turning
kernel of one species and light condition. Used to check that the turning kernel
reproduces upward swimming, and saves an animation (plankton_swimming.gif).
"""
#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from swimming_behavior import (DELTA_T_MEAS, TARGET_ANGLE, load_behavior_parameters,
                               sample_turning_kernel, wrap_angle)

np.random.seed(0)  # fixed seed so the run can be reproduced exactly

#%% Parameters
measurement = "Artemia_0805"
condition = "still"  # "still", "breeze" or "stormy"

tank_size = 90   # [mm]
T_sim = 120      # simulated time [s]
fps = 10         # time steps per second
delta_t = 1 / fps
timesteps = T_sim * fps
N_plankton = 100

if not np.isclose(delta_t, DELTA_T_MEAS):
    print(f"Warning: fps={fps} (delta_t={delta_t:.3f}s) does not match the data's native "
          f"cadence ({1/DELTA_T_MEAS:.0f} fps, delta_t={DELTA_T_MEAS:.3f}s); the turning kernel "
          "was built from single-frame transitions at that cadence, so resampling it at a "
          "different fps changes the effective turning-angle and speed statistics.")

behavioral_params = load_behavior_parameters(measurement, condition)
turning_kernel = behavioral_params["turning_kernel"]

#%% Simulate
stored_positions = np.zeros((N_plankton, 2, timesteps))

# Start at random positions, with headings drawn from the measured headings
x_pos = np.random.uniform(0, tank_size, N_plankton)
y_pos = np.random.uniform(0, tank_size, N_plankton)
phi = np.random.choice(behavioral_params["initial_headings"], size=N_plankton, replace=True)

T_end = timesteps
for t in range(timesteps):
    print(f"\r Simulating plankton movement for timestep {t}/{timesteps} ...", end='', flush=True)

    if np.isnan(x_pos).all():
        print(f"\nAll plankton have left the simulation domain at timestep {t}. Ending simulation.")
        T_end = t
        break

    # Draw a measured (turning angle, speed) pair for every plankton
    dphi, speed = sample_turning_kernel(turning_kernel, phi, TARGET_ANGLE)
    phi = wrap_angle(phi + dphi)

    x_pos = x_pos + speed * np.cos(phi) * delta_t
    y_pos = y_pos + speed * np.sin(phi) * delta_t

    # Reflect at the side walls
    x_pos[x_pos > tank_size] = 2 * tank_size - x_pos[x_pos > tank_size]
    x_pos[x_pos < 0] = -x_pos[x_pos < 0]

    # Plankton that leave through the top or bottom are removed from the simulation
    left_domain = (y_pos > tank_size) | (y_pos < 0)
    x_pos[left_domain] = np.nan
    y_pos[left_domain] = np.nan

    stored_positions[:, 0, t] = x_pos
    stored_positions[:, 1, t] = y_pos

#%% Animation
trail_length = 20  # number of previous positions shown as a trail

fig, ax = plt.subplots(figsize=(6, 6))
title = ax.set_title("T = 0 s")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")

scat = ax.scatter(stored_positions[:, 0, 0], stored_positions[:, 1, 0], c='black', s=30, alpha=0.5, label='Plankton')
trails = [ax.plot([], [], '-', linewidth=1, color="black", alpha=0.5)[0] for _ in range(N_plankton)]
ax.set_xlim(0, tank_size)
ax.set_ylim(0, tank_size)


def update(frame):
    print(f"\rProcessing frame {frame}/{T_end} ...", end='', flush=True)
    title.set_text(f"T = {frame/fps:.2f} s")
    scat.set_offsets(np.c_[stored_positions[:, 0, frame], stored_positions[:, 1, frame]])

    start = max(0, frame - trail_length)
    for i in range(N_plankton):
        trails[i].set_data(stored_positions[i, 0, start:frame+1], stored_positions[i, 1, start:frame+1])
    return (scat, title, *trails)


anim = FuncAnimation(fig, update, frames=np.arange(T_end), interval=50, blit=True)
anim.save("plankton_swimming.gif", writer=PillowWriter(fps=fps))
plt.close(fig)
