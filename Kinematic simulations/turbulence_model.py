"""
Kinematic-simulation model of 2D isotropic turbulence.

The velocity field is a sum of N random Fourier modes whose amplitudes follow the
Kolmogorov -5/3 energy spectrum and which are divergence-free by construction:

    u(x, t) = sum_n  A_n cos(k_n . x + omega_n t) + B_n sin(k_n . x + omega_n t)

This module only defines functions; it is imported by coupled_simulation.py and
validate_turbulence_model.py. Lengths are in mm and times in s, so the dissipation
rate epsilon is in mm^2/s^3 and the viscosity nu in mm^2/s (1 m^2/s^3 = 1e6 mm^2/s^3).
"""

import numpy as np


def generate_field_params(epsilon, nu, N, L):
    """
    Draw the random Fourier-mode coefficients of one turbulence realisation.

    This is the grid-independent part of the model: the velocity at any position and
    time follows from these coefficients (see velocity_at).

    Parameters
        epsilon: dissipation rate [mm^2/s^3]
        nu:      kinematic viscosity [mm^2/s]
        N:       number of Fourier modes
        L:       largest length scale of the turbulence [mm]

    Returns a dict with (float32 arrays)
        k_n:     (N, 2) wavevectors [rad/mm]
        A_n:     (N, 2) amplitudes of the cosine terms [mm/s]
        B_n:     (N, 2) amplitudes of the sine terms [mm/s]
        omega_n: (N,)   temporal frequencies [rad/s]
    """
    eta = (nu**3 / epsilon) ** (1 / 4)  # Kolmogorov length scale [mm]
    dx_physics = eta / 2                # smallest eddy the Fourier modes need to represent [mm]

    # Amplitude of the 2D energy spectrum E(k) = E0 * k^(-5/3)
    # (3D equivalent: 1.5 * epsilon**(2/3) * (1 - (eta/L)**(4/3))**(-1))
    E0 = (2 / (3 * (2 * np.pi) ** (4 / 3))) * epsilon ** (2 / 3) * (1 - (eta / L) ** (4 / 3)) ** (-1)

    # Wavenumbers: geometric series from the largest eddy (L) down to the dissipation scale (eta)
    k_min = 2 * np.pi / L
    k_max_numerical = np.pi / dx_physics                 # Nyquist limit of the modelling resolution
    k_max_resolved = min(2 * np.pi / eta, k_max_numerical)
    k_values = k_min * (k_max_resolved / k_min) ** (np.arange(N) / (N - 1))

    E_k = E0 * (k_values ** (-5 / 3))

    # Width of the wavenumber band represented by each mode
    delta_k0 = (k_values[1] - k_values[0]) / 2
    delta_kN = (k_values[-1] - k_values[-2]) / 2
    delta_k_ = (k_values[2:] - k_values[:-2]) / 2
    delta_k = np.concatenate(([delta_k0], delta_k_, [delta_kN]))

    # Mode amplitudes and temporal frequencies
    a_n = b_n = np.sqrt(2 * E_k * delta_k)
    omega_n = 0.4 * np.sqrt((k_values ** 3) * E_k)

    # Random direction of each mode. A_n and B_n are perpendicular to k_n, which makes the
    # field incompressible (divergence-free).
    angles = 2 * np.pi * np.random.rand(N)

    A_n = np.array([a_n * np.cos(angles), -a_n * np.sin(angles)]).T.astype(np.float32)
    B_n = np.array([-b_n * np.cos(angles), b_n * np.sin(angles)]).T.astype(np.float32)
    k_n = np.array([k_values * np.sin(angles), k_values * np.cos(angles)]).T.astype(np.float32)
    omega_n = omega_n.astype(np.float32)

    return dict(k_n=k_n, A_n=A_n, B_n=B_n, omega_n=omega_n)


def create_velocity_field(epsilon, nu, N, L, grid_size, tank_size, timesteps, fps):
    """
    Evaluate a new turbulence realisation on a regular grid for every time step.

    Parameters
        epsilon, nu, N, L: see generate_field_params
        grid_size: number of grid points per axis
        tank_size: side length of the square tank [mm]
        timesteps: number of time steps
        fps:       time steps per second

    Returns
        velocity_field: (timesteps, grid_size, grid_size, 2) float32 array [mm/s],
                        last axis = (x, y) component
        field_params:   the Fourier-mode coefficients (see generate_field_params)
    """
    field_params = generate_field_params(epsilon, nu, N, L)
    k_n, A_n, B_n, omega_n = field_params["k_n"], field_params["A_n"], field_params["B_n"], field_params["omega_n"]

    x = np.linspace(0, tank_size, grid_size, dtype=np.float32)
    y = np.linspace(0, tank_size, grid_size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (grid_size^2, 2)

    # The spatial part of the phase does not change in time, so compute it once
    spatial_phase = positions @ k_n.T  # (grid_size^2, N)

    velocity_field = np.zeros((timesteps, grid_size, grid_size, 2), dtype=np.float32)
    for t in range(timesteps):
        print(f"\r Simulating velocity field for timestep {t}/{timesteps} ...", end='\r', flush=True)
        phase = spatial_phase + omega_n[None, :] * np.float32(t / fps)
        u = np.cos(phase) @ A_n + np.sin(phase) @ B_n  # (grid_size^2, 2)

        velocity_field[t, ..., 0] = u[:, 0].reshape(grid_size, grid_size)
        velocity_field[t, ..., 1] = u[:, 1].reshape(grid_size, grid_size)

    return velocity_field, field_params


def velocity_at(field_params, positions, t):
    """Velocity (u_x, u_y) [mm/s] at positions (M, 2) [mm] and time t [s]."""
    k_n, A_n, B_n, omega_n = field_params["k_n"], field_params["A_n"], field_params["B_n"], field_params["omega_n"]
    phase = positions @ k_n.T + omega_n[None, :] * t  # (M, N)
    u = np.cos(phase) @ A_n + np.sin(phase) @ B_n     # (M, 2)
    return u[:, 0], u[:, 1]


def vorticity_at(field_params, positions, t):
    """Vorticity (curl of the velocity field) [1/s] at positions (M, 2) [mm] and time t [s]."""
    k_n, A_n, B_n, omega_n = field_params["k_n"], field_params["A_n"], field_params["B_n"], field_params["omega_n"]
    phase = positions @ k_n.T + omega_n[None, :] * t  # (M, N)

    alpha_n = A_n[:, 0] * k_n[:, 1] - A_n[:, 1] * k_n[:, 0]  # coefficient of sin(phase_n)
    beta_n = B_n[:, 1] * k_n[:, 0] - B_n[:, 0] * k_n[:, 1]   # coefficient of cos(phase_n)

    return np.sin(phase) @ alpha_n + np.cos(phase) @ beta_n


# Physical constants for converting between weather and dissipation rate
RHO_AIR = 1.225      # air density [kg/m^3]
RHO_WATER = 1025.0   # sea water density [kg/m^3]
KAPPA = 0.4          # von Karman constant
ALPHA_TERRAY = 0.3   # normalisation constant of Terray et al. (1996)


def drag_coefficient(wind_speed_ms):
    """Neutral 10 m wind drag coefficient of Large & Pond (1981)."""
    return np.where(wind_speed_ms < 11, 1.2e-3, (0.49 + 0.065 * wind_speed_ms) * 1e-3)


def epsilon_from_weather(wind_speed_ms, sig_wave_height_m, z_m):
    """
    Estimate the near-surface dissipation rate epsilon [m^2/s^3] from weather.

    Uses the wave-enhanced surface layer of Terray et al. (1996): within the wave-breaking
    layer (z <= 0.6 * Hs) epsilon is about constant with depth, below it epsilon follows
    the law of the wall, u*^3 / (kappa * z).

    Parameters
        wind_speed_ms:     10 m wind speed [m/s] (scalar or array)
        sig_wave_height_m: significant wave height Hs [m] (scalar or array)
        z_m:               depth below the surface [m]
    """
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=float)
    sig_wave_height_m = np.asarray(sig_wave_height_m, dtype=float)

    u_star_air = np.sqrt(drag_coefficient(wind_speed_ms)) * wind_speed_ms
    u_star_water = u_star_air * np.sqrt(RHO_AIR / RHO_WATER)  # stress continuity across the surface

    transition_depth = 0.6 * sig_wave_height_m
    epsilon_wave_layer = ALPHA_TERRAY * u_star_water**3 / sig_wave_height_m
    epsilon_log_layer = u_star_water**3 / (KAPPA * z_m)

    return np.where(z_m <= transition_depth, epsilon_wave_layer, epsilon_log_layer)


def weather_from_epsilon(epsilon, z_m):
    """
    Approximate inverse of epsilon_from_weather, using the law of the wall at depth z_m.

    Parameters
        epsilon: dissipation rate [m^2/s^3]
        z_m:     depth below the surface [m]

    Returns
        wind_speed_ms:     10 m wind speed [m/s]
        sig_wave_height_m: significant wave height Hs [m]
    """
    u_star_water = (epsilon * KAPPA * z_m) ** (1 / 3)
    u_star_air = u_star_water * np.sqrt(RHO_WATER / RHO_AIR)

    # Invert u*_air = sqrt(Cd(U10)) * U10. Above 11 m/s Cd depends on U10, so there is no
    # simple closed form: invert numerically on a fine grid.
    U10_grid = np.logspace(-5, 2, 4000)
    u_star_air_grid = np.sqrt(drag_coefficient(U10_grid)) * U10_grid
    wind_speed_ms = np.interp(u_star_air, u_star_air_grid, U10_grid)

    sig_wave_height_m = 0.6 * (u_star_water ** 2) / (ALPHA_TERRAY * u_star_water ** 3)

    return wind_speed_ms, sig_wave_height_m
