#%% Import necessary libraries
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit, OptimizeWarning
import pickle

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 10,
})

def _sigmoid_logistic(log_eps, bottom, top, log_ec, hill_slope):
    """
    4-parameter logistic (dose-response) curve in log10(epsilon) space -- the standard
    pharmacology "log(agonist) vs. response" fit. Used here because the data has exactly that
    shape: a plateau at low epsilon (turbulence negligible next to the animal's own directed
    swimming), a transition, and a second plateau at high epsilon (turbulence has randomized
    heading enough that net vertical drift saturates).
    """
    return bottom + (top - bottom) / (1 + 10 ** ((log_ec - log_eps) * hill_slope))

def fit_dose_response(epsilon_values, mean_values, label=""):
    """
    Fits _sigmoid_logistic to (epsilon_values, mean_values). epsilon_values must already exclude
    the epsilon=0 control (log10(0) is undefined; the control is what tells you which plateau,
    bottom or top, is the "no turbulence" one).

    Returns a dict with the fitted params, epsilon_c (the critical/inflection dissipation rate,
    same units as epsilon_values) with its estimated standard error, R^2, a dense curve for
    plotting, and `reliable` -- False when epsilon_c had to be extrapolated outside the tested
    range or its relative uncertainty is large (>50%), i.e. the response never actually turned
    over within the epsilon values tested (a flat "breeze"-style series, say), so the fitted
    number is not trustworthy even though curve_fit converged. Returns None if the fit didn't
    converge at all.
    """
    epsilon_values = np.asarray(epsilon_values, dtype=float)
    mean_values = np.asarray(mean_values, dtype=float)
    log_eps = np.log10(epsilon_values)

    bottom0, top0 = mean_values[0], mean_values[-1]  # lowest/highest tested epsilon's own value
    p0 = [bottom0, top0, np.median(log_eps), 1.0]
    try:
        # An unconstrained covariance here (OptimizeWarning) means the fit found *a* curve but
        # couldn't pin down its uncertainty -- exactly the "reliable=False" case computed below,
        # so it's redundant noise on stdout rather than new information; suppressed accordingly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(_sigmoid_logistic, log_eps, mean_values, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError) as e:
        print(f"Note: dose-response fit did not converge for {label or 'series'}: {e}")
        return None

    bottom, top, log_ec, hill_slope = popt
    perr = np.sqrt(np.diag(pcov))
    ec = 10 ** log_ec
    ec_err = ec * np.log(10) * perr[2]  # propagate SE(log10(ec)) -> SE(ec)

    residuals = mean_values - _sigmoid_logistic(log_eps, *popt)
    ss_res, ss_tot = np.sum(residuals ** 2), np.sum((mean_values - mean_values.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    extrapolated = not (log_eps.min() <= log_ec <= log_eps.max())
    high_uncertainty = (not np.isfinite(ec_err)) or (ec_err / ec > 0.5 if ec > 0 else True)
    reliable = not (extrapolated or high_uncertainty)

    eps_grid = np.logspace(log_eps.min(), log_eps.max(), 200)
    fit_curve = _sigmoid_logistic(np.log10(eps_grid), *popt)

    return dict(bottom=bottom, top=top, ec=ec, ec_err=ec_err, hill_slope=hill_slope,
                r_squared=r_squared, reliable=reliable, extrapolated=extrapolated,
                eps_grid=eps_grid, fit_curve=fit_curve)

def drop_nan_epsilon(epsilon_arr, mean_arr, std_arr, label=""):
    """
    Drops any epsilon values whose mean is NaN before plotting -- e.g. epsilon=1e-14 in the
    completed Artemia results, which silently went NaN in all 50 trials because eta ended up
    >= L for that dissipation rate (see turbulence_model.py's generate_field_params, which now
    raises instead of letting this happen again for new runs; old .pkl files still carry the
    NaN). Without this, the NaN point is silently skipped by scatter/plot but can distort
    fill_between's shaded band with a visible gap/artifact. Prints what was dropped so it's
    visible, not silent.
    """
    epsilon_arr, mean_arr, std_arr = np.asarray(epsilon_arr), np.asarray(mean_arr), np.asarray(std_arr)
    bad = np.isnan(mean_arr)
    if bad.any():
        print(f"Note: dropping {bad.sum()} NaN epsilon value(s) from {label or 'plot'}: {epsilon_arr[bad]}")
    return epsilon_arr[~bad], mean_arr[~bad], std_arr[~bad]

# Define colors for the three conditions (still, breeze, stormy) for consistent plotting across figures
color_still, color_breeze, color_stormy = '#94c4df', '#4a98c9', '#1764ab'

#%% Plot all three conditions in one figure for comparison

measurement = "Artemia_0805"  # change this to the species folder you want to plot
measurement_name = Path(measurement).name  # used by the plotting cells below

df_still = pickle.load(open(f"plankton_turbulence_results_{measurement_name}_still.pkl", "rb"))
df_breeze = pickle.load(open(f"plankton_turbulence_results_{measurement_name}_breeze.pkl", "rb"))
df_stormy = pickle.load(open(f"plankton_turbulence_results_{measurement_name}_stormy.pkl", "rb"))

epsilon_arr_still = np.array(df_still["epsilon_values"])
epsilon_arr_breeze = np.array(df_breeze["epsilon_values"])
epsilon_arr_stormy = np.array(df_stormy["epsilon_values"])

mean_still = np.array(df_still["mean_upward_velocities"])
mean_breeze = np.array(df_breeze["mean_upward_velocities"])
mean_stormy = np.array(df_stormy["mean_upward_velocities"])

std_still = np.array(df_still["std_upward_velocities"])
std_breeze = np.array(df_breeze["std_upward_velocities"])
std_stormy = np.array(df_stormy["std_upward_velocities"])

control_still = mean_still[0]
control_breeze = mean_breeze[0]
control_stormy = mean_stormy[0]

### Start plot
fig, ax = plt.subplots(figsize=(6, 5))

# Drop any NaN epsilon values before plotting, per condition (see drop_nan_epsilon above).
eps_still_plot, mean_still_plot, std_still_plot = drop_nan_epsilon(epsilon_arr_still[1:], mean_still[1:], std_still[1:], label="still")
eps_breeze_plot, mean_breeze_plot, std_breeze_plot = drop_nan_epsilon(epsilon_arr_breeze[1:], mean_breeze[1:], std_breeze[1:], label="breeze")
eps_stormy_plot, mean_stormy_plot, std_stormy_plot = drop_nan_epsilon(epsilon_arr_stormy[1:], mean_stormy[1:], std_stormy[1:], label="stormy")

# Plot the mean upward velocities with a shaded +/- 1 std band for each condition
# (no per-call labels: color and line-style are explained by two separate legends below)
ax.scatter(eps_still_plot, mean_still_plot, color=color_still)
ax.fill_between(eps_still_plot, mean_still_plot - std_still_plot, mean_still_plot + std_still_plot, color=color_still, alpha=0.2)

ax.scatter(eps_breeze_plot, mean_breeze_plot, color=color_breeze)
ax.fill_between(eps_breeze_plot, mean_breeze_plot - std_breeze_plot, mean_breeze_plot + std_breeze_plot, color=color_breeze, alpha=0.2)

ax.scatter(eps_stormy_plot, mean_stormy_plot, color=color_stormy)
ax.fill_between(eps_stormy_plot, mean_stormy_plot - std_stormy_plot, mean_stormy_plot + std_stormy_plot, color=color_stormy, alpha=0.2)

# Plot control bands for each condition
ax.axhline(control_still, color=color_still, linestyle='--')
ax.axhline(control_breeze, color=color_breeze, linestyle='--')
ax.axhline(control_stormy, color=color_stormy, linestyle='--')

ax.set_xscale('log')
ax.set_xlabel('Dissipation rate [m$^2$/s$^3$]')
ax.set_ylabel('Mean upward velocity of plankton [mm/s]')
ax.set_ylim(-0.5, 1.5)
ax.set_xlim(1e-14, 1e-4)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title('Effect of turbulence on mean vertical swimming speed')

# Two separate legends, proxy handles: one for color (condition), one for line style (with vs
# without turbulence) -- ax.add_artist keeps the first legend when the second legend() is added.
color_handles = [
    Line2D([0], [0], marker='o', color=color_still, linestyle='none', label='Still'),
    Line2D([0], [0], marker='o', color=color_breeze, linestyle='none', label='Breeze'),
    Line2D([0], [0], marker='o', color=color_stormy, linestyle='none', label='Stormy'),
]
style_handles = [
    Line2D([0], [0], marker='o', color='gray', linestyle='none', label='With turbulence'),
    Line2D([0], [0], color='gray', linestyle='--', label='No turbulence (control)'),
]
color_legend = ax.legend(handles=color_handles, title='Condition', loc='upper left')
ax.add_artist(color_legend)
ax.legend(handles=style_handles, title='Line style', loc='lower right')

fig.tight_layout()

#%% Load results for all species x conditions, for the two multi-panel figures below.
# Not every combination has finished running yet -- missing files are skipped with a note
# rather than raising, so this cell (and the panel figures after it) stay usable mid-run.
# Species order: left-to-right panel order in the per-species figures, and legend/print order elsewhere
order = ["Artemia_0805", "Cladocerans_2605", "BarnacleLarvae_2102"]
measurements = order
species_labels = {"Artemia_0805": "Artemia", "BarnacleLarvae_2102": "Barnacle larvae", "Cladocerans_2605": "Cladocerans"}
conditions = ["still", "breeze", "stormy"]

all_results = {}  # all_results[measurement][condition] -> loaded results dict (missing key = not run yet)
for m in measurements:
    all_results[m] = {}
    for cond in conditions:
        path = f"plankton_turbulence_results_{m}_{cond}.pkl"
        try:
            with open(path, "rb") as f:
                all_results[m][cond] = pickle.load(f)
        except FileNotFoundError:
            print(f"Note: no results yet for {species_labels[m]} / {cond} ({path}) -- panel will skip it.")

# Mean swim speed U per species x condition, used for the swimming number Phi = U / u_eta, where
# u_eta = (epsilon * nu)^(1/4) is the Kolmogorov velocity. U is the mean speed of that species'
# tracked trajectories in that condition (the same 'speed' data the turning kernel is built from).
DATA_DIR = Path(__file__).parent / "data"  # experimental trajectory data, data/<species>/ next to this script (as in swimming_behavior.py)
NU_SI = 1e-6                # kinematic viscosity [m^2/s], same value the simulations use

def kolmogorov_velocity_mm_s(epsilon_si):
    """u_eta = (epsilon * nu)^(1/4), returned in mm/s (epsilon in m^2/s^3, nu in m^2/s)."""
    return 1e3 * (np.asarray(epsilon_si, dtype=float) * NU_SI) ** 0.25

def epsilon_at_swimming_number_one(U_mm_s):
    """Dissipation rate [m^2/s^3] where u_eta equals the swim speed U (Phi = 1): U^4 / nu."""
    return (U_mm_s * 1e-3) ** 4 / NU_SI

swim_speed = {}  # swim_speed[measurement][condition] -> mean swim speed U [mm/s]
for m in measurements:
    swim_speed[m] = {}
    for cond in conditions:
        if all_results[m].get(cond) is None:
            continue
        traj = pd.read_pickle(DATA_DIR / m / f"traj_{cond}_scaled.pkl")
        swim_speed[m][cond] = traj.dropna(subset=["speed", "orientation"])["speed"].mean()

print("\nMean swim speed U [mm/s], and dissipation rate where the swimming number is 1 (U^4/nu) [m^2/s^3]:")
for m in measurements:
    print(f"  {species_labels[m]:16s}" + "  ".join(
        f"{c}: U={swim_speed[m][c]:.2f}, eps(Phi=1)={epsilon_at_swimming_number_one(swim_speed[m][c]):.1e}"
        for c in conditions if c in swim_speed[m]))

def _plot_panel_series(ax, epsilon_values, mean_values, std_values, color, label):
    """
    Shared per-series plotting (measured points + fitted response curve + control band) used by
    both panel figures below. Points are the measured data (mean per dissipation rate), plotted
    as dots only and NOT connected by straight lines -- with only ~21 discrete tested epsilon
    values, a line there would imply knowledge of what happens between them. The fitted sigmoid
    (fit_dose_response) is the response curve, so that's what gets drawn as a line. A curve drawn
    at reduced alpha means the fit converged but isn't trustworthy (see `reliable` in
    fit_dose_response) -- typically a series that stayed flat across the whole tested range.

    The transition point (epsilon_c) is marked only for a reliable fit (epsilon_c inside the
    tested range and standard error < 50% of epsilon_c): a shaded vertical band spanning
    epsilon_c +/- its standard error (band width = uncertainty) around a dotted center line. No
    numeric label is drawn on the plot -- the values are reported in the text/console report
    below instead, and what the marker means is spelled out in the bottom legend.

    Returns the fit dict (or None) so the caller can collect it into a summary report.
    """
    eps_plot, mean_plot, std_plot = drop_nan_epsilon(epsilon_values[1:], mean_values[1:], std_values[1:], label=label)
    ax.plot(eps_plot, mean_plot, 'o', color=color)
    ax.fill_between(eps_plot, mean_plot - std_plot, mean_plot + std_plot, color=color, alpha=0.2)

    fit = fit_dose_response(eps_plot, mean_plot, label=label)
    if fit is not None:
        ax.plot(fit['eps_grid'], fit['fit_curve'], '-', color=color, alpha=1.0 if fit['reliable'] else 0.4)
        if fit['reliable']:
            ax.axvspan(fit['ec'] - fit['ec_err'], fit['ec'] + fit['ec_err'], color=color, alpha=0.12, zorder=0)
            ax.axvline(fit['ec'], color=color, linestyle=':', alpha=0.7, linewidth=1.2)

    control_mean, control_std = mean_values[0], std_values[0]
    ax.axhline(control_mean, color=color, linestyle='--')
    ax.axhspan(control_mean - control_std, control_mean + control_std, color=color, alpha=0.15)

    return fit

# Shared x-axis range across every panel in both figures below (including empty "no data yet"
# ones), computed from whatever results are actually loaded rather than hardcoded, so panels are
# directly comparable instead of each autoscaling to its own data (an empty panel previously
# defaulted to matplotlib's 0-1 axis range, e.g. 10^0-10^1, wildly inconsistent with the others).
_all_eps = [np.array(res["epsilon_values"])[1:] for m in measurements for res in
            (all_results[m].get(c) for c in conditions) if res is not None]
epsilon_xlim = (min(a.min() for a in _all_eps), max(a.max() for a in _all_eps)) if _all_eps else (1e-14, 1e-4)

N_TRIALS_NOTE = "mean +/- 1 SD across N=50 trials per dissipation rate"  # adjust N here if it ever changes
PANEL_LETTERS = "ABC"

def _finish_panel_figure(fig, axes, identity_handles, savepath_stem,
                         xlim=None, transition_label=None, bottom_handles=None, bottom_labels=None,
                         ylabel='Speed [mm/s]', panel_letters=True, label_fontsize=None, separate_legends=False):
    """Shared cosmetics + save step for the panel figures: square panels, panel letters inside
    each panel, x-limits, and a single merged legend below the panels -- color identity (condition
    or species) in the first row, mark/line-style meaning in the second. No panel titles --
    journals typically want the image itself title-free, with the per-panel description in the
    external caption instead (printed below for that purpose). Saved as PNG (300 dpi), PDF and SVG.

    xlim: x-axis limits for every panel (defaults to the shared dissipation-rate range; pass
        (larger, smaller) to get an inverted axis). transition_label: legend text for the
        transition-point entry (defaults to the epsilon_c wording). bottom_handles/bottom_labels:
        replace the mark/line-style entries (for figures that don't show a fit or transition point).
        ylabel: y-axis label of the first panel. panel_letters: draw A/B/C inside the panels.
        label_fontsize: font size of the y-axis label (defaults to rcParams' axes.labelsize).
        separate_legends: save the panels without a legend, and the identity and mark/line-style
        legends as two separate figure files (<stem>_legend_identity, <stem>_legend_styles).
    """
    xlim = epsilon_xlim if xlim is None else xlim
    transition_label = transition_label or 'Transition point epsilon_c +/- SE (shown only if within tested range and SE < 50% of epsilon_c)'

    if panel_letters:
        # Headroom at the top (y is shared) so the in-panel letters don't sit on top of the data.
        ymin, ymax = axes[0].get_ylim()
        axes[0].set_ylim(ymin, ymax + 0.12 * (ymax - ymin))

    for ax, letter in zip(axes, PANEL_LETTERS):
        ax.set_xlim(*xlim)
        ax.set_box_aspect(1)
        if panel_letters:
            ax.text(0.04, 0.96, letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    axes[0].set_ylabel(ylabel, fontsize=label_fontsize)

    # Bottom legend: what each mark type means (color identity is in the top legend). The
    # transition-point entry states the condition under which it is drawn at all -- it's a
    # reliability criterion on the fit (not a hypothesis test): epsilon_c must lie inside the
    # tested range and its standard error must be < 50% of epsilon_c (see fit_dose_response).
    style_handles = [
        Line2D([0], [0], marker='o', color='gray', linestyle='none'),
        Line2D([0], [0], color='gray', linestyle='-'),
        (Patch(facecolor='gray', alpha=0.25, edgecolor='none'), Line2D([0], [0], color='gray', linestyle=':')),
        Line2D([0], [0], color='gray', linestyle='--'),
    ]
    style_labels = [
        f'Measured data ({N_TRIALS_NOTE}; shaded band = +/- 1 SD)',
        'Response curve fit (4-parameter logistic)',
        transition_label,
        'No-turbulence control (epsilon=0, mean +/- 1 SD)',
    ]
    if bottom_handles is not None:
        style_handles, style_labels = bottom_handles, bottom_labels

    fig.tight_layout(w_pad=2.5)  # extra gap so the outermost x tick labels of neighbouring panels don't collide
    legend_ax = axes[len(axes) // 2]  # anchored below the middle panel so the legends stay right under the square axes
    legend_kwargs = dict(loc='upper center', fontsize=9, frameon=False, columnspacing=1.5, handletextpad=0.5)

    if separate_legends:
        # The panels are saved without a legend, and each legend as its own figure file
        # (<stem>_legend_identity / <stem>_legend_styles), to be placed freely in the layout.
        _save_panel_figure(fig, savepath_stem)
        _save_separate_legends(fig, identity_handles, style_handles, style_labels, savepath_stem)
        return

    # One merged legend: matplotlib fills legend columns top-to-bottom, so interleave the two
    # groups (padding the shorter one with blank entries) to get identity in row 1, styles in row 2.
    identity_labels = [handle.get_label() for handle in identity_handles]
    ncol = max(len(identity_handles), len(style_handles))
    blank = Line2D([], [], linestyle='none')
    pad = lambda items, fill: list(items) + [fill] * (ncol - len(items))
    legend_handles, legend_labels = [], []
    for h1, l1, h2, l2 in zip(pad(identity_handles, blank), pad(identity_labels, ''),
                              pad(style_handles, blank), pad(style_labels, '')):
        legend_handles += [h1, h2]
        legend_labels += [l1, l2]

    legend_ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(0.5, -0.2), ncol=ncol, **legend_kwargs)
    _save_panel_figure(fig, savepath_stem)

def _save_separate_legends(fig, identity_handles, style_handles, style_labels, savepath_stem):
    """Saves the identity and mark/line-style legends as two separate one-row figure files
    (<stem>_legend_identity, <stem>_legend_styles), as wide as `fig`."""
    for suffix, handles, labels in [("identity", identity_handles, [h.get_label() for h in identity_handles]),
                                    ("styles", style_handles, style_labels)]:
        legend_fig = plt.figure(figsize=(fig.get_figwidth(), 0.4))
        legend_fig.legend(handles=handles, labels=labels, loc='center', ncol=len(handles),
                          fontsize=9, frameon=False, columnspacing=1.5, handletextpad=0.5)
        _save_panel_figure(legend_fig, f"{savepath_stem}_legend_{suffix}")

def _save_panel_figure(fig, savepath_stem):
    """Saves a panel figure as PNG (300 dpi), PDF and SVG."""
    fig.savefig(f'{savepath_stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{savepath_stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{savepath_stem}.svg', bbox_inches='tight')

def _plot_panel_series_data_only(ax, epsilon_values, mean_values, std_values, color, label, U, markersize=None):
    """Measured points, +/- 1 SD band, control mean, and the swimming-number-1 line."""
    eps_plot, mean_plot, std_plot = drop_nan_epsilon(epsilon_values[1:], mean_values[1:], std_values[1:], label=label)
    ax.plot(eps_plot, mean_plot, 'o', color=color, markersize=markersize)
    ax.fill_between(eps_plot, mean_plot - std_plot, mean_plot + std_plot, color=color, alpha=0.2)

    ax.axvline(epsilon_at_swimming_number_one(U), color=color, linestyle='-.', linewidth=1.4, alpha=0.9)

    control_mean = mean_values[0]
    ax.axhline(control_mean, color=color, linestyle='--')

# Legend entries for the mark/line styles of the data-only figures (Figures 1 and 2).
data_only_handles = [
    Line2D([0], [0], marker='o', color='gray', linestyle='none'),
    Line2D([0], [0], color='gray', linestyle='-.'),
    Line2D([0], [0], color='gray', linestyle='--'),
]
data_only_labels = [
    'Measured data (+/- 1 SD)',
    'Swimming number = 1',
    'No-turbulence control',
]

#%% Figure 1: one panel per species (measurement), each panel comparing still/breeze/stormy
# Same style as Figure 2 (measured data only, dash-dotted line where the swimming number is 1),
# with the still/breeze/stormy blues. The logistic fits are still computed -- just not drawn --

# for the epsilon_c report at the end.
condition_colors = {"still": color_still, "breeze": color_breeze, "stormy": color_stormy}
FIG1_WIDTH_INCH = 180 / 25.4
FIG1_LABEL_FONTSIZE = 10
FIG1_MARKERSIZE = 3.5  # matplotlib default is 6

fig, axes = plt.subplots(1, len(measurements), figsize=(FIG1_WIDTH_INCH, 3.4), sharey=True)

fit_results = {}  # fit_results[measurement][condition] -> fit_dose_response() dict, or None

for ax, m in zip(axes, measurements):
    fit_results[m] = {}
    plotted_any = False
    for cond in conditions:
        res = all_results[m].get(cond)
        if res is None:
            continue
        epsilon_values, mean_values = np.array(res["epsilon_values"]), np.array(res["mean_upward_velocities"])
        std_values = np.array(res["std_upward_velocities"])
        _plot_panel_series_data_only(ax, epsilon_values, mean_values, std_values, condition_colors[cond], f"{m}/{cond}",
                                     swim_speed[m][cond], markersize=FIG1_MARKERSIZE)
        eps_fit, mean_fit, _ = drop_nan_epsilon(epsilon_values[1:], mean_values[1:], std_values[1:])
        fit_results[m][cond] = fit_dose_response(eps_fit, mean_fit, label=f"{m}/{cond}")
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "No data yet", ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.set_xscale('log')
    # Unicode superscripts instead of mathtext so the whole label renders in the sans-serif text font
    ax.set_xlabel('Dissipation rate [m²/s³]', fontsize=FIG1_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.7)

condition_handles = [
    Line2D([0], [0], marker='o', color=color_still, linestyle='none', markersize=FIG1_MARKERSIZE, label='Still'),
    Line2D([0], [0], marker='o', color=color_breeze, linestyle='none', markersize=FIG1_MARKERSIZE, label='Breeze'),
    Line2D([0], [0], marker='o', color=color_stormy, linestyle='none', markersize=FIG1_MARKERSIZE, label='Stormy'),
]
# Same style entries as Figure 2, but with the gray dot matching this figure's smaller markers
fig1_style_handles = [Line2D([0], [0], marker='o', color='gray', linestyle='none', markersize=FIG1_MARKERSIZE)] + data_only_handles[1:]
_finish_panel_figure(fig, axes, condition_handles, 'upward_velocity_by_species_panels',
                     bottom_handles=fig1_style_handles, bottom_labels=data_only_labels,
                     panel_letters=False, label_fontsize=FIG1_LABEL_FONTSIZE, separate_legends=True)
print("Suggested caption: Effect of turbulence intensity on mean vertical swimming speed, by species. "
      "Each panel shows one species across the still/breeze/stormy behavioral conditions "
      f"(points: measured {N_TRIALS_NOTE}; shaded bands show +/- 1 SD across trials); dash-dotted vertical "
      "lines: dissipation rate at which the swimming number, mean swim speed / Kolmogorov velocity, equals 1 "
      "for each condition; dashed horizontal lines show each condition's no-turbulence control mean (epsilon=0).")

#%% Figure 2: one panel per condition (treatment), each panel comparing all species
# Distinct from the still/breeze/stormy blues above so the two figures' colors are never confused.
# Measured data only -- no fitted curve. Each series gets a dash-dotted line at the dissipation
# rate where its swimming number is 1, i.e. where the Kolmogorov velocity equals its mean swim
# speed (epsilon = U^4 / nu).
species_palette = {"Artemia_0805": '#2ca02c', "BarnacleLarvae_2102": '#ff7f0e', "Cladocerans_2605": '#9467bd'}

A4_WIDTH_INCH = 210 / 25.4
fig, axes = plt.subplots(1, len(conditions), figsize=(A4_WIDTH_INCH, 3.4), sharey=True)

for ax, cond in zip(axes, conditions):
    plotted_any = False
    for m in measurements:
        res = all_results[m].get(cond)
        if res is None:
            continue
        _plot_panel_series_data_only(ax, np.array(res["epsilon_values"]), np.array(res["mean_upward_velocities"]),
                            np.array(res["std_upward_velocities"]), species_palette[m], f"{m}/{cond}", swim_speed[m][cond])
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "No data yet", ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Dissipation rate [m$^2$/s$^3$]')
    ax.grid(True, linestyle='--', alpha=0.7)

species_handles = [Line2D([0], [0], marker='o', color=species_palette[m], linestyle='none', label=species_labels[m]) for m in measurements]
_finish_panel_figure(fig, axes, species_handles, 'upward_velocity_by_condition_panels', bottom_handles=data_only_handles, bottom_labels=data_only_labels)

print("\n swimming numer for each species x condition (dissipation rate [m^2/s^3] where the Kolmogorov velocity equals the mean swim speed):")
for m in measurements:
    print(f"  {species_labels[m]:16s}" + "  ".join(
        f"{c}: eps(Phi=1)={epsilon_at_swimming_number_one(swim_speed[m][c]):.1e}"
        for c in conditions if c in swim_speed[m]))

print("Suggested caption: Effect of turbulence intensity on mean vertical swimming speed, by condition. "
      f"Each panel compares species under one behavioral condition (points: measured {N_TRIALS_NOTE}; "
      "dash-dotted vertical lines: dissipation rate at which the swimming number, mean swim speed / Kolmogorov "
    "velocity, equals 1 for each species; shaded bands show +/- 1 SD across trials; dashed horizontal lines show "
    "each species' no-turbulence control mean (epsilon=0).")


#%% Report: critical dissipation rate (epsilon_c) from the fitted dose-response curve, per measurement
# epsilon_c is where the fitted sigmoid crosses its own inflection point -- the dissipation rate
# at which turbulence has disrupted vertical swimming halfway between the calm-water behavior and
# the fully-randomized (turbulence-dominated) behavior. "Unreliable" entries mean curve_fit
# converged but epsilon_c fell outside the tested range or came back with >50% relative
# uncertainty -- in practice, a series that never actually turned over within the epsilon values
# tested (e.g. a condition where the response stayed close to flat throughout).
print("\n=== Critical dissipation rate (epsilon_c) from the fitted dose-response curve ===")
for m in measurements:
    print(f"\n{species_labels[m]}:")
    for cond in conditions:
        fit = fit_results.get(m, {}).get(cond)
        if fit is None:
            print(f"  {cond:7s}: no data yet, or the fit did not converge")
        elif not fit['reliable']:
            reason = "epsilon_c fell outside the tested range" if fit['extrapolated'] else "relative uncertainty on epsilon_c > 50%"
            print(f"  {cond:7s}: epsilon_c ~ {fit['ec']:.2e} m^2/s^3 -- UNRELIABLE ({reason}); "
                  "likely no clear transition within the epsilon values tested")
        else:
            print(f"  {cond:7s}: epsilon_c = {fit['ec']:.2e} +/- {fit['ec_err']:.1e} m^2/s^3  "
                  f"(hill slope={fit['hill_slope']:.2f}, R^2={fit['r_squared']:.3f})")

#%% Figure 4: circular variance of heading vs dissipation rate, one panel per species
# Same layout and formatting as Figure 1 (still/breeze/stormy overlaid per species, measured data
# only, dash-dotted line where the swimming number is 1, legends saved as separate files), but
# y = circular variance V = 1 - R of the swimming direction (0 = all animals swim the same way,
# 1 = uniformly random). Next to Figure 1 it shows whether the drop in upward speed comes from
# orientation being randomized: if so, V should rise over the same epsilon range where the speed
# falls. Results from runs before circular variance was saved don't have it -- those
# species/conditions are skipped with a note (panel shows "No data yet").
#
# USE_WEIGHTED_V switches between the plain circular variance V and the speed-weighted V_w
# (each step's direction weighted by its speed; see speed_weighted_circular_variance in
# coupled_simulation.py). V_w is only in results from runs after it was added.
USE_WEIGHTED_V = True  # set to False to plot the plain circular variance V instead of the speed-weighted V_w

if USE_WEIGHTED_V:
    cv_key, cv_ylabel, cv_stem = "weighted_circular_variances", 'Circular variance', 'weighted_circular_variance_by_species_panels'
    cv_caption_name = "speed-weighted circular variance (each step's direction weighted by its speed)"
else:
    cv_key, cv_ylabel, cv_stem = "circular_variances", 'Circular variance', 'circular_variance_by_species_panels'
    cv_caption_name = "circular variance"

fig, axes = plt.subplots(1, len(measurements), figsize=(FIG1_WIDTH_INCH, 3.4), sharey=True)

for ax, m in zip(axes, measurements):
    plotted_any = False
    for cond in conditions:
        res = all_results[m].get(cond)
        if res is None:
            continue
        if f"mean_{cv_key}" not in res:
            print(f"Note: {species_labels[m]} / {cond} has no {cv_caption_name} saved (old run) -- rerun to include it.")
            continue
        _plot_panel_series_data_only(ax, np.array(res["epsilon_values"]), np.array(res[f"mean_{cv_key}"]),
                                     np.array(res[f"std_{cv_key}"]), condition_colors[cond], f"{m}/{cond} ({cv_key})",
                                     swim_speed[m][cond], markersize=FIG1_MARKERSIZE)
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "No data yet", ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Dissipation rate [m²/s³]', fontsize=FIG1_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.7)

_finish_panel_figure(fig, axes, condition_handles, cv_stem,
                     bottom_handles=fig1_style_handles, bottom_labels=data_only_labels,
                     ylabel=cv_ylabel, panel_letters=False, label_fontsize=FIG1_LABEL_FONTSIZE,
                     separate_legends=True)
print(f"Suggested caption: Effect of turbulence intensity on the {cv_caption_name} of swimming direction, by species. "
      "Each panel shows one species across the still/breeze/stormy behavioral conditions "
      f"(points: {N_TRIALS_NOTE}; shaded bands show +/- 1 SD across trials); dash-dotted vertical lines: "
      "dissipation rate at which the swimming number, mean swim speed / Kolmogorov velocity, equals 1 for each "
      "condition; dashed horizontal lines show each condition's no-turbulence control mean (epsilon=0). "
      "V = 0: all headings identical; V = 1: headings uniformly random.")

#%% Figure 5: Figures 1 and 4 combined -- speed (top row) and circular variance (bottom row) per species
# One column per species with the species name on top; the two panels in a column share the
# dissipation-rate axis, so a drop in upward speed can be read off directly against the rise in
# circular variance at the same epsilon. Same formatting as Figures 1 and 4, legends saved as
# separate files (they are the same as Figure 1's). Plain or speed-weighted circular variance
# follows USE_WEIGHTED_V in the Figure 4 cell (run that cell first after changing it).
species_title_style = {"Artemia_0805": 'italic'}  # genus names in italics; the others are common names

rows = [("mean_upward_velocities", "std_upward_velocities", "Speed [mm/s]"),
        (f"mean_{cv_key}", f"std_{cv_key}", cv_ylabel)]
fig5_stem = 'speed_and_weighted_circular_variance_by_species_panels' if USE_WEIGHTED_V else 'speed_and_circular_variance_by_species_panels'

fig, axes = plt.subplots(2, len(measurements), figsize=(FIG1_WIDTH_INCH, 4.6), sharex=True, sharey='row')

for col, m in enumerate(measurements):
    axes[0, col].set_title(species_labels[m], fontsize=FIG1_LABEL_FONTSIZE, fontstyle=species_title_style.get(m, 'normal'))
    for row, (mean_key, std_key, _) in enumerate(rows):
        ax = axes[row, col]
        plotted_any = False
        for cond in conditions:
            res = all_results[m].get(cond)
            if res is None or mean_key not in res:
                continue
            _plot_panel_series_data_only(ax, np.array(res["epsilon_values"]), np.array(res[mean_key]), np.array(res[std_key]),
                                         condition_colors[cond], f"{m}/{cond} ({mean_key})", swim_speed[m][cond],
                                         markersize=FIG1_MARKERSIZE)
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No data yet", ha='center', va='center', transform=ax.transAxes, color='gray')

        ax.set_xscale('log')
        ax.set_xlim(*epsilon_xlim)
        ax.set_box_aspect(1)
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[-1, col].set_xlabel('Dissipation rate [m²/s³]', fontsize=FIG1_LABEL_FONTSIZE)

for row, (_, _, ylabel) in enumerate(rows):
    axes[row, 0].set_ylabel(ylabel, fontsize=FIG1_LABEL_FONTSIZE)

fig.tight_layout(w_pad=2.5, h_pad=0.8)
_save_panel_figure(fig, fig5_stem)
_save_separate_legends(fig, condition_handles, fig1_style_handles, data_only_labels, fig5_stem)
print(f"Suggested caption: Effect of turbulence intensity on mean vertical swimming speed (top) and on the {cv_caption_name} "
      "of swimming direction (bottom), by species. Each column shows one species across the still/breeze/stormy "
      f"behavioral conditions (points: {N_TRIALS_NOTE}; shaded bands show +/- 1 SD across trials); dash-dotted vertical "
      "lines: dissipation rate at which the swimming number, mean swim speed / Kolmogorov velocity, equals 1 for each "
      "condition; dashed horizontal lines show each condition's no-turbulence control mean (epsilon=0). "
      "Circular variance V = 0: all headings identical; V = 1: headings uniformly random.")

#%% Figure 6: upward alignment <sin(theta)> vs dissipation rate, one panel per species
# Mean vertical component of the frame-to-frame swimming direction theta (theta = 90 deg is
# straight up): +1 = every step straight up, 0 = no net upward preference (random, or as much up
# as down), -1 = every step straight down. Unlike circular variance it isn't fooled by up-and-down
# (bimodal) swimming, as in the barnacle larvae, where up and down cancel and V looks near-random.
# Computed from the saved orientation histograms (bin centers; the binning error is < 0.1%), per
# trial, then mean +/- SD across trials as in the other figures. Same formatting as Figure 1.
def upward_alignment_per_trial(res):
    """<sin(theta)> per epsilon and trial, shape (n_eps, n_trials), from orientation_counts_full."""
    edges = np.asarray(res["orientation_bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    counts = np.asarray(res["orientation_counts_full"], dtype=float)  # (n_eps, n_trials, n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):  # a trial with no kept steps -> NaN
        return (counts * np.sin(centers)).sum(-1) / counts.sum(-1)

fig, axes = plt.subplots(1, len(measurements), figsize=(FIG1_WIDTH_INCH, 3.4), sharey=True)

for ax, m in zip(axes, measurements):
    plotted_any = False
    for cond in conditions:
        res = all_results[m].get(cond)
        if res is None:
            continue
        if "orientation_counts_full" not in res:
            print(f"Note: {species_labels[m]} / {cond} has no orientation histograms saved (old run) -- rerun to include it.")
            continue
        alignment = upward_alignment_per_trial(res)
        _plot_panel_series_data_only(ax, np.array(res["epsilon_values"]), np.nanmean(alignment, axis=1),
                                     np.nanstd(alignment, axis=1, ddof=1), condition_colors[cond], f"{m}/{cond} (alignment)",
                                     swim_speed[m][cond], markersize=FIG1_MARKERSIZE)
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "No data yet", ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Dissipation rate [m²/s³]', fontsize=FIG1_LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.7)

_finish_panel_figure(fig, axes, condition_handles, 'upward_alignment_by_species_panels',
                     bottom_handles=fig1_style_handles, bottom_labels=data_only_labels,
                     ylabel='Upward alignment (mean sin θ)', panel_letters=False, label_fontsize=FIG1_LABEL_FONTSIZE,
                     separate_legends=True)
print("Suggested caption: Effect of turbulence intensity on the upward alignment of swimming direction, by species. "
      "Upward alignment is the mean vertical component <sin(theta)> of the frame-to-frame swimming direction theta "
      "(1: all steps straight up; 0: no net upward preference; -1: all steps straight down). Each panel shows one "
      f"species across the still/breeze/stormy behavioral conditions (points: {N_TRIALS_NOTE}; shaded bands show "
      "+/- 1 SD across trials); dash-dotted vertical lines: dissipation rate at which the swimming number, mean swim "
      "speed / Kolmogorov velocity, equals 1 for each condition; dashed horizontal lines show each condition's "
      "no-turbulence control mean (epsilon=0).")

# %%
