#%% Load packages
import numpy as np
import pandas as pd
from post_processing_functions import *
import os
import pickle
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

def mean_dy_per_particle(df_treatment, tank_id):
    df_tank = df_treatment[df_treatment['tank'] == tank_id]
    df_particle = df_tank.groupby('particle', as_index=False).agg(
        dy=('dy', 'mean'),
        nSpots=('dy', 'size'),
    )
    return np.average(df_particle['dy'], weights=df_particle['nSpots'])


def compact_letter_display(group_names, reject_lookup, group_means):
    ordered_names = sorted(group_names, key=lambda name: group_means[name], reverse=True)
    name_to_index = {name: idx for idx, name in enumerate(ordered_names)}
    nonsig_pairs = {
        tuple(sorted((name_to_index[a], name_to_index[b])))
        for a, b in combinations(ordered_names, 2)
        if not reject_lookup[frozenset((a, b))]
    }

    universe = set(range(len(ordered_names))) | nonsig_pairs
    candidate_subsets = []

    for subset_size in range(1, len(ordered_names) + 1):
        for subset in combinations(range(len(ordered_names)), subset_size):
            if all(tuple(sorted(pair)) in nonsig_pairs for pair in combinations(subset, 2)):
                cover: set[object] = set(subset)
                cover.update({tuple(sorted(pair)) for pair in combinations(subset, 2)})
                candidate_subsets.append((subset, cover))

    candidate_subsets.sort(key=lambda item: (-len(item[1]), len(item[0]), item[0]))

    best_solution = None

    def search(covered, chosen):
        nonlocal best_solution
        if best_solution is not None and len(chosen) >= len(best_solution):
            return
        if covered == universe:
            best_solution = list(chosen)
            return

        target = next(iter(universe - covered))
        for subset, cover in candidate_subsets:
            if target in cover:
                search(covered | cover, chosen + [subset])

    search(set(), [])

    letters = {}
    for idx, name in enumerate(ordered_names):
        letters[name] = ""

    if best_solution is None:
        return {name: "a" for name in group_names}

    alphabet = [chr(ord('a') + i) for i in range(26)]
    for letter_idx, subset in enumerate(best_solution):
        letter = alphabet[letter_idx]
        for idx in subset:
            letters[ordered_names[idx]] += letter

    return letters


def tukey_posthoc(group_names, group_values):
    combined_values = np.concatenate([np.asarray(values) for values in group_values])
    combined_labels = np.concatenate([
        np.repeat(name, len(values))
        for name, values in zip(group_names, group_values)
    ])

    tukey = pairwise_tukeyhsd(combined_values, combined_labels, alpha=0.05)

    reject_lookup = {}
    pvalue_lookup = {}
    for row in tukey.summary().data[1:]:
        group_a, group_b, meandiff, p_adj, lower, upper, reject = row
        key = frozenset((group_a, group_b))
        reject_lookup[key] = bool(reject)
        pvalue_lookup[key] = float(p_adj)

    group_means = {
        name: float(np.mean(values))
        for name, values in zip(group_names, group_values)
    }
    letters = compact_letter_display(group_names, reject_lookup, group_means)

    summary_rows = []
    for row in tukey.summary().data[1:]:
        group_a, group_b, meandiff, p_adj, lower, upper, reject = row
        summary_rows.append({
            'group1': group_a,
            'group2': group_b,
            'meandiff': meandiff,
            'p_adj': p_adj,
            'lower': lower,
            'upper': upper,
            'reject': reject,
        })

    return group_means, letters, pd.DataFrame(summary_rows)

#%% Define treatment, load data and calculate scale
# measurement = 'AcatiaTonsa_2601'
# measurement = 'Artemia_0805'
# measurement = 'Copepods_2407'
# measurement = 'BarnacleLarvae_2102'  
measurement = 'Cladocerans_2605'

# Define path to data and where to save data
path_to_tracking_data = "R:\\LU24A1047-PLS\\TrackingData\\" #Research data folder
path_to_tracking_results = os.path.join(path_to_tracking_data, measurement, 'tracking_results')
path_to_save_scaledtraj = os.path.join(path_to_tracking_data, measurement, 'trajectories_scaled')
os.makedirs(path_to_save_scaledtraj, exist_ok=True)

### calculate scale
if measurement == 'Artemia_0805':
    path_raw_avi = os.path.join(path_to_tracking_data, "Artemia_0805\\resized_videos\\still1.avi")
    path_cropped_avi = os.path.join(path_to_tracking_data, "Artemia_0805\\cropped_videos\\still1.avi")
    # cropped_frame = load_frames(path_cropped_avi,1)[0]
    h_cropped_pxl, w_cropped_pxl = 1280, 853 # cropped_frame.shape # Vertical videos, width is the smaller dimension
    # raw_frame = load_frames(path_raw_avi,1)[0]
    w_raw_pxl, h_raw_pxl = 853, 1280  # Vertical videos, width is the smaller dimension
    w_raw_mm = 70.6 # Width of the raw video [mm]
    binning = 1 # binning factor in post processing
    time_start = 150  # Start time [frames]
    time_end =  10000 # End time [frames], max = 684
    min_frames = 30  # Minimum number of frames to keep a trajectory
    tanks = ['1', '2', '3']

elif measurement == 'Copepods_2407':
    path_raw_avi = os.path.join(path_to_tracking_data, "Copepods_2407\\raw_videos\\still1\\img_1.png")
    path_cropped_avi = os.path.join(path_to_tracking_data, "Copepods_2407\\cropped_videos\\still1.avi")
    # cropped_frame = load_frames(path_cropped_avi,1)[0]
    h_cropped_pxl, w_cropped_pxl = 1136, 1732
    # raw_frame = plt.imread(path_raw_avi)
    h_raw_pxl, w_raw_pxl = 4512, 4512 # Horizontal videos, width is the larger dimension
    w_raw_mm = 103.7 # Width of the raw video [mm]
    binning = 2 # binning factor in post processing
    time_start = 0  # Start time [frames]
    time_end =  100000 # End time [frames], max = 794
    min_frames = 30  # Minimum number of frames to keep a trajectory
    tanks = ['1', '2', '3']

elif measurement == 'BarnacleLarvae_2102':
    path_raw_avi = os.path.join(path_to_tracking_data, os.path.join(measurement, "raw_videos", "still1.avi"))
    path_cropped_avi = os.path.join(path_to_tracking_data, os.path.join(measurement, "cropped_videos", "still1.avi"))
    cropped_frame = load_frames(path_cropped_avi,1)[0]
    h_cropped_pxl, w_cropped_pxl = 630, 930 #cropped_frame.shape
    # raw_frame = load_frames(path_raw_avi,1)[0]
    w_raw_pxl, h_raw_pxl = 3648, 5472 #raw_frame.shape # Vertical videos, width is the smaller dimension
    w_raw_mm = 110 # Width of the raw video [mm]
    binning = 4 # binning factor in post processing
    time_start = 0  # Start time [frames]
    time_end =  10000 # End time [frames], max = 1041
    min_frames = 30  # Minimum number of frames to keep a trajectory
    tanks = ['1', '2', '3']

elif measurement == 'AcatiaTonsa_2601':
    path_raw_avi = os.path.join(path_to_tracking_data, os.path.join(measurement, "raw_videos", "still1.avi"))
    path_cropped_avi = os.path.join(path_to_tracking_data, os.path.join(measurement, "cropped_videos", "still1.png"))
    # cropped_frame = plt.imread(path_cropped_avi)
    h_cropped_pxl, w_cropped_pxl = 2217, 2256 #cropped_frame.shape
    # raw_frame = load_frames(path_raw_avi,1)[0]
    h_raw_pxl, w_raw_pxl = 2256, 2256 #raw_frame.shape # Vertical videos, width is the smaller dimension
    w_raw_mm = 95 # Width of the raw video [mm]
    binning = 1 # binning factor in post processing
    time_start = 0  # Start time [frames]
    time_end = 10000 #10000 # End time [frames], max = 1041
    min_frames = 30  # Minimum number of frames to keep a trajectory
    tanks = ['1', '2', '3'] # number of replicates
    
elif measurement == 'Cladocerans_2605':
    path_raw_avi = os.path.join(path_to_tracking_data, "Cladocerans_2605\\resized_videos\\still1.avi")
    path_cropped_avi = os.path.join(path_to_tracking_data, "Cladocerans_2605\\cropped_videos\\still1.avi")
    # cropped_frame = load_frames(path_cropped_avi,1)[0]
    h_cropped_pxl, w_cropped_pxl = 533, 549 #cropped_frame.shape
    # raw_frame = load_frames(path_raw_avi,1)[0]
    h_raw_pxl, w_raw_pxl = 2256, 2256 #raw_frame.shape # Vertical videos, width is the smaller dimension
    w_raw_mm = 90.6 # Width of the raw video [mm]
    binning = 4 # binning factor in post processing
    time_start = 0  # Start time [frames]
    time_end =  10000 # End time [frames], max = 684
    min_frames = 30  # Minimum number of frames to keep a trajectory
    tanks = ['1', '2', '3', '4', '5'] # number of replicates
    
else:
    raise ValueError("Measurement not recognized.")

scale = w_raw_mm / w_raw_pxl  # scale in mm/pxl (equivalent to mm per binned pixel)
scale = scale * binning # adjust scale for binning factor

# Set parameters for cleaning up the data
# Minimum movement in mm to consider a trajectory non-stationary [mm]
min_movement = 0.3 # [mm]

# Movement to consider something an artifact
artifact_treshold = 0.0001 #[mm]

frame_rate = 10  # frames per second for conversion to mm/s

# Preprocessing can cover the full recording while the final statistics can
# be computed on a narrower analysis window chosen later.
analysis_time_start = time_start
analysis_time_end = time_end

treatments = ['still', 'breeze', 'stormy', 'nothing']

### Print cutoff conditions 
# print(f"=== Data cleaning parameters for {measurement} ===")
# print(f"Time window: {time_start} to {time_end} frames")
# print(f"Scale: {scale:.3f} mm/pxl")
# print(f"Artifact threshold: {artifact_treshold:.4f} mm / {artifact_treshold/scale:.4f} pxls")
# print(f"Minimum movement to consider non-stationary: {min_movement:.2f} mm / {min_movement/scale:.2f} pxls")
# print(f"Minimum trajectory length: {min_frames} frames/ {min_frames/frame_rate:.1f} seconds")

# print(f"\n=== Metadata for loaded data for {measurement} ===")
# print(f"time window: {time_start} to {time_end} frames")
# print(f"WxH raw video: {w_raw_pxl} x {h_raw_pxl} pxl", f"({w_raw_mm} mm width)")
# print(f"WxH cropped video: {cropped_frame.shape[1]} x {cropped_frame.shape[0]} pxl")
# print(f"Binning factor: {binning}")
# print(f"Scale: {scale:.3f} mm/pxl")

# Plot 10 frames with spots to make sure laoded correctly
treatment = 'still1'
path_xml = os.path.join(path_to_tracking_results, treatment + ".xml")
path_avi = os.path.join(path_to_tracking_data, measurement, "cropped_videos", treatment + ".avi")

# Load data
# df = load_xml(path_xml)
# frames = load_frames(path_avi)

# num_frames = len(frames)
# indices = np.random.randint(0, num_frames, size=10)

# w,h = frames[1].shape

# for i in indices:
#     print(['figure ' + str(i) + '/' + str(len(frames))])
#     frame = frames[i]
 
#     df_current_frame = df[df['t'] == i]
#     x_pos = df_current_frame['x'].values
#     y_pos = df_current_frame['y'].values

#     plt.figure()
#     plt.imshow(frame, cmap='gray')
#     plt.scatter(x_pos, y_pos, s=25, edgecolors='r', facecolors='none')
#     plt.show()
#     plt.close()  # Close the plot window before moving to the next frame


# Load data

#############################
## Load all data per frame ##
#############################

dfs=[None]*len(treatments)
organisms_per_tank = []

for i, treatment in enumerate(treatments):
    data_perframe = []

    for tank in tanks:
        print(f"\rProcessing {treatment} {tank} ...", end='\r', flush=True)
        
        # define path to xml file
        path_xml = os.path.join(path_to_tracking_results, treatment + tank + ".xml")           
        df_tank = load_xml(path_xml)   
        
        organisms_tank = len(df_tank[df_tank['t'] == 0])*2 # number of organisms in the tank at t=0
        organisms_per_tank.append(organisms_tank)
        
        # Scale x and y positions [pxls -> mm]
        df_tank['x'] = df_tank['x'].astype(float) * scale
        df_tank['y'] = (h_cropped_pxl - df_tank['y'].astype(float) - 1) * scale    
        
        # 1. Load and scale full data 
        df_tank = df_tank.sort_values(['particle', 't']).reset_index(drop=True)

        # 2. Compute dx/dy on full trajectory
        df_tank['dt'] = df_tank.groupby('particle')['t'].diff()
        df_tank['dx'] = df_tank.groupby('particle')['x'].diff() / df_tank['dt'] * frame_rate
        df_tank['dy'] = df_tank.groupby('particle')['y'].diff() / df_tank['dt'] * frame_rate

        # Drop rows with NaN values in dx, dy, dt (first entry of each trajectory and any trajectories with only one spot)
        df_tank = df_tank.dropna(subset=['dx', 'dy', 'dt']).reset_index(drop=True)
        
        # 3. Filter time window
        if treatment != 'nothing':  # For control treatment, keep full time window for analysis
            df_tank = slice_time_window(df_tank, time_start, time_end)
        
        df_tank = df_tank.reset_index(drop=True)
        
        # 4. remove artifacts
        x_diff = df_tank.groupby('particle', sort=False)['x'].diff().abs()
        y_diff = df_tank.groupby('particle', sort=False)['y'].diff().abs()
        df_tank = df_tank[(x_diff >= artifact_treshold) & (y_diff >= artifact_treshold)]
                
        ### 5. Fill gaps in trajectories ###
        df_tank = fill_gaps_linear(df_tank, verbose = False)
 
 
        ### 6. if acacia, remove all trajectories wehere the total dy is negative
        if measurement == 'AcatiaTonsa_2601':
            dy = df_tank.groupby('particle')['y'].last() - df_tank.groupby('particle')['y'].first()
            traj_to_remove = dy[dy < 0].index
            df_tank = df_tank[~df_tank['particle'].isin(traj_to_remove)].reset_index(drop=True)
            
        # Calculate orientation and speed
        df_tank['orientation'] = np.arctan2(df_tank['dy'], df_tank['dx']) #[radians]
        df_tank['orientation_degrees'] = np.degrees(df_tank['orientation']) #[degrees]
        df_tank['speed'] = np.sqrt(df_tank['dx'] ** 2 + df_tank['dy'] ** 2) #[mm/s]  
        
        ### Clean data by removing particles based on criteria
        # 2. Remove stationary trajectories
        stat_ids = count_stationary_traj(df_tank, threshold=min_movement)
        if stat_ids:
            df_tank = df_tank[~df_tank['particle'].isin(stat_ids)].reset_index(drop=True)

        df_tank['tank'] = int(tank)  # Add tank number as integer column

        # 4. Remove short trajectories
        df_tank['nSpots'] = df_tank.groupby('particle')['t'].transform('count')
        df_tank = df_tank[df_tank['nSpots'] >= min_frames].reset_index(drop=True)

        data_perframe.append(df_tank) # Dataframe after processing with all trajectories per frame
        
    dfs[i] = pd.concat(data_perframe, ignore_index=True)


# Save scaled and processed data into pickle files for plotting scripts    
for i, df in enumerate(dfs):
    df.to_pickle(os.path.join(path_to_save_scaledtraj, f'traj_{treatments[i]}_scaled.pkl'))

# calculate shortest measurement to set maximum end time for analysis
min_end_time = float('inf')
for df in dfs[:-1]:  # Exclude control treatment which has full time window
    for tank in tanks:
        end_time = df[df['tank'] == int(tank)]['t'].max()
        if end_time < min_end_time:
            min_end_time = end_time

if analysis_time_end > min_end_time:
    # print(f"Warning: analysis_time_end ({analysis_time_end} frames) is greater than the maximum time available in the data ({min_end_time} frames). Setting analysis_time_end to {min_end_time} frames.")
    analysis_time_end = min_end_time

### Print statistics
# Calculate per-frame means for ANOVA later
df_still_perframe = slice_time_window(dfs[0], analysis_time_start, analysis_time_end)
df_breeze_perframe = slice_time_window(dfs[1], analysis_time_start, analysis_time_end)
df_stormy_perframe = slice_time_window(dfs[2], analysis_time_start, analysis_time_end)
df_control_perframe = slice_time_window(dfs[3], 0, analysis_time_end)

still_means_perframe = [np.mean(df_still_perframe['dy'][df_still_perframe['tank'] == int(tank)]) for tank in tanks]
breeze_means_perframe = [np.mean(df_breeze_perframe['dy'][df_breeze_perframe['tank'] == int(tank)]) for tank in tanks]
stormy_means_perframe = [np.mean(df_stormy_perframe['dy'][df_stormy_perframe['tank'] == int(tank)]) for tank in tanks]
control_means_perframe = [np.mean(df_control_perframe['dy'][df_control_perframe['tank'] == int(tank)]) for tank in tanks]    

# Calculate means and 95% CI for each treatment
means_perframe = [np.mean(still_means_perframe), np.mean(breeze_means_perframe), np.mean(stormy_means_perframe), np.mean(control_means_perframe)]
STD_perframe = [np.std(still_means_perframe, ddof=1), np.std(breeze_means_perframe, ddof=1), np.std(stormy_means_perframe, ddof=1), np.std(control_means_perframe, ddof=1)]
n_replicates = len(tanks)
t_critical = sp.stats.t.ppf(0.975, df=n_replicates - 1)
CI95_perframe = [STD_perframe[i] / np.sqrt(n_replicates) * t_critical for i in range(len(treatments))]

# Calculate circular variance (V) for orientation data for each treatment
def V(angles):
    # Drop NaN values from the angles array
    angles = angles[~np.isnan(angles)]
    R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
    V = 1 - R
    return V

still_V_perframe = [V(df_still_perframe['orientation'][df_still_perframe['tank'] == int(tank)].values) for tank in tanks]
breeze_V_perframe = [V(df_breeze_perframe['orientation'][df_breeze_perframe['tank'] == int(tank)].values) for tank in tanks]
stormy_V_perframe = [V(df_stormy_perframe['orientation'][df_stormy_perframe['tank'] == int(tank)].values) for tank in tanks]
control_V_perframe = [V(df_control_perframe['orientation'][df_control_perframe['tank'] == int(tank)].values) for tank in tanks]

mean_V_perframe = [np.mean(still_V_perframe), np.mean(breeze_V_perframe), np.mean(stormy_V_perframe), np.mean(control_V_perframe)]
STD_V_perframe = [np.std(still_V_perframe, ddof=1), np.std(breeze_V_perframe, ddof=1), np.std(stormy_V_perframe, ddof=1), np.std(control_V_perframe, ddof=1)]
CI95_V_perframe = [STD_V_perframe[i] / np.sqrt(n_replicates) * t_critical for i in range(len(treatments))]

# ANOVA: per-tank mean dy across still, breeze, stormy
f_stat_perframe, p_value_perframe = sp.stats.f_oneway(still_means_perframe, breeze_means_perframe, stormy_means_perframe)
f_stat_V_perframe, p_value_V_perframe = sp.stats.f_oneway(still_V_perframe, breeze_V_perframe, stormy_V_perframe)

# Print results
# print(f"=== Statistical results for {measurement} ===")
# print(f"\nCalculated per frame ")
# print metadata used to obtain these result
print(f"\n=== Statistical results for {measurement} ===")
print(f"Analysis window: {analysis_time_start} to {analysis_time_end} frames")
print(f"Artifact threshold: {artifact_treshold:.4f} mm / {artifact_treshold/scale:.4f} pxls")
print(f"Minimum movement to consider non-stationary: {min_movement:.2f} mm / {min_movement/scale:.2f} pxls")
print(f"Minimum trajectory length: {min_frames} frames / {min_frames/frame_rate:.1f} seconds")

# Print results
print(f"\n=== ANOVA results for {measurement} ===")
print(f"P-value dy: {p_value_perframe:.4f}")
print(f"P-value V: {p_value_V_perframe:.4f}")

group_means_perframe, letters_perframe, tukey_table_perframe = tukey_posthoc(
    treatments,
    [still_means_perframe, breeze_means_perframe, stormy_means_perframe, control_means_perframe],
)

group_means_V_perframe, letters_V_perframe, tukey_table_V_perframe = tukey_posthoc(
    treatments,
    [still_V_perframe, breeze_V_perframe, stormy_V_perframe, control_V_perframe],
)

print("\n=== Mean dy (per frame)===")
for i, treatment in enumerate(treatments):
    # print(f"{treatment.capitalize()}: Mean dy = {means_perframe[i]:.2f} ± {CI95_perframe[i]:.2f} mm/s  [{letters_perframe[treatment]}], V = {mean_V_perframe[i]:.2f} ± {CI95_V_perframe[i]:.2f}")
    # print with STD instead of CI95
    print(f"{treatment.capitalize()}: Mean dy = {means_perframe[i]:.2f} ± {STD_perframe[i]:.2f} mm/s  [{letters_perframe[treatment]}], V = {mean_V_perframe[i]:.2f} ± {STD_V_perframe[i]:.2f}")

print(f"mean number of organisms per tank: {np.mean(organisms_per_tank):.1f} ± {np.std(organisms_per_tank, ddof=1):.1f}")
    
# print("\n=== replicates: ===")
# print(f"Still:")
# for tank in tanks:
#     print(f"Tank {tank}: {still_means_perframe[int(tank)-1]:.3f} mm/s")
    
# print(f"Breeze:")
# for tank in tanks:
#     print(f"Tank {tank}: {breeze_means_perframe[int(tank)-1]:.3f} mm/s")
    
# print(f"Stormy:")
# for tank in tanks:
#     print(f"Tank {tank}: {stormy_means_perframe[int(tank)-1]:.3f} mm/s")
    
# print(f"Control:")
# for tank in tanks:
#     print(f"Tank {tank}: {control_means_perframe[int(tank)-1]:.3f} mm/s")
#%% #################################
# Load all data per trajectory ##
#################################
dfs_traj: list[pd.DataFrame] = [pd.DataFrame() for _ in treatments]

for i, treatment in enumerate(treatments):
    data_pertraj = []
    for tank in tanks:
        print(f"\rProcessing {treatment} {tank} ...", end='\r', flush=True)
        
        # define path to xml file
        path_xml = os.path.join(path_to_tracking_results, treatment + tank + ".xml")           
        df_tank = load_xml(path_xml)   
        
        # Scale x and y positions [pxls -> mm]
        df_tank['x'] = df_tank['x'].astype(float) * scale
        df_tank['y'] = (h_cropped_pxl - df_tank['y'].astype(float) - 1) * scale    
        
        # 1. Filter time window
        if treatment != 'nothing':  # For control treatment, keep full time window for analysis 
            df_tank = slice_time_window(df_tank, time_start, time_end)
        df_tank = df_tank.reset_index(drop=True)
        
        # 2. Remove stationary trajectories
        stat_ids = count_stationary_traj(df_tank, threshold=min_movement)
        if stat_ids:
            df_tank = df_tank[~df_tank['particle'].isin(stat_ids)].reset_index(drop=True)
        
        ### Calculate df for entire trajectories        
        # Create dataframe with one entry per trajectory
        df_tank_traj = df_tank.groupby('particle', as_index=False).agg({'x': ['first', 'last'], 'y': ['first', 'last'], 't': ['min', 'max', 'count']})
        df_tank_traj.columns = ['particle', 'x_first', 'x_last', 'y_first', 'y_last', 't_min', 't_max', 'nSpots']
        
        # Calculate displacement in x and y direction
        df_tank_traj['dt'] = (df_tank_traj['t_max'] - df_tank_traj['t_min'] + 1) / frame_rate  # calculate dt in seconds
        df_tank_traj['dx'] = (df_tank_traj['x_last'] - df_tank_traj['x_first']) / df_tank_traj['dt']  # [mm/s]
        df_tank_traj['dy'] = (df_tank_traj['y_last'] - df_tank_traj['y_first']) / df_tank_traj['dt']  # [mm/s]
            
        # calculate orientation and speed
        df_tank_traj['orientation'] = np.arctan2(df_tank_traj['dy'], df_tank_traj['dx']) #[radians]
        df_tank_traj['orientation_degrees'] = np.degrees(df_tank_traj['orientation']) #[degrees]
        df_tank_traj['speed'] = np.sqrt(df_tank_traj['dx'] ** 2 + df_tank_traj['dy'] ** 2) #[mm/s]
        
        ### Clean data by removing particles based on criteria
        # 3. Remove short trajectories   
        df_tank_traj = df_tank_traj[df_tank_traj['dt'] >= min_frames/frame_rate].reset_index(drop=True)
        
        ### Finalize dataframe
        df_tank_traj['tank'] = int(tank)  # Add tank number as integer column
        df_tank_traj['particle'] = df_tank_traj['particle'] + (int(tank)-1)*1000000  # Offset particle IDs to be unique across tanks
        
        ### Append to list
        data_pertraj.append(df_tank_traj) # Dataframe after processing with one entry per trajectory
        
    dfs_traj[i] = pd.concat(data_pertraj, ignore_index=True)

### Save scaled and processed data into pickle files for plotting scripts
for i, df in enumerate(dfs_traj):
    df.to_pickle(os.path.join(path_to_save_scaledtraj, f'traj_{treatments[i]}_scaled_pertraj.pkl'))


### Statistical analysis for data per trajectory
df_still_pertraj = slice_time_window(dfs[0], analysis_time_start, analysis_time_end)
df_breeze_pertraj = slice_time_window(dfs[1], analysis_time_start, analysis_time_end)
df_stormy_pertraj = slice_time_window(dfs[2], analysis_time_start, analysis_time_end)
df_control_pertraj = slice_time_window(dfs[3], 0, analysis_time_end)

# Check that the same time window is used for all treatments
# print(f"Time window for analysis per trajectory: {analysis_time_start} to {analysis_time_end} frames")
# print(f"Still: {df_still_pertraj['t'].min()} to {df_still_pertraj['t'].max()} frames")
# print(f"Breeze: {df_breeze_pertraj['t'].min()} to {df_breeze_pertraj['t'].max()} frames")
# print(f"Stormy: {df_stormy_pertraj['t'].min()} to {df_stormy_pertraj['t'].max()} frames")
# print(f"Control: {df_control_pertraj['t'].min()} to {df_control_pertraj['t'].max()} frames")

still_means_pertraj = [mean_dy_per_particle(df_still_pertraj, int(tank)) for tank in tanks]
breeze_means_pertraj = [mean_dy_per_particle(df_breeze_pertraj, int(tank)) for tank in tanks]
stormy_means_pertraj = [mean_dy_per_particle(df_stormy_pertraj, int(tank)) for tank in tanks]
control_means_pertraj = [mean_dy_per_particle(df_control_pertraj, int(tank)) for tank in tanks]   

#  Calculate means and 95% CI for each treatment
means_pertraj = [np.mean(still_means_pertraj), np.mean(breeze_means_pertraj), np.mean(stormy_means_pertraj), np.mean(control_means_pertraj)]
STD_pertraj = [np.std(still_means_pertraj), np.std(breeze_means_pertraj), np.std(stormy_means_pertraj), np.std(control_means_pertraj)]
CI_pertraj = [STD_pertraj[i] / np.sqrt(n_replicates) * t_critical for i in range(len(treatments))]

# ANOVA: per-tank mean dy across still, breeze, stormy
f_stat_pertraj, p_value_pertraj = sp.stats.f_oneway(still_means_pertraj, breeze_means_pertraj, stormy_means_pertraj)

print("\n === Mean dy (per trajectory) with 95% CI ===")
for i, treatment in enumerate(treatments):
    print(f"{treatment.capitalize()}: Mean dy = {means_pertraj[i]:.2f} ± {CI_pertraj[i]:.2f} mm/s")

# %%
