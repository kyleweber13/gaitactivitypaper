import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import os


def find_bouts(peaks_inds, fs, start_time, subj="", min_steps=3, min_duration=180, max_break=5, min_cadence=0,
               show_plot=False, quiet=True):

    print(f"\nFinding bouts of minimum {min_steps} steps, miniumum duration of {min_duration} seconds,"
          f" maximum break of {max_break} seconds, and minimum cadence of {min_cadence} steps/min...")

    starts = []
    stops = []
    n_steps_list = []

    peaks_inds = list(peaks_inds)

    curr_ind = 0
    for i in range(len(peaks_inds)):
        if i >= curr_ind:

            prev_step = peaks_inds[i]
            n_steps = 1

            for j in range(i + 1, len(peaks_inds)):
                if (peaks_inds[j] - prev_step) / fs <= max_break:
                    n_steps += 1

                    if not quiet:
                        print(f"Peak = {peaks_inds[j]}, prev peak = {prev_step}, "
                              f"gap = {round((peaks_inds[j] - prev_step) / fs, 1)}, steps = {n_steps}")
                    prev_step = peaks_inds[j]

                if (peaks_inds[j] - prev_step) / fs > max_break:
                    if n_steps >= min_steps:
                        starts.append(peaks_inds[i])
                        curr_ind = j
                        stops.append(peaks_inds[j - 1])
                        n_steps_list.append(n_steps)

                    if n_steps < min_steps:
                        if not quiet:
                            print("Not enough steps in bout.")
                    break

    df_out = pd.DataFrame({'full_id': [subj] * len(starts), "start": starts, "end": stops, "number_steps": n_steps_list})
    df_out['duration'] = [(j - i) / fs for i, j in zip(starts, stops)]

    df_out = df_out.loc[df_out['duration'] >= min_duration]

    df_out['cadence'] = 60 * df_out['number_steps'] / df_out['duration']

    df_out = df_out.loc[df_out['cadence'] >= min_cadence]

    df_out['start_timestamp'] = [start_time + timedelta(seconds=row.start / fs) for row in df_out.itertuples()]
    df_out['end_timestamp'] = [start_time + timedelta(seconds=row.end / fs) for row in df_out.itertuples()]

    if show_plot:
        fig, ax = plt.subplots(1, sharey='col', sharex='col')
        ax.hist(df_out['cadence'], bins=np.arange(0, 150, 5), edgecolor='black')
        ax.set_ylabel("N_walks")
        ax.set_xlabel("cadence")
        ax.set_title(f"New ({min_steps} step min., {max_break}s max break, {min_duration}s min. duration)")

    print(f"-Found {df_out.shape[0]} bouts.")

    return df_out.reset_index(drop=True)


def combine_all_walkintensity(folder_name, file_names, df_demos=None):

    print(f"\nCombining {len(file_names)} files...")

    df = pd.DataFrame(columns=['full_id', 'sedentary', 'light', 'moderate', 'n_epochs', 'bout_num',
                               'epoch_dur', 'cutpoint', 'sed%', 'light%', 'mod%', 'avm_mean', 'avm_sd',
                               'min_cadence', 'max_cadence', 'med_cadence'])

    for file in file_names:
        d = pd.read_excel(folder_name + file)
        d['full_id'] = [file.split("_")[0] + "_" + file.split("_")[1]] * d.shape[0]

        for col in ['sedentary', 'light', 'moderate', 'n_epochs', 'bout_num',
                    'epoch_dur', 'sed%', 'light%', 'mod%', 'avm_mean', 'avm_sd',
                    'min_cadence', 'max_cadence', 'med_cadence']:
            d[col] = [float(i) for i in d[col]]

        df = df.append(d)

    if df_demos is not None:
        df['age'] = [df_demos.loc[df_demos['full_id'] == row.full_id]['Age'].iloc[0] for row in df.itertuples()]
        df['cohort'] = [df_demos.loc[df_demos['full_id'] == row.full_id]['NDD'].iloc[0] for row in df.itertuples()]

    return df.reset_index(drop=True)


def calculate_intensity_totals_participant(df_epoch_intensity, df_demos):

    n_subj = len(df_epoch_intensity['full_id'].unique())
    print("\nCalculating activity totals for each of the {} participant{}...".format(n_subj, 's' if n_subj != 1 else ""))

    vals_out = []
    g = df_epoch_intensity.groupby("full_id")

    for subj in df_epoch_intensity['full_id'].unique():
        subj_data = g.get_group(subj)

        n_walks = subj_data['bout_num'].max()
        n_sed = list(subj_data['intensity']).count("sedentary")
        n_light = list(subj_data['intensity']).count("light")
        n_mod = list(subj_data['intensity']).count("moderate")
        med_cad = subj_data['cadence'].median()
        sd_cad = subj_data['cadence'].std()

        subj_demos = df_demos.loc[df_demos['full_id'] == subj]
        age = subj_demos['Age'].iloc[0] if 'Age' in subj_demos.columns else 'age'
        ndd = subj_demos['NDD'].iloc[0]

        vals_out.append([subj, n_sed, n_light, n_mod, n_sed + n_light + n_mod, n_walks, med_cad, sd_cad, age, ndd])

    df_out = pd.DataFrame(vals_out, columns=['full_id', 'sedentary', "light", 'moderate',
                                             'n_epochs', 'n_walks', 'med_cadence', 'sd_cadence', 'age', 'ndd'])

    df_out['sed%'] = df_out['sedentary'] * 100 / df_out['n_epochs']
    df_out['light%'] = df_out['light'] * 100 / df_out['n_epochs']
    df_out['mod%'] = df_out['moderate'] * 100 / df_out['n_epochs']

    return df_out.sort_values("sed%")


def calculate_context_in_walkingbouts(df_wrist_epochs, df_epoch_intensity):

    print("\nLooking for wrist non-wear and sleep during included walking bouts...")

    df_out = df_epoch_intensity.copy()

    nw = []
    sleep = []

    epoch_len = (df_epoch_intensity.iloc[1]['start_time'] - df_epoch_intensity.iloc[0]['start_time']).total_seconds()

    for row in df_epoch_intensity.itertuples():
        df_context = df_wrist_epochs.loc[(df_wrist_epochs['start_time'] >= row.start_time) &
                                         (df_wrist_epochs['start_time'] < row.start_time + timedelta(seconds=epoch_len))]

        nw.append(sum(list(df_context['nw'])) * 100 / df_context.shape[0])
        sleep.append(sum(list(df_context['sleep'])) * 100 / df_context.shape[0])

    df_out['nw%'] = nw
    df_out['sleep%'] = sleep

    print("-Found {} walks with wrist non-wear.".format(df_out.loc[df_out['nw%'] > 0].shape[0]))
    print("-Found {} walks with sleep.".format(df_out.loc[df_out['sleep%'] > 0].shape[0]))

    return df_out


def combine_dataframes(filenames):

    print("\nCombining {} files from {}...".format(len(filenames), os.path.dirname(filenames[0])))

    df = None

    for i, file in enumerate(filenames):
        full_id = os.path.basename(file)[:10]

        if i == 0:
            df = pd.read_csv(file)

            if 'full_id' not in df.columns:
                df['full_id'] = [full_id] * df.shape[0]

        if i > 0:
            d = pd.read_csv(file)

            if 'full_id' not in d.columns:
                d['full_id'] = [full_id] * d.shape[0]

            df = df.append(d, ignore_index=True)

    return df


def remove_low_cadence_edge_epochs(df_epochs, min_cadence=80):
    """Removes the first/last epoch in each walk is removed if its cadence is below threshold.
       Loops until no epochs are removed.

        arguments:
        -df_epochs: df with epoched data from within gait bouts
        -min_cadence: steps per minute

        returns:
        -cropped df
    """

    print(f"\nRemoving epochs at start/end of bout until none on the 'edges' have a cadence below {min_cadence} spm...")

    df2 = df_epochs.copy()
    df2['final_epoch'] = [row.epoch_num == df2.loc[df2['bout_num'] == row.bout_num].iloc[-1]['epoch_num'] for
                          row in df2.itertuples()]

    removed_epochs = True
    iters = 1
    while removed_epochs:
        n_start = df2.shape[0]

        mask = [False if (row.epoch_num == df2.loc[df2['bout_num'] == row.bout_num].iloc[0]['epoch_num'] and row.cadence < min_cadence) or
                         (row.epoch_num == df2.loc[df2['bout_num'] == row.bout_num].iloc[-1]['epoch_num'] and row.cadence < min_cadence)
                else True for row in df2.itertuples()]

        df2 = df2.loc[mask]
        n_end = df2.shape[0]

        if n_start > n_end:
            removed_epochs = True
        if n_start == n_end:
            removed_epochs = False

        iters += 1

    print(f"-Removed {df_epochs.shape[0] - df2.shape[0]} 'edge' epochs with cadence below "
          f"{min_cadence} spm with {iters} iterations")

    return df2


def calculate_freeliving_walktime(df_totals, epoch_len=15, gaitbout_folder=None):
    df_totals = df_totals.copy()

    if 'study_code' not in df_totals.columns:
        df_totals['study_code'] = [i[:5] for i in df_totals['full_id']]

    if gaitbout_folder is None:
        gaitbout_folder = {"OND09": 'W:/sync_cal_test/nimbalwear/OND09/analytics/gait/bouts/{}_01_GAIT_BOUTS.csv',
                           'OND06': 'W:/sync_cal_test/nimbalwear/OND06/analytics/gait/bouts/{}_01_GAIT_BOUTS.csv'}

    freeliving_walksecs = {}
    for row in df_totals.itertuples():
        try:
            df_bouts = pd.read_csv(gaitbout_folder[row.study_code].format(row.full_id))
            df_bouts['start_timestamp'] = pd.to_datetime(df_bouts['start_timestamp'])
            df_bouts['end_timestamp'] = pd.to_datetime(df_bouts['end_timestamp'])
            df_bouts['duration'] = [(j - i).total_seconds() for i, j in zip(df_bouts['start_timestamp'],
                                                                            df_bouts['end_timestamp'])]
            sec_walking = df_bouts['duration'].sum()
        except FileNotFoundError:
            sec_walking = None

        freeliving_walksecs[row.full_id] = sec_walking

    df_totals['fl_walktime'] = [freeliving_walksecs[row.full_id] for row in df_totals.itertuples()]
    df_totals['long_walktime'] = df_totals['n_epochs'] * epoch_len
    df_totals['perc_fl_walktime'] = df_totals['long_walktime'] * 100 / df_totals['fl_walktime']

    return df_totals

