import pandas as pd
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt


def process_bouts(df_bouts, df_1s_epochs, df_steps, epoch_len=15, study_code="Test", subj="Test",
                  method='crop', cutpoints=(62.5, 92.5), cutpoint_name='Fraysse', show_plot=False, save_dir=None):

    print(f"\nRe-epoching wrist data into {epoch_len}-second bouts during walking bouts and "
          f"calculating intensity using {cutpoint_name} cutpoints...")

    # data for output df
    bout_num = []  # walk bout index
    epoch_num = []  # epoch index within bout
    intensity = []  # list of epochs' intensities for each bout
    timestamps = []  # actual timestamp
    cadences = []  # epoch cadence
    avms = []  # epoch avm
    epoch_dur = []  # epoch duration in seconds
    epoch_steps = []  # number of steps in epoch

    bout_number = 1  # gait bout index
    # Loops gait bouts, crops to time contained within df1s (probably redundant)
    for row in df_bouts.loc[(df_bouts['start_timestamp'] >= df_1s_epochs.iloc[0]['start_time']) &
                            (df_bouts['end_timestamp'] <= df_1s_epochs.iloc[-1]['end_time'])].itertuples():

        # timestamp to ensure wrist epoching doesn't go beyond end of gait bout
        # example: if bout is 32 seconds long and epoch_len == 15, only includes time 0-30
        end_stamp = row.start_timestamp + timedelta(seconds=int(np.floor(row.duration/epoch_len)) * epoch_len)

        # Crops df1s to {epoch_len}-second epochs and ensures last epoch does not end after gait bout ends
        df_epoch = df_1s_epochs.loc[(df_1s_epochs['start_time'] >= row.start_timestamp) &
                                    (df_1s_epochs['end_time'] < end_stamp)]

        # cropping df_allsteps to epoch
        steps = df_steps.loc[(df_steps['step_time'] >= row.start_timestamp) & (df_steps['step_time'] < end_stamp)]
        steps = steps.sort_values("step_time")

        # wrist intensity: Fraysse cutpoints
        sub_epoch = 1  # epoch index within gait bout
        for i in range(0, df_epoch.shape[0], epoch_len):

            # average avm value in epoch_len epochs
            avm = np.mean(df_epoch.iloc[i:i+epoch_len]['avm'])
            avms.append(avm)

            if avm < cutpoints[0]:
                intensity.append('sedentary')
            if cutpoints[0] <= avm < cutpoints[1]:
                intensity.append("light")
            if avm >= cutpoints[1]:
                intensity.append("moderate")

            epoch_num.append(sub_epoch)
            sub_epoch += 1
            bout_num.append(bout_number)
            timestamps.append(df_epoch.iloc[i]['start_time'])

        # timestamps for epochs
        epoch_stamps = list(pd.date_range(start=row.start_timestamp, end=end_stamp, freq=f"{epoch_len}s"))

        for i, j in zip(epoch_stamps[:], epoch_stamps[1:]):
            epoch_dur.append((j-i).total_seconds())
            steps_sub = steps.loc[(steps['step_time'] >= i) & (steps['step_time'] < j)]

            if method == 'crop':
                dur = (steps_sub.iloc[-1]['step_time'] - steps_sub.iloc[0]['step_time']).total_seconds()
            if method == 'full':
                dur = epoch_len

            n_steps = steps_sub.shape[0] - 1

            if n_steps < 0:
                n_steps = 0
            epoch_steps.append(n_steps)
            cad = 60 * n_steps / dur if dur != 0 else None

            try:
                cadences.append(cad)
            except TypeError:
                cadences.append(None)

        bout_number += 1

    df_out = pd.DataFrame({'full_id': [subj]*len(timestamps),
                           "start_time": timestamps,
                           'end_time': [i + timedelta(seconds=epoch_len) for i in timestamps],
                           'bout_num': bout_num, 'epoch_num': epoch_num,
                           'epoch_dur': epoch_dur, 'avm': avms, 'intensity': intensity,
                           'cutpoint': [cutpoint_name] * len(timestamps),
                           'number_steps': epoch_steps, 'cadence': cadences})
    df_out = df_out.dropna()

    # Calculates activity intensities for each bout -------------------------
    intensities = []
    for bout in df_out['bout_num'].unique():
        d = df_out.loc[df_out['bout_num'] == bout]
        vals = d['intensity'].value_counts()

        if 'sedentary' not in vals.keys():
            vals['sedentary'] = 0
        if 'light' not in vals.keys():
            vals['light'] = 0
        if 'moderate' not in vals.keys():
            vals['moderate'] = 0

        intensities.append([bout, vals['sedentary'], vals['light'], vals['moderate'], d.shape[0]])

    df_boutint = pd.DataFrame(intensities, columns=['bout_num', "sed", "light", "mod", 'n_epochs'])
    df_boutint['cutpoint'] = [cutpoint_name] * df_boutint.shape[0]

    df_bouts['sed'] = list(df_boutint['sed'])
    df_bouts['light'] = list(df_boutint['light'])
    df_bouts['mod'] = list(df_boutint['mod'])
    df_bouts['n_epochs'] = list(df_boutint['n_epochs'])

    if show_plot:
        plt.close("all")

        g = df_out.groupby("bout_num")
        plt.scatter(df_out['avm'], df_out['cadence'], color='black', label='All15sEpochs')

        for i, bout in enumerate(df_out['bout_num'].unique()):
            g_bout = g.get_group(bout)
            if i == 0:
                plt.scatter(g_bout['avm'].mean(), g_bout['cadence'].mean(), color='red', label='BoutMeans', zorder=1)
            if i > 0:
                plt.scatter(g_bout['avm'].mean(), g_bout['cadence'].mean(), color='red', zorder=1)

        fill_val_y = plt.ylim()[1]
        fill_val_x = plt.xlim()[1]
        plt.fill_between(x=[0, cutpoints[0]], y1=0, y2=fill_val_y,
                         color='grey', label='sed', alpha=.35)
        plt.fill_between(x=[cutpoints[0], cutpoints[1]], y1=0, y2=fill_val_y,
                         color='limegreen', label='light', alpha=.35)
        plt.fill_between(x=[cutpoints[1], fill_val_x], y1=0, y2=fill_val_y,
                         color='orange', label='moderate', alpha=.35)
        plt.xlabel("avm")
        plt.ylabel("cadence")
        plt.legend()
        plt.ylim(0, fill_val_y)
        plt.xlim(0, fill_val_x)
        plt.grid(zorder=0)

        if save_dir is not None:
            plt.savefig(f"{save_dir}{study_code}_{subj}_{cutpoint_name}_BoutAVMCadence.png")

    return df_out


def calculate_bout_data(full_id, df_epoch_intensity):

    print("\nAnalyzing each walking bout for totals...")

    bout_data = []
    for bout_num in df_epoch_intensity['bout_num'].unique():
        walk = df_epoch_intensity.loc[df_epoch_intensity['bout_num'] == bout_num]

        i = list(walk['intensity'])

        bout_data.append([full_id, walk.iloc[0]['start_time'],
                          walk.iloc[-1]['start_time'] + timedelta(seconds=walk.iloc[0]['epoch_dur']),
                          i.count("sedentary"), i.count("light"), i.count('moderate'), max(walk['epoch_num']),
                          bout_num, walk.iloc[0]['epoch_dur'], 'Fraysse',
                          100 * i.count("sedentary") / walk.shape[0], 100 * i.count("light") / walk.shape[0],
                          100 * i.count("moderate") / walk.shape[0], walk['avm'].mean(), walk['avm'].std(),
                          walk['cadence'].min(), walk['cadence'].max(), walk['cadence'].median()])

    df_walk_intensity = pd.DataFrame(bout_data, columns=['full_id', 'start_time', 'end_time',
                                                         'sedentary', 'light', 'moderate',
                                                         'n_epochs', 'bout_num', 'epoch_dur', 'cutpoint', 'sed%',
                                                         'light%', 'mod%', 'avm_mean', 'avm_sd',
                                                         'min_cadence', 'max_cadence', 'med_cadence'])

    return df_walk_intensity


def epoch_intensity(df_wrist_1s, cutpoints=(62.5, 92.5), epoch_len=15, author='Fraysse'):

    print(f"\nRe-epoching into {epoch_len}-sec epochs and applying {author} cutpoints...")

    avm = []
    nw = []
    sleep = []

    """if 'nw' not in df_wrist_1s.columns:
        df_wrist_1s['nw'] = [0] * df_wrist_1s.shape[0]
    if 'sleep' not in df_wrist_1s.columns:
        df_wrist_1s['sleep'] = [0] * df_wrist_1s.shape[0]"""

    for i in range(0, df_wrist_1s.shape[0], epoch_len):
        d = df_wrist_1s.iloc[i:i+epoch_len]
        avm.append(np.mean(d['avm']))
        nw.append(100*np.sum(d['nw'])/epoch_len)
        sleep.append(100*np.sum(d['sleep'])/epoch_len)

    df = pd.DataFrame({"start_time": df_wrist_1s.iloc[::epoch_len]['start_time'], 'avm': avm,
                       'nw': nw, 'sleep': sleep,
                       'use_epoch': [i + j == 0 for i, j in zip(nw, sleep)]})

    vals = []
    for row in df.itertuples():
        if row.avm < cutpoints[0]:
            vals.append('sedentary')
        if cutpoints[0] <= row.avm < cutpoints[1]:
            vals.append('light')
        if row.avm >= cutpoints[1]:
            vals.append("moderate")

    df["intensity"] = vals

    return df.reset_index(drop=True)


def add_powell_intensity(df, side='dominant'):

    cp = {'dominant': [51/450*1000, 68/450*1000, 142/450*1000],
          'nondominant': [47/450*1000, 64/450*1000, 157/450*1000]}

    df = df.copy()

    df['powell'] = ['sedentary'] * df.shape[0]

    df.loc[(df['avm'] > cp[side][0]) & (df['avm'] <= cp[side][1]), 'powell'] = 'light'
    df.loc[(df['avm'] > cp[side][1]) & (df['avm'] <= cp[side][2]), 'powell'] = 'moderate'
    df.loc[df['avm'] > cp[side][2], 'powell'] = 'vigorous'

    return df


def compare_cutpoint_totals(df):
    d = []
    for subj in df['full_id'].unique():
        df_subj = df.loc[df['full_id'] == subj]
        subj_d = [subj]
        for col in ['intensity', 'powell']:
            for intensity in ['sedentary', 'light', 'moderate']:
                subj_d.append(list(df_subj[col]).count(intensity))
                subj_d.append(list(df_subj[col]).count(intensity) * 100 / df_subj.shape[0])

        d.append(subj_d)

    df_totals = pd.DataFrame(d, columns=['full_id', 'fraysse_sed', 'fraysse_sedp',
                                         'fraysse_light', 'fraysse_lightp',
                                         'fraysse_mod', 'fraysse_modp',
                                         'powell_sed', 'powell_sedp',
                                         'powell_light', 'powell_lightp',
                                         'powell_mod', 'powell_modp'])
    df_totals['n_epochs'] = [row.fraysse_sed + row.fraysse_light + row.fraysse_mod for row in df_totals.itertuples()]
    df_totals['diff_sedp'] = df_totals['fraysse_sedp'] - df_totals['powell_sedp']
    df_totals['diff_lightp'] = df_totals['fraysse_lightp'] - df_totals['powell_lightp']
    df_totals['diff_modp'] = df_totals['fraysse_modp'] - df_totals['powell_modp']

    return df_totals

