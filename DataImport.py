import nwdata
import pandas as pd
import pyedflib
from nwdata import EDF
from WristProcessing import create_wrist_filenames, create_wrist_epoch_filenames
import numpy as np
import os
from datetime import timedelta as td


def import_demos(filename="O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Working Folder/Data/Demographics_InProgress.xlsx"):

    print(f"Importing demographics from {filename}...")

    df = pd.read_excel(filename, skiprows=0)
    df['subject_id'] = ["0" * (4 - len(str(row.subject_id))) + str(row.subject_id) for row in df.itertuples()]
    df['full_id'] = [str(row.study_code) + "_" + str(row.subject_id) for row in df.itertuples()]

    return df


def import_edf(filename):

    print(f"\nImporting {filename}...")

    data = nwdata.NWData()
    data.import_edf(file_path=filename, quiet=False)

    data.ts = pd.date_range(start=data.header["start_datetime"] if 'start_datetime' in data.header.keys() else data.header['startdate'],
                            periods=len(data.signals[0]),
                            freq="{}ms".format(round(1000 / data.signal_headers[0]["sample_rate"], 6)))

    return data


def import_steps_file(filename):

    print(f"\nImporting {filename}...")

    df = pd.read_csv(filename)

    try:
        df = df.loc[df['step_state'] == 'success']
    except KeyError:
        pass

    if 'full_id' not in df.columns:
        df['full_id'] = [str(row.study_code) + "_" + str(row.subject_id) for row in df.itertuples()]

    try:
        df = df[['full_id', 'step_index', 'step_time']]
    except KeyError:
        df = df[['full_id', 'step_idx', 'step_time']]
        df.columns = ['full_id', 'step_index', 'step_time']

    df['step_time'] = pd.to_datetime(df['step_time'])
    df = df.sort_values('step_time').reset_index(drop=True)

    return df


def import_bouts_file(filename):

    print(f"\nImporting {filename}...")

    df = pd.read_csv(filename)

    df = df[['gait_bout_num', 'start_timestamp', 'end_timestamp', 'start_dp', 'end_dp', 'bout_length_sec', 'number_steps']]
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])

    return df


def get_sample_rates(filenames):

    print(f"\nChecking sample rate for {len(filenames)} files...")

    sample_rates = []
    for i, file in enumerate(list(filenames)):
        if file is not None:
            if os.path.exists(file):
                a = EDF.EDFFile(file)
                a.read_header()

                for h in a.signal_headers:
                    if h['label'] == 'Accelerometer x':
                        sample_rates.append(h['sample_rate'])
                        break

            if not os.path.exists(file):
                sample_rates.append(1)
        if file is None:
            sample_rates.append(1)

    return sample_rates


def get_starttimes(filenames):

    filenames = list(filenames)

    print(f"\nChecking start times for {len(filenames)} files...")

    start_times = []
    for i, file in enumerate(filenames):
        if file is not None:
            if os.path.exists(file):
                a = EDF.EDFFile(file)
                a.read_header()

                try:
                    start_times.append(a.header['startdate'])
                except KeyError:
                    start_times.append(a.header['start_datetime'])
            if not os.path.exists(file):
                start_times.append(None)
        if file is None:
            start_times.append(None)

    return start_times


def import_wrist_epoch(filename):

    print(f"\nImporting {filename}...")

    df_wrist = pd.read_csv(filename)
    try:
        df_wrist['start_time'] = pd.to_datetime(df_wrist['start_time'])
    except KeyError:
        pass

    try:
        df_wrist['end_time'] = pd.to_datetime(df_wrist['end_time'])
    except KeyError:
        pass

    try:
        df_wrist['timestamp'] = pd.to_datetime(df_wrist['timestamp'])
    except KeyError:
        pass

    return df_wrist


def create_nw_sleep_flags():

    def epoch_intensity(df_1s, cutpoint_key, epoch_len=15, author='Fraysse'):
        cutpoints = {"PowellDominant": [51 * 1000 / 30 / 15, 68 * 1000 / 30 / 15, 142 * 1000 / 30 / 15],
                     "PowellNon-dominant": [47 * 1000 / 30 / 15, 64 * 1000 / 30 / 15, 157 * 1000 / 30 / 15],
                     'FraysseDominant': [62.5, 92.5, 10000],
                     'FraysseNon-dominant': [42.5, 98, 10000]}

        print(f"\nRe-epoching into {epoch_len}-sec epochs and applying {author} {cutpoint_key} cutpoints...")

        avm = [np.mean(df_1s.iloc[i:i + epoch_len]['avm']) for i in range(0, df_1s.shape[0], epoch_len)]
        df = pd.DataFrame({"timestamp": df_1s.iloc[::epoch_len]['timestamp'], 'avm': avm})

        vals = []
        for row in df.itertuples():
            if row.avm < cutpoints[author + cutpoint_key][0]:
                vals.append('sedentary')
            if cutpoints[author + cutpoint_key][0] <= row.avm < cutpoints[author + cutpoint_key][1]:
                vals.append('light')
            if row.avm >= cutpoints[author + cutpoint_key][1]:
                vals.append("moderate")

        df["intensity"] = vals

        return df.reset_index(drop=True)

    def import_1s_epochs(epoch_folder, full_id):
        df_epoch = pd.read_csv(epoch_folder + '{}_DomWrist.csv'.format(full_id))
        df_epoch['timestamp'] = pd.to_datetime(df_epoch['timestamp'])
        df_epoch['date'] = [i.date() for i in df_epoch['timestamp']]
        start_day = df_epoch['timestamp'].iloc[0].date() + td(days=1)
        end_day = df_epoch['timestamp'].iloc[-1].date()
        df_epoch = df_epoch.loc[(df_epoch['date'] >= start_day) &
                                (df_epoch['timestamp'] <= pd.to_datetime(end_day))].reset_index(drop=True)

        return df_epoch, start_day, end_day

    def import_daily_steps(start_day, end_day, full_id, file):
        df_daily_steps = pd.read_csv(file.format(full_id))
        df_daily_steps['date'] = pd.to_datetime(df_daily_steps['date'])
        df_daily_steps['date'] = [i.date() for i in df_daily_steps['date']]
        df_daily_steps = df_daily_steps.loc[(df_daily_steps['date'] >= start_day) & (df_daily_steps['date'] < end_day)]

        return df_daily_steps

    def import_sptw(file, full_id, df_epoch, epoch_len):
        df_sleep = pd.read_csv(file.format(full_id))
        df_sleep['start_time'] = pd.to_datetime(df_sleep['start_time'])
        df_sleep['end_time'] = pd.to_datetime(df_sleep['end_time'])

        start_stamp = pd.to_datetime(df_epoch.iloc[0]['timestamp'])

        sleep_mask = np.zeros(df_epoch.shape[0])
        for row in df_sleep.itertuples():
            start_idx = int(np.floor((row.start_time - start_stamp).total_seconds() / epoch_len))
            end_idx = int(np.ceil((row.end_time - start_stamp).total_seconds() / epoch_len))

            if start_idx < 0 and end_idx >= 0:
                sleep_mask[0:end_idx] = 1

            if start_idx >= 0 and end_idx >= 0:
                if end_idx < df_epoch.shape[0]:
                    sleep_mask[start_idx:end_idx] = 1
                if end_idx >= df_epoch.shape[0]:
                    sleep_mask[start_idx:-1] = 1

        return sleep_mask

    def import_nw(file, full_id, hand_side, df_epoch):
        df_nw = pd.read_csv(file.format(full_id, hand_side))
        df_nw['start_time'] = pd.to_datetime(df_nw['start_time'])
        df_nw['end_time'] = pd.to_datetime(df_nw['end_time'])

        start_stamp = df_epoch['timestamp'].iloc[0]

        nw_mask = np.zeros(df_epoch.shape[0])
        for row in df_nw.itertuples():
            start_idx = int(np.floor((row.start_time - start_stamp).total_seconds() / epoch_len))
            end_idx = int(np.ceil((row.end_time - start_stamp).total_seconds() / epoch_len))

            if start_idx < 0 and end_idx >= 0:
                nw_mask[0:end_idx] = 1

            if start_idx >= 0 and end_idx >= 0:
                if end_idx < df_epoch.shape[0]:
                    nw_mask[start_idx:end_idx] = 1
                if end_idx >= df_epoch.shape[0]:
                    nw_mask[start_idx:-1] = 1

        return nw_mask

    def check_walks_for_nw_sleep(full_id, df_walking_epochs, df_epoch):
        subj_epochs = df_walking_epochs.loc[df_dom_epochs['full_id'] == full_id]

        walks = []
        nw_total = []
        sleep = []
        for walk_num in subj_epochs['bout_num'].unique():
            d = subj_epochs.loc[subj_epochs['bout_num'] == walk_num].reset_index(drop=True)

            w = df_epoch.loc[(df_epoch['timestamp'] >= d.iloc[0]['timestamp']) &
                             (df_epoch['timestamp'] <= d.iloc[-1]['timestamp'])]

            walks.append(walk_num)
            nw_total.append(w['nw'].sum())
            sleep.append(w['sleep'].sum())

        df_use_walks = pd.DataFrame({"full_id": [subj] * len(walks), 'bout_num': walks,
                                     'no_nw': [True if i == 0 else False for i in nw_total],
                                     'no_sleep': [True if i == 0 else False for i in sleep]})
        df_use_walks['final_use'] = [i and j for i, j in zip(df_use_walks['no_nw'], df_use_walks['no_sleep'])]

        return df_use_walks

    epoch_len = 15

    file_dict = {"OND06": {"daily_steps": "W:/NiMBaLWEAR/OND06/analyzed/gait/daily_gait/{}_01_DAILY_GAIT.csv",
                           'sptw': "W:/NiMBaLWEAR/OND06/analyzed/sleep/sptw/{}_01_SPTW.csv",
                           'nw': "W:/NiMBaLWEAR/OND06/analyzed/nonwear/standard_nonwear_times/GNAC/{}_01_GNAC_{}Wrist_NONWEAR.csv"},
                 "OND09": {"daily_steps": 'W:/NiMBaLWEAR/OND09/analytics/gait/daily/{}_01_GAIT_DAILY.csv',
                           "sptw": 'W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/{}_01_SPTW.csv',
                           'nw': 'W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{}_01_AXV6_{}Wrist_NONWEAR.csv'}}

    for subj in df_dom_int_total['full_id'].unique():
        print(f"========== {subj} =========")
        df_epoch, start_day, end_day = import_1s_epochs(
            epoch_folder="O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Data/WristEpoch1s/",
            full_id=subj)
        df_fraysse = epoch_intensity(df_1s=df_epoch, cutpoint_key='Dominant', author='Fraysse', epoch_len=epoch_len)

        df_daily_steps = import_daily_steps(file=file_dict[subj.split("_")[0]]['daily_steps'],
                                            start_day=start_day, end_day=end_day, full_id=subj)

        df_fraysse['sleep'] = import_sptw(file=file_dict[subj.split("_")[0]]['sptw'],
                                          full_id=subj, df_epoch=df_fraysse, epoch_len=epoch_len)

        df_fraysse['nw'] = import_nw(hand_side=df_demos.loc[df_demos['full_id'] == subj]['Hand'].iloc[0],
                                     full_id=subj, df_epoch=df_fraysse,
                                     file=file_dict[subj.split("_")[0]]['nw'])

        df_fraysse['use_epoch'] = df_fraysse['nw'] + df_fraysse['sleep'] == 0

        df_use_walks = check_walks_for_nw_sleep(full_id=subj, df_epoch=df_fraysse, df_walking_epochs=df_dom_epochs)

        df_fraysse.to_csv(f"C:/Users/ksweber/Desktop/Processed/Fraysse_All_Epochs/{subj}_FraysseEpochs.csv",
                          index=False)
        df_use_walks.to_csv(f"C:/Users/ksweber/Desktop/Processed/Use_Walks/{subj}_Walks.csv", index=False)


def create_context_mask(subj, df_wrist_epochs, df_demos, file_dict=None):

    print("\nFlagging periods of sleep and wrist non-wear...")

    df_wrist_epochs = df_wrist_epochs.copy()

    if file_dict is None:
        file_dict = {"OND06": {"daily_steps": "W:/NiMBaLWEAR/OND06/analyzed/gait/daily_gait/{}_01_DAILY_GAIT.csv",
                               'sptw': "W:/NiMBaLWEAR/OND06/analyzed/sleep/sptw/{}_01_SPTW.csv",
                               'nw': "W:/NiMBaLWEAR/OND06/analyzed/nonwear/standard_nonwear_times/GNAC/{}_01_GNAC_{}Wrist_NONWEAR.csv"},
                     "OND09": {"daily_steps": 'W:/NiMBaLWEAR/OND09/analytics/gait/daily/{}_01_GAIT_DAILY.csv',
                               "sptw": 'W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/{}_01_SPTW.csv',
                               'nw': 'W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{}_01_AXV6_{}Wrist_NONWEAR.csv'}}

    sleep_mask = np.zeros(df_wrist_epochs.shape[0])
    nw_mask = np.zeros(df_wrist_epochs.shape[0])

    start_time = df_wrist_epochs.iloc[0]['start_time']
    study_code = subj.split("_")[0]

    if 'Hand' in df_demos.columns:
        hand = df_demos.loc[df_demos['full_id'] == subj]['Hand'].iloc[0]
    if 'Hand' not in df_demos.columns:
        hand = df_demos.loc[df_demos['full_id'] == subj]['UseSide'].iloc[0][0].capitalize()

    df_sleep = pd.read_csv(file_dict[study_code]['sptw'].format(subj))
    df_sleep['start_time'] = pd.to_datetime(df_sleep['start_time'])
    df_sleep['end_time'] = pd.to_datetime(df_sleep['end_time'])

    df_nw = pd.read_csv(file_dict[study_code]['nw'].format(subj, hand))

    if 'event' in df_nw.columns:
        df_nw = df_nw.loc[df_nw['event'] == 'nonwear'].reset_index(drop=True)

    df_nw['start_time'] = pd.to_datetime(df_nw['start_time'])
    df_nw['end_time'] = pd.to_datetime(df_nw['end_time'])

    for row in df_sleep.itertuples():
        start_i = int((row.start_time - start_time).total_seconds())
        end_i = int((row.end_time - start_time).total_seconds())
        sleep_mask[start_i:end_i] = 1

    for row in df_nw.itertuples():
        start_i = int((row.start_time - start_time).total_seconds())
        end_i = int((row.end_time - start_time).total_seconds())
        nw_mask[start_i:end_i] = 1

    df_wrist_epochs['nw'] = nw_mask
    df_wrist_epochs['sleep'] = sleep_mask

    df_wrist_epochs['use_epoch'] = df_wrist_epochs['nw'] + df_wrist_epochs['sleep'] == 0

    return df_wrist_epochs


def format_df_demos(df_demos, root_dir, edf_dict, file_dict):

    df = df_demos.copy().sort_values("full_id").reset_index(drop=True)

    df['UseSide'] = df['hand'].copy() if 'hand' in df.columns else df['Hand'].copy()

    # creating file names -----------

    df['wrist_edf'] = create_wrist_filenames(df_demos=df,
                                             edf_dict=edf_dict,
                                             fname_dict=file_dict)

    df['wrist_epoch'] = create_wrist_epoch_filenames(df=df, folder=f"{root_dir}EpochedWrist/")
    df['wrist_starts'] = get_starttimes(filenames=df['wrist_edf'])

    from GaitActivityPaper.Organized.StepsProcessing import create_ankle_edf_filenames
    df['ankle_edf'] = create_ankle_edf_filenames(df=df)

    df['ankle_fs'] = get_sample_rates(filenames=df['ankle_edf'])
    df['ankle_starts'] = get_starttimes(filenames=df['ankle_edf'])

    folder = edf_dict['OND09']
    files = os.listdir(folder)

    for subj in df_demos['full_id'].unique():
        subj_files = [i for i in files if subj in i]
        wrist_file = [i for i in subj_files if 'Wrist' in i]

        dom = df_demos.loc[df_demos['full_id'] == subj]['hand' if 'hand' in df_demos.columns else 'Hand'].iloc[0][0]

        if f'{subj}_01_AXV6_{dom}Wrist.edf' in wrist_file:
            df_demos.loc[df_demos['full_id'] == subj, 'device_side'] = 'Dom'
        if f'{subj}_01_AXV6_{dom}Wrist.edf' not in wrist_file and len(wrist_file) == 1:
            df_demos.loc[df_demos['full_id'] == subj, 'device_side'] = 'NonDom'

    return df
