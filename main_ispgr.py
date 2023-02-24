import DataImport
import Other
import Stats
import pandas as pd
from datetime import timedelta

""" =============================================== SET-UP ========================================================="""


def screen_subjs(df_crit, crits=(), min_bouts=5, min_bouts_criter="n_180sec", min_age=0, max_age=1000):

    print(f"\nScreening for {crits} and more than {min_bouts} bouts of {min_bouts_criter} "
          f"with an age range of {min_age} - {max_age} years...")

    d = df_crit.copy()

    try:
        for key in crits.keys():
            d = d.loc[d[key].isin(crits[key])]

        if min_bouts > 0:
            d = d.loc[d[min_bouts_criter] >= min_bouts]

    except KeyError:
        d = None
        print("Invalid key. Options are:")
        print(list(df_crit.columns))

    d = d.loc[(d['Age'] >= min_age) & (d['Age'] <= max_age)]

    print("{}/{} subjects meet criteri{}.".format(d.shape[0], df_crit.shape[0], "on" if len(crits) == 1 else "a"))

    return df_crit, d


def import_tabular(root_dir, subjs):

    data_dicts = {}

    if type(subjs) is str:
        subjs = list([subjs])

    for subj in subjs:
        df_wrist = pd.read_csv(f"{root_dir}EpochedWrist/{subj}_EpochedWrist.csv")
        df_epoch = pd.read_csv(f"{root_dir}WalkingEpochs/{subj}_WalkEpochs.csv")
        df_walktotals = pd.read_csv(f"{root_dir}WalkingTotals/{subj}_WalkingTotals.csv")

        df_bouts = pd.read_csv(f"{root_dir}ProcessedBouts/{subj}_WalkingBouts.csv")
        df_bouts.columns = [i if i != 'start_timestamp' else 'start_time' for i in df_bouts.columns]
        df_bouts.columns = [i if i != 'end_timestamp' else 'end_time' for i in df_bouts.columns]

        df_steps = pd.read_csv(f"{root_dir}Steps/{subj}_01_GAIT_STEPS.csv")
        df_steps.columns = [i if i != 'step_idx' else 'step_index' for i in df_steps.columns]
        df_steps = df_steps.sort_values("step_index").reset_index(drop=True)

        df_daily = pd.read_csv(f"{root_dir}DailyStats/{subj}_FreeLivingDaily.csv")

        data_dict = {"wrist1": df_wrist, 'epochs': df_epoch, 'steps': df_steps,
                     'totals': df_walktotals, 'bouts': df_bouts, 'daily': df_daily}

        for key in data_dict:
            cols = data_dict[key].columns

            for col in cols:
                if 'time' in col:
                    data_dict[key][col] = pd.to_datetime(data_dict[key][col])

        df_epoch['end_time'] = [row.start_time + timedelta(seconds=row.epoch_dur) for row in df_epoch.itertuples()]

        data_dicts[subj] = data_dict

    return data_dicts


subj = 'OND06_9525'

cutpoints = (62.5, 92.5)

file_dict_old = {"OND06": {"daily_steps": "W:/NiMBaLWEAR/OND06/analyzed/gait/daily_gait/{}_01_DAILY_GAIT.csv",
                           'sptw': "W:/NiMBaLWEAR/OND06/analyzed/sleep/sptw/{}_01_SPTW.csv",
                           'nw': "W:/NiMBaLWEAR/OND06/analyzed/nonwear/standard_nonwear_times/GNAC/{}_01_GNAC_{}Wrist_NONWEAR.csv"},
                 "OND09": {"daily_steps": 'W:/NiMBaLWEAR/OND09/analytics/gait/daily/{}_01_GAIT_DAILY.csv',
                           "sptw": 'W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/{}_01_SPTW.csv',
                           'nw': 'W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{}_01_AXV6_{}Wrist_NONWEAR.csv'}}

file_dict_new = {"OND06": {"daily_steps": "W:/sync_cal_test/nimbalwear/OND06/analytics/gait/daily/{}_01_DAILY_GAIT.csv",
                           'sptw': "W:/sync_cal_test/nimbalwear/OND06/analytics/sleep/sptw/{}_01_SPTW.csv",
                           'nw': "W:/sync_cal_test/nimbalwear/OND06/analytics/nonwear/bouts_cropped/{}_01_GNOR_{}Wrist_NONWEAR.csv"},
                 "OND09": {"daily_steps": "W:/sync_cal_test/nimbalwear/OND09/analytics/gait/daily/{}_01_DAILY_GAIT.csv",
                           'sptw': "W:/sync_cal_test/nimbalwear/OND09/analytics/sleep/sptw/{}_01_SPTW.csv",
                           'nw': "W:/sync_cal_test/nimbalwear/OND09/analytics/nonwear/bouts_cropped/{}_01_AXV6_{}Wrist_NONWEAR.csv"}}

""" =============================================================================================================== """
""" ========================================= FUNCTION CALLS ======================================================="""
""" =============================================================================================================== """

root_dir = "O:/OBI/ONDRI@Home/Papers/ISPGR_2022/Kyle and Beth/Take 3/Data/Current/"

df = DataImport.import_demos(f"{root_dir}OND06_PD_Demographics.xlsx")

# df['cohort_id'] = Other.create_cohort_ids(df_demos=df)

df_demos, subjs = screen_subjs(df_crit=df, crits={'DEVICE_SIDE': ['D', 'Both'], 'AnyGaitAid': ['No', 'Yes'], 'study_code': ['OND06', 'OND09']}, min_bouts=0, min_bouts_criter='n_180sec', min_age=55)

# creating file names -----------
"""
df['wrist_edf'] = WristProcessing.create_wrist_filenames(df_demos=df,
                                                         edf_dict={"OND09": "",
                                                                   "OND06": "W:/sync_cal_test/nimbalwear/OND06/wearables/device_edf_cropped/"},
                                                         fname_dict={"OND09": 'OND09_{}_01_AXV6_{}Wrist',
                                                                     'OND06': "OND06_{}_01_GNOR_{}Wrist"})

df['wrist_epoch'] = WristProcessing.create_wrist_epoch_filenames(df=df, folder=f"{root_dir}EpochedWrist/")
df['wrist_starts'] = DataImport.get_starttimes(filenames=df['wrist_edf'])
df['ankle_edf'] = StepsProcessing.create_ankle_edf_filenames(df)
df['ankle_fs'] = DataImport.get_sample_rates(filenames=df['ankle_edf'])
df['ankle_starts'] = DataImport.get_starttimes(filenames=df['ankle_edf'])
"""

""" ============================================ RUNNING PARTICIPANTS IN LOOP ===================================== """

"""
failed_files, data = RunLoop.run_loop(df_demos=df_demos, full_ids=list([subj]), cutpoints=cutpoints,
                                      save_files=False, root_dir=root_dir, min_step_time=60/200,
                                      min_cadence=80, min_bout_dur=60, mrp=5)
"""

""" ============================================ INDIVIDUAL PROCESSING =========================================== """

"""
subj = 'OND06_1027'
# df_wrist = DataImport.import_wrist_epoch(filename=f'C:/Users/ksweber/Desktop/Processed/WristGaitPaper/EpochedWrist/{subj}_EpochedWrist.csv')
# df_steps = DataImport.import_steps_file(f"C:/Users/ksweber/Desktop/Processed/WristGaitPaper/Steps/{subj}_01_AXV6_RAnkle_Steps.csv")

df_wrist = DataImport.import_wrist_epoch(filename=f'{root_dir}/EpochedWrist/{subj}_EpochedWrist.csv')
df_steps = DataImport.import_steps_file(f'{root_dir}/Steps/{subj}_01_GAIT_STEPS.csv')

df_bouts = BoutProcessing.find_bouts(peaks_inds=df_steps['step_index'], subj=subj,
                                     fs=df.loc[df['full_id'] == subj].iloc[0]['ankle_fs'],
                                     min_steps=3, min_duration=60, max_break=5, min_cadence=80,
                                     show_plot=False, quiet=True,
                                     start_time=df.loc[df['full_id'] == subj]['ankle_starts'].iloc[0])

df_wrist = DataImport.create_context_mask(subj=subj, df_wrist_epochs=df_wrist, df_demos=df)
"""

""" ============================================== COMBINED PROCESSING ============================================ """

"""
df_bout_intensity = IntensityProcessing.process_bouts(df_bouts=df_bouts, df_1s_epochs=df_wrist, df_steps=df_steps, subj=subj,
                                                      epoch_len=15, study_code=subj.split("_")[0], method='crop',
                                                      cutpoints=cutpoints, cutpoint_name='Fraysse',
                                                      show_plot=False, save_dir=None)

df_bout_intensity = BoutProcessing.calculate_context_in_walkingbouts(df_wrist_epochs=df_wrist,
                                                                     df_bout_intensity=df_bout_intensity)

df_walk_intensity = IntensityProcessing.calculate_bout_data(full_id=subj, df_bout_intensity=df_bout_intensity)
df_totals = BoutProcessing.calculate_intensity_totals_participant(df_walk_epochs=df_bout_intensity, df_demos=df)

df_wrist15 = IntensityProcessing.epoch_intensity(df_wrist_1s=df_wrist, cutpoints=cutpoints, epoch_len=15, author='Fraysse')
df_daily = FreeLiving.calculate_daily_values(full_id=subj,  n_valid_hours=10, df_wrist=df_wrist15, gait_file_dict=None,
                                             save_file=None)"""

""" =================================== COMBINING MULTIPLE PARTICIPANTS' FILES ==================================== """

stat_names = ["count", "mean", '25%', "50%", '75%', "min", "max", "std"]

# demographics descriptive stats
df_demos_desc = Stats.descriptive_stats(df=df, column_names=['Age'], stat_names=stat_names, groupby=None)

# summary of all walking epochs
# df_walk_epochs_all = Other.combine_dataframes(folder='C:/Users/ksweber/Desktop/Processed/WristGaitPaper/WalkingEpochs/', keyword="WalkEpochs.csv")
df_walk_epochs_all = Other.combine_dataframes(folder=f"{root_dir}WalkingEpochs/", keyword="WalkEpochs.csv")
df_walk_epochs_all['cohort_id'] = Other.copy_cohort_ids(df_copy=df, df_new=df_walk_epochs_all)

# summary of free-living activity/stepping
df_daily_all = Other.combine_dataframes(folder=f"{root_dir}DailyStats/", keyword="FreeLivingDaily.csv")
df_daily_all['cohort_id'] = Other.copy_cohort_ids(df_copy=df, df_new=df_daily_all)

# summary of all bouts
df_procbouts_all = Other.combine_dataframes(folder=f'{root_dir}ProcessedBouts/', keyword="WalkingBouts.csv")
df_procbouts_all['cohort_id'] = Other.copy_cohort_ids(df_copy=df, df_new=df_procbouts_all)

# summary of participant totals
df_walktotals_all = Other.combine_dataframes(folder=f'{root_dir}WalkingTotals/', keyword="WalkingTotals.csv")
df_walktotals_all = Stats.flag_quartiles(df=df_walktotals_all, sort_column='sed%', ascending=False, n_per_q=1)
df_walktotals_all['cohort_id'] = [f'PD{i}' for i in range(1, df_walktotals_all.shape[0]+1)]

df_demos['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walk_epochs_all, df_new=df_demos)

""" =============================================== DESCRIPTIVE STATISTICS ======================================== """

# epoch descriptive statistics, by participant
df_walk_desc = Stats.descriptive_stats(df=df_walk_epochs_all, column_names=['avm', 'cadence'], stat_names=stat_names, groupby='full_id')

# daily free-living descriptive stats, by participant, valid days only
df_daily_desc = Stats.descriptive_stats(df=df_daily_all.loc[df_daily_all['valid_day']], stat_names=stat_names, groupby='full_id',
                                        column_names=['steps', 'mod_norm', 'light_norm', 'sed_norm', 'mod_epochs', 'light_epochs', 'sed_epochs'])

# walking bout descriptive statistics, by participant
df_procbouts_desc = Stats.descriptive_stats(df=df_procbouts_all, column_names=['number_steps', 'duration', 'cadence'], stat_names=stat_names, groupby='cohort_id')

# descriptive statistics for participant totals (whole sample)
df_walktotals_desc = Stats.descriptive_stats(df=df_walktotals_all, stat_names=stat_names, groupby=None, column_names=['n_walks', 'n_epochs', 'med_cadence', 'sd_cadence', 'sed%', 'light%', 'mod%'])

"""
barplot = Plotting.intensity_barplot(df=df_walktotals_all, min_walks=0, sort_col='sed%', ascending=True, figsize=(8, 8),
                                     df_sig=None, sig_icon="*", lw=1.5, fontsize=16, legend_fontsize=12,
                                     ytick_subjs='cohort_id', greyscale=False, incl_legend=True,
                                     binary_mvpa=False, binary_activity=False)
"""

""" =============================================== STATISTICAL ANALYSIS ========================================== """

# fisher_p_group, df_fisher = Stats.run_fisher_exact(df_all_epochs=df_walk_epochs_all, group_active=True, alpha=.05, bonferroni=True)

""" ============================================= Optional Function Calls ========================================= """

"""
ankle = DataImport.import_edf(df.loc[df['full_id'] == subj]['ankle_edf'].iloc[0])
wrist = DataImport.import_edf(df.loc[df['full_id'] == subj]['wrist_edf'].iloc[0])

data = import_tabular(root_dir=root_dir, subjs=[subj])

bout_fig = Plotting.plot_bouts(ankle_obj=ankle, wrist_obj=wrist,
                               df_wrist=data[subj]['wrist1'], epoch_intensity=data[subj]['epochs'],
                               df_long_bouts=data[subj]['bouts'], df_all_bouts=None, use_median_cadence=True,
                               df_steps=data[subj]['steps'], df_walk_epochs=data[subj]['epochs'],
                               cutpoints=cutpoints, ankle_axis='Accelerometer y', wrist_axis='Accelerometer y',
                               bout_steps_only=True)
"""

