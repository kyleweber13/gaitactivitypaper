import Plotting
import DataImport
import IntensityProcessing
import Other
import Multisubject_Processing as MSP
import Stats
import matplotlib.pyplot as plt

# from DataReview.Analysis import freq_analysis
plt.rcParams.update({'font.serif': 'Times New Roman', 'font.family': 'serif'})


# =============================================== SET-UP =============================================================


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


cutpoints = (62.5, 92.5)

file_dict_new = {"OND06": {"daily_steps": "W:/sync_cal_test/nimbalwear/OND06/analytics/gait/daily/{}_01_DAILY_GAIT.csv",
                           'sptw': "W:/sync_cal_test/nimbalwear/OND06/analytics/sleep/sptw/{}_01_SPTW.csv",
                           'nw': "W:/sync_cal_test/nimbalwear/OND06/analytics/nonwear/bouts_cropped/{}_01_GNOR_{}Wrist_NONWEAR.csv"},
                 "OND09": {"daily_steps": "W:/sync_cal_test/nimbalwear/OND09/analytics/gait/daily/{}_01_DAILY_GAIT.csv",
                           'sptw': "W:/sync_cal_test/nimbalwear/OND09/analytics/sleep/sptw/{}_01_SPTW.csv",
                           'nw': "W:/sync_cal_test/nimbalwear/OND09/analytics/nonwear/bouts_cropped/{}_01_AXV6_{}Wrist_NONWEAR.csv"}}

# ====================================================================================================================
# =========================================== FUNCTION CALLS =========================================================
# ====================================================================================================================

root_dir = "O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Calibrated/Data/"

# df['cohort_id'] = Other.create_cohort_ids(df_demos=df)
df_demos, subjs = screen_subjs(df_crit=DataImport.import_demos(f"{root_dir}Demographics_NEW.xlsx"),
                               crits={'DEVICE_SIDE': ['D', 'Both'], 'AnyGaitAid': ['No', 'Yes'],
                                      'study_code': ['OND06', 'OND09']},
                               min_bouts=0, min_bouts_criter='n_180sec', min_age=55)

df_demos = DataImport.format_df_demos(df_demos=df_demos, root_dir=root_dir,
                                      edf_dict={"OND09": "W:/sync_cal_test/nimbalwear/OND09/wearables/device_edf_cropped/",
                                                "OND06": "W:/sync_cal_test/nimbalwear/OND06/wearables/device_edf_cropped/"},
                                      file_dict={"OND09": 'OND09_{}_01_AXV6_{}Wrist', 'OND06': "OND06_{}_01_GNOR_{}Wrist"})

# ============================================ RUNNING PARTICIPANTS IN LOOP ==========================================

"""
# ond09_subjs = [i for i in df_demos['full_id'].unique() if 'OND09' in i and i not in ['OND09_0023', 'OND09_0029', 'OND09_0012', 'OND09_0043', 'OND09_0060']]
failed_files, data = RunLoop.run_loop(df_demos=df_demos, full_ids=sorted(list(['OND09_0037', 'OND06_9525', 'OND06_3413'])),
                                      cutpoints=cutpoints,
                                      save_files=False, root_dir=root_dir, correct_ond09_cadence=True,
                                      min_cadence=80, min_bout_dur=60, mrp=5,
                                      min_step_time=60/200, remove_edge_low_cadence=True)
"""

# ============================================ INDIVIDUAL PROCESSING =================================================

"""
subj = 'OND06_1027'

df_wrist = DataImport.import_wrist_epoch(filename=f'{root_dir}/EpochedWrist/{subj}_EpochedWrist.csv')
df_steps = DataImport.import_steps_file(f'{root_dir}/Steps/{subj}_01_GAIT_STEPS.csv')

df_bouts = BoutProcessing.find_bouts(peaks_inds=df_steps['step_index'], subj=subj,
                                     fs=df_demos.loc[df_demos['full_id'] == subj].iloc[0]['ankle_fs'],
                                     min_steps=3, min_duration=60, max_break=5, min_cadence=80,
                                     show_plot=False, quiet=True,
                                     start_time=df_demos.loc[df_demos['full_id'] == subj]['ankle_starts'].iloc[0])

df_wrist = DataImport.create_context_mask(subj=subj, df_wrist_epochs=df_wrist, df_demos=df_demos)
"""

# ============================================== COMBINED PROCESSING =================================================

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
                                             save_file=None)
"""

# =================================== COMBINING MULTIPLE PARTICIPANTS' FILES =========================================


stat_names = ["count", "mean", '25%', "50%", '75%', "min", "max", "std"]

# demographics descriptive stats
df_demos_desc = Stats.descriptive_stats(df=df_demos, column_names=['Age'], stat_names=stat_names, groupby=None)

# summary of participant totals
df_walktotals_all = MSP.combine_walktotals(root_dir=root_dir)
df_walktotals_all = Stats.flag_quartiles(df=df_walktotals_all, sort_column='sed%', n_per_q=9, ascending=True)
df_walktotals_all['NDD'] = [df_demos.loc[df_demos['full_id'] == row.full_id].iloc[0]['NDD'] for row in df_walktotals_all.itertuples()]
df_walktotals_all.sort_values('sed%', ascending=False, inplace=True)
df_walktotals_all['cohort_id'] = Other.create_cohort_ids(df_walktotals_all, ctrl_flag='OA')

# summary of all walking epochs
df_walk_epochs_all = MSP.combine_walkepochs(root_dir=root_dir, df_walktotals=df_walktotals_all)

# summary of free-living activity/stepping
df_daily_all = MSP.combine_freeliving(df_walktotals=df_walktotals_all, root_dir=root_dir)

# summary of all bouts
df_procbouts_all = Other.combine_dataframes(folder=f'{root_dir}ProcessedBouts/', keyword="WalkingBouts.csv")
df_procbouts_all['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walktotals_all, df_new=df_procbouts_all)

# won't work until all are processed
# df_demos['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walk_epochs_all, df_new=df_demos)

df_cp_totals = IntensityProcessing.compare_cutpoint_totals(df=df_walk_epochs_all).sort_values("fraysse_sedp", ascending=False).reset_index(drop=True)

# =============================================== DESCRIPTIVE STATISTICS =============================================

# cutpoint comparison
# df_cp_desc = Stats.descriptive_stats(df=df_cp_totals, stat_names=stat_names, groupby=None, column_names=['fraysse_sedp', 'fraysse_lightp', 'fraysse_modp', 'powell_sedp', 'powell_lightp', 'powell_modp', 'diff_sedp', 'diff_lightp', 'diff_modp'])

# epoch descriptive statistics, by participant
df_walk_desc = Stats.descriptive_stats(df=df_walk_epochs_all, column_names=['avm', 'cadence'], stat_names=stat_names, groupby='full_id')
df_walk_desc['avm']['cov'] = df_walk_desc['avm']['std'] * 100 / df_walk_desc['avm']['mean']
df_walktotals_all['avm_cov'] = [df_walk_desc['avm'].loc[row.full_id]['cov'] for row in df_walktotals_all.itertuples()]

# daily free-living descriptive stats, by participant, valid days only
df_daily_desc = Stats.descriptive_stats(df=df_daily_all.loc[df_daily_all['valid_day']], stat_names=stat_names, groupby='full_id',
                                        column_names=['valid_hours', 'steps', 'mod_norm', 'light_norm', 'sed_norm', 'mod_epochs', 'light_epochs', 'sed_epochs'])

# walking bout descriptive statistics, by participant
df_procbouts_desc = Stats.descriptive_stats(df=df_procbouts_all, column_names=['number_steps', 'duration', 'cadence'], stat_names=stat_names, groupby='cohort_id')

# descriptive statistics for participant totals (whole sample)
df_walktotals_desc = Stats.descriptive_stats(df=df_walktotals_all, stat_names=stat_names, groupby=None,
                                             column_names=['n_walks', 'n_epochs', 'med_cadence', 'sd_cadence', 'sed%',
                                                           'light%', 'mod%', 'long_walktime', 'fl_walktime', 'perc_fl_walktime'])

#  =============================================== STATISTICAL ANALYSIS ==============================================

fisher_p_group, df_fisher = Stats.run_fisher_exact(df_all_epochs=df_walk_epochs_all, group_active=True, alpha=.05, bonferroni=True)
# df_cp_ttest = Stats.run_cutpoint_ttest(df=df_cp_totals)

df_cp_totals['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walktotals_all, df_new=df_cp_totals)

# ============================================= Optional Function Calls ==============================================

# cutpoint comparison plots ----------
# cp_fig = plot_comparison_barplot(df_cp_totals, figsize=(13, 8), binary_mvpa=False, binary_activity=False, greyscale=False, greyscale_diff=False, label_fontsize=12, fontsize=10, legend_fontsize=10)
# fig = plot_cp_diff_density(df_cp_totals)
# fig = cp_diff_hist(df_cp_totals, incl_density=False)
# fig = cp_diff_scatter(df_cp_totals)
# fig = cp_comp_barplot_all(df_cp_totals)
# fig = cp_comp_mean_barplot(df_cp_desc, err_colname='std')
# fig = cp_comp_meandiff_barplot(df_cp_desc, err_colname='std')
# plt.savefig("O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/CSEP_Abstract/Plot Samples/barplot_meandiffs.png", dpi=150)


"""
df_daily_desc['steps'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Daily_Steps.xlsx")
df_daily_desc['valid_hours'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Valid_Hours.xlsx")
df_daily_desc['mod_norm'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Daily_ModNorm.xlsx")
df_daily_desc['light_norm'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Daily_LightNorm.xlsx")
df_daily_desc['sed_norm'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Daily_SedNorm.xlsx")
df_daily_desc['mod_epochs'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Daily_ModEpochs.xlsx")
df_daily_desc['light_epochs'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Daily_LightEpochs.xlsx")
df_daily_desc['sed_epochs'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Daily_SedEpochs.xlsx")

df_procbouts_desc['number_steps'].to_excel("C:/Users/ksweber/Desktop/ToUpload/ProcBouts_NumberSteps.xlsx")
df_procbouts_desc['duration'].to_excel("C:/Users/ksweber/Desktop/ToUpload/ProcBouts_Duration.xlsx")
df_procbouts_desc['cadence'].to_excel("C:/Users/ksweber/Desktop/ToUpload/ProcBouts_Cadence.xlsx")

df_walk_desc['avm'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Walks_avm.xlsx")
df_walk_desc['cadence'].to_excel("C:/Users/ksweber/Desktop/ToUpload/Walks_cadence.xlsx")

df_walktotals_desc.to_excel("C:/Users/ksweber/Desktop/ToUpload/WalkTotals.xlsx")
"""

"""
df_q1 = df_walktotals_all.loc[df_walktotals_all['quartile'] == 'Q1']
df_q4 = df_walktotals_all.loc[df_walktotals_all['quartile'] == 'Q4']

df_q_stats = Stats.run_quartile_stats(col_names=['n_walks', 'n_epochs', 'med_cadence', 'sed%', 'age', 'steps', 'avm_cov',
                                                 'perc_fl_walktime', 'sed_norm', 'sed_epochs', 'light_epochs', 'mod_epochs', 'sleep_epochs'],
                                      dfq1=df_q1, dfq4=df_q4, df_walktotals=df_walktotals_all, df_daily=df_daily_all)
"""

"""
subj = 'OND09_0037'
ankle = DataImport.import_edf(df_demos.loc[df_demos['full_id'] == subj]['ankle_edf'].iloc[0])
wrist = DataImport.import_edf(df_demos.loc[df_demos['full_id'] == subj]['wrist_edf'].iloc[0])

from GaitActivityPaper.Organized.Organized_ISPGR import import_tabular
data = import_tabular(root_dir=root_dir, subjs=[subj])

i = 0

# OND09_0037: variety of arm swing ------
windows = [['2021-11-11 15:40:45', '2021-11-11 15:41:16'], ['2021-11-10 19:37:30', '2021-11-10 19:38:01'],
           ['2021-11-13 11:42:49', '2021-11-13 11:43:20'], ['2021-11-13 11:42:52', '2021-11-13 11:43:23']]
regions = [[pd.to_datetime(windows[i][0]) + timedelta(seconds=10), pd.to_datetime(windows[i][0]) + timedelta(seconds=20)],
           [pd.to_datetime(windows[i][0]) + timedelta(seconds=10), pd.to_datetime(windows[i][0]) + timedelta(seconds=20)]]

# OND06_9525: PD tremor ------
# ankle.signals[ankle.get_signal_index('Accelerometer y')] *= -1

# separate windows (15-sec)
# windows = [['2020-03-06 8:03:00', '2020-03-06 8:04:30']]
# regions = [['2020-03-06 8:03:05', '2020-03-06 8:03:19.9'], ['2020-03-06 8:03:42.5', '2020-03-06 8:03:57.4']]

# non-tremor + tremor in one 20-sec window
# windows = [['2020-03-06 8:03:12.5', '2020-03-06 8:04:12.5']]
# regions = [['2020-03-06 8:03:32.5', '2020-03-06 8:03:52.5'], ['2020-03-06 8:03:42.5', '2020-03-06 8:03:52.4']]

# OND06_3413: walker ------
# windows = [['2020-01-29 12:36:50', '2020-01-29 12:37:20']]
# regions = [['2020-01-29 12:36:58.4', '2020-01-29 12:37:08.3'], ['2020-01-29 12:36:58', '2020-01-29 12:37:12.7']]

# OND06_5919: cane --------
# windows = [['2019-10-10 10:23:00', '2019-10-10 10:23:30']]
# regions = [['2019-10-10 10:23:10', '2019-10-10 10:23:20'], ['2019-10-10 10:23:10', '2019-10-10 10:23:20']]

plot1, plot2, wrist, ankle = Plotting.data_sections3(full_id=subj, wrist=wrist, ankle=ankle,
                                                    window_start=pd.to_datetime(windows[i][0]),
                                                    window_end=pd.to_datetime(windows[i][1]),
                                                    pad_window=0,
                                                     # save_dir="C:/Users/ksweber/Desktop/Paper_Images/",
                                                    regions=[regions[0]], df_demos=subjs,
                                                    # avm_markers=True,
                                                    avm_bar=False,
                                                    colors=['gold', 'dodgerblue'], alpha=(.2, .2),
                                                    # show_fig2_raw=False,
                                                    show_legend=False, show_legend2=True,
                                                    rem_ankle_base=True, fig_width=14, steps=data[subj]['steps'], epoch_len=5,
                                                    force_wrist_ylim=(None, None), force_ankle_ylim=(None, None),
                                                    force_raw_vm_ylim=(-10, 1000), force_avm_ylim=(None, None),
                                           # region_name="TEST",
                                                    wrist_up=wrist.signals[wrist.get_signal_index("Accelerometer y")],
                                                    wrist_ant=wrist.signals[wrist.get_signal_index("Accelerometer x")],
                                                    wrist_med=wrist.signals[wrist.get_signal_index("Accelerometer z")],
                                                    ankle_axis={'data': ankle.signals[ankle.get_signal_index("Accelerometer x")],
                                                                'label': 'AP'})
"""

# fft_fig, df_fft = freq_analysis(obj=wrist, subj=subj, channel='Accelerometer x', ts='2020-03-06 7:35:26', lowpass=None, highpass=None, sample_rate=None, n_secs=60*5, stft_mult=20, stft=True, show_plot=True)

df_value_counts = df_daily_all.loc[df_daily_all['valid_day']].groupby("full_id")

df_cp_totals['fraysse_activep'] = df_cp_totals['fraysse_lightp'] + df_cp_totals['fraysse_modp']
df_cp_totals['n_days'] = [len(df_value_counts.groups[group]) for group in df_cp_totals['full_id']]
df_cp_totals['long_walktime'] = [df_walktotals_all.loc[df_walktotals_all['full_id'] == full_id]['long_walktime'].iloc[0] for full_id in df_cp_totals['full_id']]
df_cp_totals['walk_mins_daily_avg'] = df_cp_totals['long_walktime'] / 60 / df_cp_totals['n_days']

df_cp_totals['med_cadence'] = [df_walktotals_all.loc[df_walktotals_all['full_id'] == row.full_id].iloc[0]['med_cadence'] for row in df_cp_totals.itertuples()]

avg_daily_walkmins = df_cp_totals['walk_mins_daily_avg'].mean()
fraysse_percent = df_cp_totals['fraysse_modp'].mean()
powell_percent = df_cp_totals['powell_modp'].mean()
diff_percent = fraysse_percent - powell_percent

df_cp_totals['cp_mins_diff_day'] = diff_percent / 100 * df_cp_totals['walk_mins_daily_avg']
df_cp_totals['cp_mins_diff_day'].describe()
df_cp_totals['cohort'] = ['CTRL' if 'CTRL' in row.cohort_id else 'NDD' for row in df_cp_totals.itertuples()]



# figure 1
fig = Plotting.intensity_barplot(df=df_cp_totals, cp_author='fraysse', figsize=(10.25, 10.25), df_sig=None, sig_icon="*",
                                 ytick_subjs='cohort_id', greyscale=True, incl_legend=True, lw=1.5,
                                 fontsize=14, legend_fontsize=14,
                                 binary_mvpa=False, binary_activity=False, ax=None)
fig.axes[0].set_xticklabels([int(i) for i in fig.axes[0].get_xticks()], fontsize=14)
fig.axes[0].set_xlabel("Wrist-Derived Intensity Classification\n(% of 15-second epochs during LONG walks)", fontsize=16)
fig.axes[0].set_ylabel("Participants", fontsize=16)
plt.tight_layout()
# plt.savefig("O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Calibrated/JMIR mHealth and uHealth submission/Response to Reviewers and Revisions_Jan2023/Revised Figures/figure1_OA_highres.png", dpi=250)
plt.savefig("O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Calibrated/JMIR mHealth and uHealth submission/Copyediting/Revised Figures/figure1.png", dpi=117)

plt.close()

