import Other, Stats
import BoutProcessing, IntensityProcessing


def combine_walktotals(root_dir, sort_col='sed%'):

    df_walktotals_all = Other.combine_dataframes(folder=f'{root_dir}WalkingTotals/', keyword="WalkingTotals.csv")
    df_walktotals_all['cohort_id'] = Other.create_cohort_ids(df=df_walktotals_all)
    df_walktotals_all = BoutProcessing.calculate_freeliving_walktime(df_totals=df_walktotals_all, epoch_len=15, gaitbout_folder=None)

    return df_walktotals_all


def combine_walkepochs(root_dir, df_walktotals):

    df_walk_epochs_all = Other.combine_dataframes(folder=f"{root_dir}WalkingEpochs/", keyword="WalkEpochs.csv")
    df_walk_epochs_all['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walktotals, df_new=df_walk_epochs_all)
    df_walk_epochs_all = IntensityProcessing.add_powell_intensity(df=df_walk_epochs_all, side='dominant')
    df_walk_epochs_all['powell'].replace({"vigorous": 'moderate'}, inplace=True)

    return df_walk_epochs_all


def combine_freeliving(root_dir, df_walktotals):

    df_daily_all = Other.combine_dataframes(folder=f"{root_dir}DailyStats/", keyword="FreeLivingDaily.csv")
    df_daily_all['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walktotals, df_new=df_daily_all)
    df_daily_all = df_daily_all.reset_index(drop=True)
    df_daily_all = df_daily_all.loc[df_daily_all['sleep_epochs'] > 0]

    return df_daily_all

