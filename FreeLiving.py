import pandas as pd


def calculate_daily_values(full_id, save_file=None, n_valid_hours=10, df_wrist=None, gait_file_dict=None):
    """Function that combines processed activity data with steps data to give daily totals.

        arguments:
        -full_id: full_id (studycode_subjID)
        -save_file: if pathway given, will save output df to that location. None to skip.
        -n_valid_hours: number of hours of waking wear time to be considered a valid day.
        -df_wrist: epoched wrist df from which activity volumes and calculated. Must be in desired epoch length.
        -gait_file_dict: dictionary with study_code as key and value is pathway to daily gait files

        returns:
        -df with a row of daily totals for each day in the collection period
    """

    if gait_file_dict is None:
        gait_file_dict = {"OND06": "W:/NiMBaLWEAR/OND06/analyzed/gait/daily_gait/{}_01_DAILY_GAIT.csv",
                          "OND09": 'W:/NiMBaLWEAR/OND09/analytics/gait/daily/{}_01_GAIT_DAILY.csv'}

    print(f"\nCalculating daily sleep, nonwear, activity, and valid data for {full_id}...")

    study_code = full_id.split("_")[0]
    df_wrist['date'] = [i.date() for i in df_wrist['start_time']]

    epoch_len = int((df_wrist['start_time'].iloc[1] - df_wrist['start_time'].iloc[0]).total_seconds())

    df_daily_steps = pd.read_csv(gait_file_dict[study_code].format(full_id))
    df_daily_steps['date'] = pd.to_datetime(df_daily_steps['date'])
    df_daily_steps['date'] = [i.date() for i in df_daily_steps['date']]

    dates = df_wrist['date'].unique()[:-1]

    daily_vals = []
    for date in dates:
        d = df_wrist.loc[df_wrist['date'] == date].reset_index(drop=True)
        valid = d.loc[d['use_epoch']]
        sleep = d.loc[d['sleep'] > 0]
        nw = d.loc[d['nw'] > 0]
        mod = valid.loc[valid['intensity'] == 'moderate'].shape[0]
        light = valid.loc[valid['intensity'] == 'light'].shape[0]
        sed = valid.loc[valid['intensity'] == 'sedentary'].shape[0]

        steps = df_daily_steps.loc[df_daily_steps['date'] == date]
        if steps.shape[0] == 1:
            daily_steps = steps.iloc[0]['total_steps']
        if steps.shape[0] == 0:
            daily_steps = None

        total_time = d.shape[0]
        valid_time = valid.shape[0]
        sleep_time = sleep.shape[0]
        nw_time = nw.shape[0]

        daily_vals.append([full_id, date, epoch_len, total_time, valid_time, sleep_time, nw_time,
                           daily_steps, mod, light, sed])

    df_daily = pd.DataFrame(daily_vals,
                            columns=['full_id', 'date', 'epoch_len',
                                     'total_epochs', 'valid_epochs', 'sleep_epochs', 'nw_epochs',
                                     'steps', 'mod_epochs', 'light_epochs', 'sed_epochs'])
    df_daily['valid_hours'] = df_daily['valid_epochs'] * epoch_len / 3600
    df_daily['valid_day'] = df_daily['valid_hours'] >= n_valid_hours
    df_daily['req_hours'] = [n_valid_hours] * df_daily.shape[0]

    # activity normalized to % of valid epochs
    df_daily['mod_norm'] = 100 * df_daily['mod_epochs'] / df_daily['valid_epochs']
    df_daily['light_norm'] = 100 * df_daily['light_epochs'] / df_daily['valid_epochs']
    df_daily['sed_norm'] = 100 * df_daily['sed_epochs'] / df_daily['valid_epochs']

    if save_file is not None:
        df_daily.to_csv(f"{save_file}{full_id}_DailyData.csv", index=False)
        print(f"File saved to {save_file}")

    return df_daily


def calculate_daily_summaries(subjects,
                              dailystats_folder='O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Data/Daily_Stats/'):

    df_daily_all = pd.DataFrame(columns=['full_id', 'date', 'epoch_len', 'total_epochs', 'valid_epochs',
                                         'sleep_epochs', 'nw_epochs', 'steps', 'mod_epochs', 'light_epochs',
                                         'sed_epochs', 'valid_day', 'mod_norm', 'light_norm', 'sed_norm'])
    subj_means = []

    for full_id in subjects:
        # Reads in participant's summary data
        df = pd.read_csv(dailystats_folder + f"{full_id}_FreeLivingDaily.csv")

        # crops df to valid days only
        df = df.loc[df['valid_day']]

        epoch_len = df.iloc[0]['epoch_len']

        subj_means.append([full_id, df.shape[0], df['mod_epochs'].mean() * epoch_len/60,
                           df['light_epochs'].mean() * epoch_len/60, df['sed_epochs'].mean() * epoch_len/60,
                           df['valid_epochs'].mean() * epoch_len/60, df['sleep_epochs'].mean() * epoch_len/60,
                           df['nw_epochs'].mean() * epoch_len/60, df['steps'].mean()])

        df_daily_all = df_daily_all.append(df)

    df_subj_mean = pd.DataFrame(subj_means, columns=['full_id', 'n_days', 'mod_mins', 'light_mins', 'sed_mins',
                                                     'valid_mins', 'sleep_mins', 'nw_mins', 'steps'])

    df_subj_mean['mod_norm'] = df_subj_mean['mod_mins'] * 100 / df_subj_mean['valid_mins']
    df_subj_mean['light_norm'] = df_subj_mean['light_mins'] * 100 / df_subj_mean['valid_mins']
    df_subj_mean['sed_norm'] = df_subj_mean['sed_mins'] * 100 / df_subj_mean['valid_mins']

    return df_daily_all, df_subj_mean
