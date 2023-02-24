import pandas as pd
import numpy as np
import scipy.stats
import pingouin as pg


def descriptive_stats(df, column_names=(), stat_names=(), groupby=None):

    if groupby is None:
        data_out = []
        for column in column_names:
            d = df[column].describe()
            stats_order = d.loc[d.index.isin(stat_names)].index
            d = list(d.loc[d.index.isin(stat_names)])

            data_out.append(d)

        df_out = pd.DataFrame(data_out, columns=stats_order, index=column_names)

        return df_out

    if groupby is not None:
        df_g = df.groupby(groupby)

        df_dict = {}
        for column in column_names:

            group_data = []

            for group in df[groupby].unique():

                d = df_g.get_group(group)[column].describe()
                data = list(d.loc[d.index.isin(stat_names)])

                stats_order = d.loc[d.index.isin(stat_names)].index

                group_data.append(data)

            g = pd.DataFrame(group_data, columns=stats_order, index=df[groupby].unique())

            df_dict[column] = g

        return df_dict


def flag_quartiles(df, sort_column, n_per_q, ascending):

    df2 = df.copy()

    df2 = df2.sort_values(sort_column, ascending=ascending).reset_index(drop=True)

    n_iqr = int(df2.shape[0] - 2 * n_per_q)

    q = []
    for i in range(n_per_q):
        q.append('Q1')
    for i in range(n_iqr):
        q.append("IQR")
    for i in range(n_per_q):
        q.append('Q4')

    df2['quartile'] = q

    return df2


def run_fisher_exact(df_all_epochs, group_active=True, alpha=.05, bonferroni=True):

    print("\nRunning Fischer exact test, grouping {} "
          "against other intensities...".format("MVPA + Light" if group_active else 'sedentary'))

    intensities = ["moderate"]
    if group_active:
        intensities.append("light")

    df = df_all_epochs.copy()
    df['output'] = df['intensity'].isin(intensities)
    df['output'] = [int(i) for i in df['output']]
    df = df[['full_id', 'cohort_id', 'intensity', 'output']]

    # pooled data ---------------
    a_all = [[df['output'].value_counts().loc[1], df.shape[0]],
             [df['output'].value_counts().loc[0], 0]]

    f_all = scipy.stats.fisher_exact(table=a_all, alternative="less")

    # by participant ------------------
    rows = []
    for subj in df['full_id'].unique():
        d = df.loc[df['full_id'] == subj]
        cohort_id = d.iloc[0]['cohort_id']

        try:
            n_wrong = d['output'].value_counts().loc[0]
        except KeyError:
            n_wrong = 0

        try:
            n_right = d['output'].value_counts().loc[1]
        except KeyError:
            n_right = 0

        a = [[d['output'].value_counts().loc[1], d.shape[0]],
             [n_wrong, 0]]

        t = scipy.stats.fisher_exact(table=a, alternative="less")
        p = t[1]

        row = [subj, cohort_id, n_right, n_wrong, d.shape[0], p]
        rows.append(row)

    df_all = pd.DataFrame(rows, columns=['full_id', 'cohort_id', "n_correct", "n_incorrect", "n", "p"])

    if bonferroni:
        df_all['sig'] = df_all['p'] <= (alpha / df_all.shape[0])
    if not bonferroni:
        df_all['sig'] = df_all['p'] <= alpha

    return f_all[0], df_all


def run_quartile_stats(col_names, dfq1, dfq4, df_walktotals, df_daily):

    data_out = []
    cols_out = ['variable', 'q1_mean', 'q1_sd', 'q4_val', 'q4_sd', 'df', 't', 'p']
    for col_name in col_names:
        print(f"\n========== {col_name} ==========")

        try:
            ttest = pg.ttest(dfq1[col_name], dfq4[col_name], paired=False)
            d = df_walktotals.groupby('quartile')[col_name].describe()
            print(f"Q1: {d.loc['Q1']['mean']:.1f} +- {d.loc['Q1']['std']:.1f}")
            print(f"Q4: {d.loc['Q4']['mean']:.1f} +- {d.loc['Q4']['std']:.1f}")
            data_out.append([col_name, d.loc['Q1']['mean'], d.loc['Q1']['std'],
                             d.loc['Q4']['mean'], d.loc['Q4']['std'],
                             ttest['dof'].iloc[0], ttest['T'].iloc[0], ttest['p-val'].iloc[0]])
        except KeyError:
            q1 = (df_daily.loc[df_daily['full_id'].isin(list(dfq1['full_id']))].groupby('full_id')[col_name])
            q4 = (df_daily.loc[df_daily['full_id'].isin(list(dfq4['full_id']))].groupby('full_id')[col_name])

            ttest = pg.ttest(q1.describe()['mean'], q4.describe()['mean'])

            print(f"Q1: {q1.describe()['mean'].mean():.1f} +- {q1.describe()['mean'].std():.1f}")
            print(f"Q4: {q4.describe()['mean'].mean():.1f} +- {q4.describe()['mean'].std():.1f}")

            data_out.append([col_name, q1.describe()['mean'].mean(), q1.describe()['mean'].std(),
                             q4.describe()['mean'].mean(), q4.describe()['mean'].std(),
                             ttest['dof'].iloc[0], ttest['T'].iloc[0], ttest['p-val'].iloc[0]])

        print(f"\nt({ttest['dof'].iloc[0]}) = {ttest['T'].iloc[0]:.3f}, p = {ttest['p-val'].iloc[0]:.3f}")

    return pd.DataFrame(data_out, columns=cols_out)


def run_cutpoint_ttest(df):

    df_ttest = pd.concat([pg.ttest(df['fraysse_sedp'], df['powell_sedp'], paired=True),
                          pg.ttest(df['fraysse_lightp'], df['powell_lightp'], paired=True),
                          pg.ttest(df['fraysse_modp'], df['powell_modp'], paired=True)]
                         )
    df_ttest.index = ['sedp', 'lightp', 'modp']

    return df_ttest
