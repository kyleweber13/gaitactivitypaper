import os
import pandas as pd


def save_dataframes(obj_list=()):
    """Takes a list of lists where each list's index[0] is a dataframe and index[1] is the pathway to save to.
       Will write to csv or xlsx based on given filename

        sample: save_dataframes([[df_bout_intensity, f"C:/Users/ksweber/Desktop/{subj}_WalkEpochs.csv"],
                                 [df_bouts, f"C:/Users/ksweber/Desktop/Processed/{subj}_WalkingBouts.xlsx"]])
    """

    for obj in obj_list:
        if 'csv' in obj[1]:
            obj[0].to_csv(obj[1], index=False)
        if 'xlsx' in obj[1]:
            obj[0].to_excel(obj[1], index=False)
        print(f"Saved a file to {obj[1]}")


def create_cohort_ids(df, ctrl_flag='CTRL'):

    ndd_dict = {}

    if 'NDD' in df.keys():
        ndds = df['NDD'].replace({"None": ctrl_flag}).unique()
    if 'NDD' not in df.keys():
        ndds = df['ndd'].replace({"None": ctrl_flag}).unique()
        df['NDD'] = df['ndd'].copy()

    for d in ndds:
        ndd_dict[d] = 1

    cohort_ids = []

    for i in df['NDD'].replace({"None": ctrl_flag}):
        cohort_id = f"{i}{ndd_dict[i]}"
        cohort_ids.append(cohort_id)
        ndd_dict[i] += 1

    return cohort_ids


def combine_dataframes(folder, keyword="", full_ids=None):

    files = os.listdir(folder)
    files = [i for i in files if keyword in i]

    print(f"\nCombining files: {files}")

    df_out = pd.read_csv(folder + files[0]) if "csv" in files[0] else pd.read_excel(folder + files[0])

    print(f"\nCombining {len(files)} files from {folder}...")

    if len(files) > 1:
        for file in files[1:]:

            if full_ids is None:
                df = pd.read_csv(folder + file) if "csv" in file else pd.read_excel(folder + file)
                df_out = pd.concat(objs=[df_out, df])

            if full_ids is not None:
                s = file.split("_")
                file_id = s[0] + "_" + s[1]

                if file_id not in full_ids:
                    print(f"-Skipping {file}")
                if file_id in full_ids:
                    df = pd.read_csv(folder + file) if "csv" in file else pd.read_excel(folder + file)
                    df_out = pd.concat(objs=[df_out, df])

    return df_out


def copy_cohort_ids(df_copy, df_new):

    id_dict = {}

    for row in df_copy.itertuples():
        id_dict[row.full_id] = row.cohort_id

    cohort_ids = [id_dict[row.full_id] for row in df_new.itertuples()]

    return cohort_ids
