import pandas as pd
import nimbalwear.activity
import os
import numpy as np


def create_wrist_filenames(df_demos, edf_dict=None, fname_dict=None):

    fnames = []
    for row in df_demos.itertuples():
        try:
            fname = edf_dict[row.study_code] + fname_dict[row.study_code].format(row.subject_id, row.UseSide[0].capitalize()) + '.edf'
        except AttributeError:
            fname = edf_dict[row.study_code] + fname_dict[row.study_code].format(row.subject_id, row.Hand[0].capitalize()) + '.edf'

        fnames.append(fname)

    return fnames


def create_epochfiles(df_demos, full_ids=None, save_dir='C:/Users/ksweber/Desktop/Processed/'):
    """Runs nwactivity.activity_wrist_avm() and saves output for each wrisit file specified in df_demos['wrist_edf']
       and the full_ids specified using full_ids."""

    print(" ====== WRIST FILE EPOCHING =======")

    failed = []
    output_filenames = []

    if full_ids is None:
        df = df_demos
    if full_ids is not None:
        df = df_demos.loc[df_demos['full_id'].isin(full_ids)]
        print("\nOnly processing subjects: ", full_ids)

    n_proc = df.shape[0]
    curr_subj = 1

    for row in df_demos.itertuples():

        output_filename = "{}{}_EpochedWrist.csv".format(save_dir, row.full_id)

        if row.full_id in list(df['full_id']):
            print(f"\n{row.study_code}_{row.subject_id}: {curr_subj}/{n_proc}")

            curr_subj += 1

            try:

                data = nwdata.NWData()
                data.import_edf(row.wrist_edf)

                sig_ind = {"x": data.get_signal_index("Accelerometer x"),
                           "y": data.get_signal_index("Accelerometer y"),
                           "z": data.get_signal_index("Accelerometer z")}

                print("-Processing...")

                start_key = 'start_datetime' if 'start_datetime' in data.header.keys() else 'startdate'
                df_act = nwactivity.activity_wrist_avm(x=data.signals[sig_ind['x']],
                                                       y=data.signals[sig_ind['y']],
                                                       z=data.signals[sig_ind['z']],
                                                       epoch_length=1,
                                                       start_datetime=data.header[start_key],
                                                       sample_rate=data.signal_headers[sig_ind['x']]['sample_rate'],
                                                       sptw=pd.DataFrame({"sptw_num": []}),
                                                       sleep_bouts=pd.DataFrame({"sptw_num": []}),
                                                       cutpoint='Fraysse', dominant=True,
                                                       quiet=True)[0]
                print("    -Counts calculated.")

                print("    -Timestamps generated.")

                df_act.to_csv(output_filename, index=False)

                output_filenames.append(output_filename)

            except:
                failed.append(row.full_id)
                output_filenames.append(None)

        if row.full_id not in list(df['full_id']):
            output_filenames.append(None)

    return output_filenames, failed


def create_epochfiles2(df_demos, full_id=None, save_dir='C:/Users/ksweber/Desktop/Processed/', save_file=False):
    """Runs nwactivity.activity_wrist_avm() and saves output for each wrisit file specified in df_demos['wrist_edf']
       and the full_ids specified using full_ids."""

    df = df_demos.loc[df_demos['full_id'] == full_id]

    output_filename = "{}{}_EpochedWrist.csv".format(save_dir, full_id)

    df_act = None

    try:
        data = nimbalwear.data.Device()
        data.import_edf(df.iloc[0]["wrist_edf"])

        sig_ind = {"x": data.get_signal_index("Accelerometer x"),
                   "y": data.get_signal_index("Accelerometer y"),
                   "z": data.get_signal_index("Accelerometer z")}

        print("-Processing activity counts...")

        start_key = 'start_datetime' if 'start_datetime' in data.header.keys() else 'startdate'
        avm = nimbalwear.activity.activity_wrist_avm(x=data.signals[sig_ind['x']],
                                                     y=data.signals[sig_ind['y']],
                                                     z=data.signals[sig_ind['z']],
                                                     epoch_length=1,
                                                     start_datetime=data.header[start_key],
                                                     sample_rate=data.signal_headers[sig_ind['x']]['sample_rate'],
                                                     sptw=pd.DataFrame({"sptw_num": []}),
                                                     sleep_bouts=pd.DataFrame({"sptw_num": []}),
                                                     cutpoint='Fraysse', dominant=True,
                                                     quiet=True)[4]

        df_act = pd.DataFrame({'epoch_num': np.arange(1, len(avm) + 1),
                               'timestamp': pd.date_range(start=data.header[start_key], periods=len(avm), freq='1S'),
                               'avm': avm})

        print("    -Counts calculated.")

        if save_file:
            df_act.to_csv(output_filename, index=False)

    except:
        print("\nData epoching failed")


    return df_act


def create_wrist_epoch_filenames(df, folder):

    files = []

    for row in df.itertuples():
        fname = folder + f"{row.full_id}_EpochedWrist.csv"
        files.append(fname)

    return files
