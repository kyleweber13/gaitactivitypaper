import os
import pandas as pd
from datetime import timedelta
import WristProcessing
import StepsProcessing
import BoutProcessing
import DataImport
import IntensityProcessing
import Other
import FreeLiving


def run_loop(df_demos, full_ids=(), cutpoints=(62.5, 92.5), correct_ond09_cadence=True,
             min_step_time=60/200, min_bout_dur=60, min_cadence=80, mrp=5, save_files=True,
             remove_edge_low_cadence=True, file_dict=None, root_dir=""):

    failed = []
    subj_dict = {}

    fail_stage = 'start'

    for subj in sorted(full_ids):
        print(f"\n========== {subj} ==========")
        try:

            """ ============= Wrist epoching =========== """
            df_wrist = None

            if os.path.exists(f'{root_dir}EpochedWrist/{subj}_EpochedWrist.csv'):
                fail_stage = 'reading epoch file'

                df_wrist = DataImport.import_wrist_epoch(filename=f'{root_dir}EpochedWrist/{subj}_EpochedWrist.csv')
                print("     -Epoch file imported")

            if not os.path.exists(f'{root_dir}EpochedWrist/{subj}_EpochedWrist.csv'):
                print("-Epoched wrist file does not exist. Running epoching processing...")
                print("     -RUNNING SECONDARY CREATE_EPOCHFILES2 FUNCTIONS")

                fail_stage = 'generating epoch file'
                df_wrist = WristProcessing.create_epochfiles2(df_demos, full_id=subj,
                                                              save_dir=f'{root_dir}EpochedWrist/', save_file=False)

            fail_stage = 'epoch file formatting'

            if 'timestamp' not in df_wrist.columns and 'start_time' not in df_wrist.columns:
                df_wrist['start_time'] = pd.date_range(start=df_demos.loc[df_demos['full_id'] == subj]['wrist_starts'].iloc[0],
                                                       periods=df_wrist.shape[0], freq='1S')

            if 'timestamp' in df_wrist.columns and 'start_time' not in df_wrist.columns:
                df_wrist.columns = [i if i != 'timestamp' else 'start_time' for i in df_wrist.columns]

            if 'end_time' not in df_wrist.columns:
                e_len = int((df_wrist.iloc[1]['start_time'] - df_wrist.iloc[0]['start_time']).total_seconds())
                df_wrist['end_time'] = [i + timedelta(seconds=e_len) for i in df_wrist['start_time']]

            fail_stage = 'nonwear/sleep context'
            # if 'nw' not in df_wrist.columns or 'sleep' not in df_wrist.columns:
            df_wrist = DataImport.create_context_mask(subj=subj, df_wrist_epochs=df_wrist, df_demos=df_demos,
                                                      file_dict=file_dict)

            # within-bout epoching done later; this is for free-living
            fail_stage = '15-second re-epoching'
            df_wrist15 = IntensityProcessing.epoch_intensity(df_wrist_1s=df_wrist, cutpoints=cutpoints,
                                                             epoch_len=15, author='Fraysse')
            print("     -Wrist processing complete")

            """ ============= Step detection =========== """
            steps_filename = f"{root_dir}Steps/{subj}_01_GAIT_STEPS.csv"

            if not os.path.exists(steps_filename):
                print("\n-Steps file does not exist. Running nwgait...")
                fail_stage = 'running nwgait'
                a, b = StepsProcessing.run_nwgait(df=df_demos, subjs=(subj),
                                                  save_file=True, save_dir=f"{root_dir}Steps/")

            if os.path.exists(steps_filename):
                fail_stage = 'steps import'
                df_steps = DataImport.import_steps_file(steps_filename)

                if min_step_time > 0:
                    print(f"-Removing steps < {min_step_time:.1f} seconds apart...")
                    df_steps = StepsProcessing.remove_toosoon_steps(df_steps, min_step_time=min_step_time)

            """ ============= Gait bout processing ============ """

            fail_stage = 'bout detection'

            if 'OND09' in subj and correct_ond09_cadence:
                print("\nOND09 participant with single ankle file --> correcting bout thresholds")
                min_cadence /= 2

            df_bouts = BoutProcessing.find_bouts(peaks_inds=df_steps['step_index'], subj=subj,
                                                 fs=df_demos.loc[df_demos['full_id'] == subj].iloc[0]['ankle_fs'],
                                                 min_steps=3, min_duration=min_bout_dur,
                                                 max_break=mrp, min_cadence=min_cadence,
                                                 show_plot=False, quiet=True,
                                                 start_time=df_demos.loc[df_demos['full_id'] == subj]['ankle_starts'].iloc[0])

            fail_stage = 'bout processing'
            df_epoch_intensity = IntensityProcessing.process_bouts(df_bouts=df_bouts, df_1s_epochs=df_wrist,
                                                                   df_steps=df_steps, subj=subj,
                                                                   epoch_len=15, study_code=subj.split("_")[0],
                                                                   method='crop', cutpoints=cutpoints,
                                                                   cutpoint_name='Fraysse',
                                                                   show_plot=False, save_dir=None)

            if remove_edge_low_cadence:
                fail_stage = 'low cadence epoch removal'
                df_epoch_intensity = BoutProcessing.remove_low_cadence_edge_epochs(df_epochs=df_epoch_intensity,
                                                                                   min_cadence=min_cadence)
            if not remove_edge_low_cadence:
                print("-'Edge' epochs with low cadences NOT being removed")

            fail_stage = 'walking bout context'
            df_epoch_intensity = BoutProcessing.calculate_context_in_walkingbouts(df_wrist_epochs=df_wrist,
                                                                                  df_epoch_intensity=df_epoch_intensity)

            fail_stage = "calculating bout data"
            df_walk_intensity = IntensityProcessing.calculate_bout_data(full_id=subj,
                                                                        df_epoch_intensity=df_epoch_intensity)

            fail_stage = 'calculating participant totals'
            df_totals = BoutProcessing.calculate_intensity_totals_participant(df_epoch_intensity=df_epoch_intensity,
                                                                              df_demos=df_demos)

            fail_stage = 'calculating daily values'
            df_daily = FreeLiving.calculate_daily_values(full_id=subj,  n_valid_hours=10, df_wrist=df_wrist15,
                                                         gait_file_dict=None, save_file=None)

            # Corrects cadences for OND09 participant (single ankle)
            if 'OND09' in subj and correct_ond09_cadence:
                fail_stage = 'correcting cadences'
                df_bouts['cadence'] *= 2

                df_epoch_intensity['cadence'] *= 2
                df_epoch_intensity['number_steps'] *= 2

                df_totals['med_cadence'] *= 2
                df_totals['sd_cadence'] *= 2

            fail_stage = 'output formatting'
            subj_dict[subj] = {"wrist1": df_wrist, 'wrist15': df_wrist15,
                               'steps': df_steps, 'bouts': df_bouts,
                               'epoch_intensity': df_epoch_intensity,
                               'walk_intensity': df_walk_intensity, 'total_intensity': df_totals,
                               'daily': df_daily}

            if save_files:
                fail_stage = 'saving files'
                Other.save_dataframes([[df_epoch_intensity, f"{root_dir}WalkingEpochs/{subj}_WalkEpochs.csv"],
                                       [df_bouts, f"{root_dir}ProcessedBouts/{subj}_WalkingBouts.csv"],
                                       [df_wrist, f"{root_dir}EpochedWrist/{subj}_EpochedWrist.csv"],
                                       [df_wrist15, f"{root_dir}EpochedWrist/{subj}_EpochedWrist15.csv"],
                                       [df_walk_intensity, f"{root_dir}WalkingBouts/{subj}_ProcessedBouts.csv"],
                                       [df_daily, f"{root_dir}DailyStats/{subj}_FreeLivingDaily.csv"],
                                       [df_totals, f"{root_dir}WalkingTotals/{subj}_WalkingTotals.csv"]])
                print("\nFiles saved.")

        except:
            failed.append([subj, fail_stage])

    return failed, subj_dict
