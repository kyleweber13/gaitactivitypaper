import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
from datetime import timedelta
import numpy as np
import pandas as pd
import nwdata
from nimbalwear.activity import activity_wrist_avm as nw_act
import scipy.stats
import seaborn as sns

""" THIS REQUIRES NEWER VERSIONS OF MATPLOTLIB THAT SUPPORT LEGEND FORMATTING """


def plot_bouts(ankle_obj, wrist_obj,
               df_wrist=None, epoch_intensity=None, df_walk_epochs=None, cutpoints=(62.5, 92.5),
               ankle_axis='x', wrist_axis='x',
               df_long_bouts=None, df_all_bouts=None, df_steps=None, use_median_cadence=True, bout_steps_only=False):

    cad_type = 'median' if use_median_cadence else 'mean'

    if df_walk_epochs is not None:
        fig, ax = plt.subplots(4, sharex='col', figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1, .5]})

    if df_walk_epochs is None:
        fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

    ax[0].plot(ankle_obj.ts, ankle_obj.signals[ankle_obj.get_signal_index(f"{ankle_axis}")], color='black', zorder=0)

    long_fill_val = [0, 1] if df_all_bouts is None else [.5, 1]
    all_fill_val = [0, 1] if df_long_bouts is None else [0, .5]

    if df_long_bouts is not None:
        for row in df_long_bouts.itertuples():
            ax[0].axvspan(xmin=ankle_obj.ts[int(row.start)], xmax=ankle_obj.ts[int(row.end)],
                          ymin=long_fill_val[0], ymax=long_fill_val[1], color='gold', alpha=.35)

            if 'start_time' in df_long_bouts.columns:
                ax[0].text(x=row.start_time + timedelta(seconds=row.duration/2), y=3, s=f"#{row.Index}", color='red')
            if 'start_timestamp' in df_long_bouts.columns:
                ax[0].text(x=row.start_timestamp + timedelta(seconds=row.duration/2), y=3, s=f"#{row.Index}", color='red')

            if bout_steps_only and df_steps is not None:
                df_steps_bout = df_steps.loc[(df_steps['step_time'] >= ankle_obj.ts[row.start]) &
                                             (df_steps['step_time'] < ankle_obj.ts[row.end])]

                ax[0].scatter(ankle_obj.ts[df_steps_bout['step_index']],
                              [5 if 'Accelerometer' in ankle_axis else 250]*df_steps_bout.shape[0],
                              color='limegreen', s=25, marker='v', zorder=1)

    if df_steps is not None and not bout_steps_only:
        ax[0].scatter(ankle_obj.ts[df_steps['step_index']],
                      [5 if 'Accelerometer' in ankle_axis else 250]*df_steps.shape[0],
                      color='limegreen', s=25, marker='v', zorder=1)

    if df_all_bouts is not None:
        for row in df_all_bouts.itertuples():
            try:
                ax[0].axvspan(xmin=ankle_obj.ts[int(row.start)], xmax=ankle_obj.ts[int(row.end)],
                              ymin=all_fill_val[0], ymax=all_fill_val[1], color='dodgerblue', alpha=.35)
            except (KeyError, AttributeError):
                ax[0].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp,
                              ymin=all_fill_val[0], ymax=all_fill_val[1], color='dodgerblue', alpha=.35)

    ax[0].set_title("Raw Ankle {} Data with steps marked "
                    "({} bouts)\n ({} {})".format(ankle_axis, 'all' if not bout_steps_only else 'long',
                                                  "all bouts in blue, " if df_all_bouts is not None else "",
                                                  "long bouts in yellow" if df_long_bouts is not None else ""))
    ax[0].set_ylabel("G")

    ax[1].plot(wrist_obj.ts, wrist_obj.signals[wrist_obj.get_signal_index(f"{wrist_axis}")], color='black', zorder=0)
    ax[1].set_title(f"Raw Wrist {wrist_axis} Data")

    if df_wrist is not None:
        epoch_len = int((df_wrist.iloc[1]['start_time'] - df_wrist.iloc[0]['start_time']).total_seconds())
        ax[2].set_title(f"Wrist AVM ({epoch_len}-second epochs)")
        ax[2].plot(df_wrist['start_time'], df_wrist['avm'], color='black')
        ax[2].axhline(y=0, color='grey', linestyle='dotted')
        ax[2].axhline(y=cutpoints[0], color='limegreen', linestyle='dotted')
        ax[2].axhline(y=cutpoints[1], color='orange', linestyle='dotted')

        ax[2].set_ylim(0, )

    if epoch_intensity is not None:
        c = {'sedentary': 'grey', 'light': 'limegreen', 'moderate': 'orange'}

        for row in epoch_intensity.itertuples():
            ax[2].plot([row.start_time, row.start_time + timedelta(seconds=row.epoch_dur - .5)],
                       [row.avm, row.avm], color=c[row.intensity], lw=5, alpha=.8)

            ax[3].plot([row.start_time, row.start_time + timedelta(seconds=row.epoch_dur)],
                       [row.cadence, row.cadence], color='black', zorder=1, lw=2)

        ax[2].set_ylabel("AVM")

    if df_walk_epochs is not None:
        cols = df_walk_epochs.columns
        if use_median_cadence and 'med_cadence' not in cols:
            print("-Median cadences not found in dataframe. Using mean cadences.")
            cad_type = 'mean'

        ax[3].set_title("Cadence (green = walk's {} epoch, black = epochs)".format(cad_type))
        for row in df_walk_epochs.itertuples():

            if use_median_cadence and 'med_cadence' in cols:
                ax[3].plot([row.start_time, row.end_time],
                           [row.med_cadence, row.med_cadence], color='limegreen', lw=5, zorder=0)

            if not use_median_cadence or 'med_cadence' not in cols:
                if 'mean_cadence' in cols:
                    ax[3].plot([row.start_time if 'start_time' in df_walk_epochs.columns else row.start_timestamp,
                                row.end_time if 'end_time' in df_walk_epochs.columns else row.end_timestamp],
                               [row.mean_cadence, row.mean_cadence], color='limegreen', lw=5, zorder=0)
                if 'mean_cadence' not in cols:
                    ax[3].plot([row.start_time if 'start_time' in df_walk_epochs.columns else row.start_timestamp,
                                row.end_time if 'end_time' in df_walk_epochs.columns else row.end_timestamp],
                               [row.cadence, row.cadence], color='limegreen', lw=5, zorder=0)

        ax[3].grid()
        ax[3].set_ylabel("Cadence (spm)")

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig


def intensity_barplot(df, cp_author='fraysse', figsize=(10, 6), df_sig=None, sig_icon="*",
                      ytick_subjs='cohort_id', greyscale=True, incl_legend=False, lw=1.5,
                      fontsize=12, legend_fontsize=10,
                      binary_mvpa=False, binary_activity=False, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    if not binary_activity and not binary_mvpa:
        ax.barh(y=np.arange(1, df.shape[0] + 1), width=df[f'{cp_author}_sedp'], lw=lw,
                color='grey' if not greyscale else 'white', edgecolor='black', label='Sedentary', alpha=.8)
        ax.barh(y=np.arange(1, df.shape[0] + 1), width=df[f'{cp_author}_lightp'], left=df[f'{cp_author}_sedp'], lw=lw,
                color='green' if not greyscale else 'lightgrey', edgecolor='black', label='Light', alpha=.8)
        ax.barh(y=np.arange(1, df.shape[0] + 1), width=df[f'{cp_author}_modp'],
                left=df[f'{cp_author}_sedp'] + df[f'{cp_author}_lightp'], lw=lw,
                color='orange' if not greyscale else 'grey', edgecolor='black', label='MVPA', alpha=.8)

    if binary_activity and not binary_mvpa:
        ax.barh(y=np.arange(1, df.shape[0] + 1), width=df[f'{cp_author}_sedp'],
                color='grey' if not greyscale else 'black', edgecolor='black', lw=lw,
                label='Sedentary', alpha=.8)
        ax.barh(y=np.arange(1, df.shape[0] + 1), width=df[f'{cp_author}_lightp'] + df[f'{cp_author}_modp'],
                left=df[f'{cp_author}_sedp'], lw=lw,
                color='green' if not greyscale else 'darkgrey', edgecolor='black', label='Active', alpha=.8)

    if not binary_activity and binary_mvpa:
        ax.barh(y=np.arange(1, df.shape[0] + 1), width=df[f'{cp_author}_sedp'] + df[f'{cp_author}_lightp'], lw=lw,
                color='grey' if not greyscale else 'black', edgecolor='black', label='Sed/light', alpha=.8)
        ax.barh(y=np.arange(1, df.shape[0] + 1), width=df[f'{cp_author}_modp'],
                left=df[f'{cp_author}_sedp'] + df[f'{cp_author}_lightp'], lw=lw,
                color='orange' if not greyscale else 'darkgrey', edgecolor='black', label='MVPA', alpha=.8)

    if binary_activity and binary_mvpa:
        print("Got conflicting types of data. Try again.")

    ax.set_yticks(np.arange(1, df.shape[0] + 1))
    # ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
    ax.set_xticks([10, 30, 50, 70, 90], minor=True)

    if type(ytick_subjs) is str:
        ax.set_yticklabels(df[ytick_subjs], fontsize=fontsize)

    ax.set_xlim(0, 100)
    ax.set_ylim(.4, df.shape[0] + .6)

    ax.set_ylabel("Participants", fontsize=fontsize, labelpad=16)

    if incl_legend:
        ax.legend(bbox_to_anchor=[1, 1], fontsize=legend_fontsize,
                  title='Intensity\nclassification',
                  title_fontproperties={"weight": 'bold', 'size': legend_fontsize + 2})

    plt.tight_layout()

    box = ax.get_position()
    box_height = box.max[1] - box.min[1]

    if df_sig is not None:
        df_sig = df_sig.set_index('full_id')
        df_sig = df_sig.loc[list(df['full_id'])]
        for row in df.itertuples():
            sig_row = df_sig.loc[row.full_id]

            # if sig_row['sig'].iloc[0]:
            if sig_row['sig']:
                plt.gcf().text(box.max[0] + .025,
                               (box_height / (df.shape[0] + 1)) * (df.shape[0] - 1 - row.Index + .625) + box.min[1],
                               sig_icon, fontsize=fontsize)

        for tick, row in zip(ax.yaxis.get_major_ticks(), df_sig.itertuples()):
            if row.sig:
                tick.label1.set_fontweight('bold')

    plt.subplots_adjust(left=.15)

    return fig


def data_sections(full_id, window_start, window_end, df_demos,
                  wrist_up, wrist_ant, wrist_med, ankle_axis, epoch_len=1,
                  edf_folders=None, norm_ankle=False,
                  force_wrist_ylim=(None, None), wrist_yticks=None,
                  force_ankle_ylim=(None, None), ankle_yticks=None,
                  force_raw_vm_ylim=(None, None), force_avm_ylim=(None, None),
                  fig_width=7, show_legend=False, show_legend2=True,
                  steps=None, colors=None, alpha=1.0, avm_bar=False,
                  raw_wrist=False, region_name="Test",
                  use_grid=True, pad_window=15, regions=(), rem_ankle_base=True,
                  wrist=None, ankle=None, save_dir=None):

    def fig2_plot1(wrist, ankle, wrist_up, wrist_ant, wrist_med, ankle_axis, window_start, window_end,
                   fig_width=7, pad_window=15, colors=None, alpha=(.33), shade_regions=(), save_dir=None,
                   force_wrist_ylim=(None, None), wrist_yticks=None,
                   force_ankle_ylim=(None, None), ankle_yticks=None,
                   rem_ankle_base=True,
                   full_id="", use_grid=False, show_legend=False, region_name="Test", norm_ankle=False):

        if colors is None:
            colors = ['grey'] * len(shade_regions)

        wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
        ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

        wrist_start = wrist.header['startdate']
        ankle_start = ankle.header['startdate']

        wrist_idx = [int((pd.to_datetime(window_start) - wrist_start).total_seconds() * wrist_fs),
                     int((pd.to_datetime(window_end) - wrist_start).total_seconds() * wrist_fs)]
        ankle_idx = [int((pd.to_datetime(window_start) - ankle_start).total_seconds() * ankle_fs),
                     int((pd.to_datetime(window_end) - ankle_start).total_seconds() * ankle_fs)]

        t = np.arange(wrist_idx[-1] - wrist_idx[0]) / wrist_fs
        t = [i - pad_window for i in t]

        fig, ax = plt.subplots(2, sharex='col', figsize=(fig_width, 6))
        ax[0].plot(t, wrist_up[wrist_idx[0]:wrist_idx[1]], color='black', label='Up')
        ax[0].plot(t, wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red', label='Uln/Rad')
        ax[0].plot(t, wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue', label='Med.')

        if force_wrist_ylim[0] is None and force_wrist_ylim[1] is None:
            ylim0 = ax[0].get_ylim()
        if force_wrist_ylim[0] is not None or force_wrist_ylim[1] is not None:
            ylim0 = force_wrist_ylim

        for region, color, a in zip(shade_regions, colors, alpha):
            ax[0].axvspan(xmin=(pd.to_datetime(region[0]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                          xmax=(pd.to_datetime(region[1]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                          ymin=0, ymax=1, color=color, alpha=a)

        ax[0].set_ylim(ylim0)

        if wrist_yticks is not None:
            ax[0].set_yticks(wrist_yticks)

        if show_legend:
            ax[0].legend(loc='lower right')

        ax[0].set_ylabel("G")
        ax[0].set_title("Raw {} Wrist Accelerometer ({}Hz)".format("Right" if "RWrist" in wrist.header['patient_additional'] else 'Left',
                                                                   int(wrist_fs)))

        use_ankle_data = ankle_axis['data'][ankle_idx[0]:ankle_idx[1]]

        if rem_ankle_base:
            mean_a = np.mean(use_ankle_data)
            use_ankle_data = [i - mean_a for i in use_ankle_data]

        if norm_ankle:
            ankle_min = min(use_ankle_data)
            ankle_max = max(use_ankle_data)
            ankle_range = ankle_max - ankle_min
            use_ankle_data = [(i - ankle_min) / (ankle_range) * 2 - 1 for i in use_ankle_data]

        ax[1].plot(t, use_ankle_data, color='red')

        if force_ankle_ylim[0] is None and force_ankle_ylim[1] is None:
            ylim1 = ax[1].get_ylim()
        if force_ankle_ylim[0] is not None or force_ankle_ylim[1] is not None:
            ylim1 = force_ankle_ylim

        for region, color, a in zip(shade_regions, colors, alpha):
            ax[1].axvspan(xmin=(pd.to_datetime(region[0]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                          xmax=(pd.to_datetime(region[1]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                          ymin=0, ymax=1, color=color, alpha=a)

        if not norm_ankle:
            ax[1].set_ylabel("G")
            ax[1].set_ylim(ylim1)

            if ankle_yticks is not None:
                ax[1].set_yticks(ankle_yticks)

        if norm_ankle:
            ax[1].set_ylabel("G (normalized)")
            ax[1].set_ylim(-1.05, 1.05)
            ax[1].set_yticks([-1, 0, 1])

        ax[1].set_xlim(-pad_window, max(t)+pad_window+.05)

        ax[1].set_title("{} Ankle Accelerometer Axis ({} Hz)".format(ankle_axis['label'], int(ankle_fs)))
        ax[1].set_xlabel("Seconds")

        if use_grid:
            ax[0].grid()
            ax[1].grid()

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir + full_id + f"Plot1_{region_name}.tiff", dpi=125)

        return fig

    def fig2_plot2(wrist, ankle, ankle_axis, regions=(), colors=None, full_id="", epoch_len=1, show_legend2=False,
                   force_ankle_ylim=(None, None), force_raw_vm_ylim=(None, None), force_avm_ylim=(None, None),
                   alpha=(.33), fig_width=7, save_dir=None, use_grid=False, rem_ankle_base=True, avm_bar=False):

        wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
        ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

        fig, ax = plt.subplots(3, len(regions), sharex='col', sharey='row', figsize=(fig_width, 6))

        # Loops through rows of data
        for i, region in enumerate(regions):

            if 'timestamp' in steps.columns:
                window_steps = steps.loc[(steps['timestamp'] >= pd.to_datetime(region[0])) &
                                         (steps['timestamp'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['timestamp'] - window_steps.iloc[0]['timestamp']).total_seconds()
            if 'step_time' in steps.columns:
                window_steps = steps.loc[(steps['step_time'] >= pd.to_datetime(region[0])) &
                                         (steps['step_time'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['step_time'] - window_steps.iloc[0]['step_time']).total_seconds()

            n_steps = window_steps.shape[0] - 1
            cad = 60 * 2 * (n_steps / step_dur) if "OND09" in full_id else 60 * (n_steps / step_dur)
            # cad = 60 * n_steps / step_dur

            print(f"Window #{i + 1} ({region[0]} to {region[1]}): cadence = {cad:.1f} steps/min")

            # Indexes for whole data regions
            wrist_idx = [int((pd.to_datetime(region[0]) - wrist.header['startdate']).total_seconds() * wrist_fs),
                         int((pd.to_datetime(region[1]) - wrist.header['startdate']).total_seconds() * wrist_fs)]
            ankle_idx = [int((pd.to_datetime(region[0]) - ankle.header['startdate']).total_seconds() * ankle_fs),
                         int((pd.to_datetime(region[1]) - ankle.header['startdate']).total_seconds() * ankle_fs)]

            # time in seconds
            t = np.arange(wrist_idx[-1] + 1 - wrist_idx[0]) / wrist_fs

            # magnitude and AVM data
            wrist_vm = np.sqrt(
                np.square([wrist.signals[wrist.get_signal_index('Accelerometer x')][wrist_idx[0]:wrist_idx[1]+int(wrist_fs)],
                           wrist.signals[wrist.get_signal_index('Accelerometer y')][wrist_idx[0]:wrist_idx[1]+int(wrist_fs)],
                           wrist.signals[wrist.get_signal_index('Accelerometer z')][wrist_idx[0]:wrist_idx[1]+int(wrist_fs)]]).sum(
                    axis=0)) - 1
            wrist_vm[wrist_vm < 0] = 0
            wrist_vm *= 1000
            avm = [np.mean(wrist_vm[i:i + int(wrist_fs*epoch_len)]) for i in np.arange(0, len(wrist_vm), int(wrist_fs*epoch_len))]

            # wrist VM
            # ax[0][i].plot(t, wrist_vm[:len(t)], color='black')
            ax[0][i].plot(t, wrist_vm[:len(t)], color='black')
            ax[0][i].set_title("{} Wrist VM ({}Hz)".format("Right" if "RWrist" in wrist.header['patient_additional'] else 'Left',
                                                           int(wrist.signal_headers[wrist.get_signal_index("Accelerometer x")]['sample_rate'])))

            if None not in force_raw_vm_ylim:
                ax[0][i].set_ylim(force_raw_vm_ylim)

            # y-axis formatting for left-most column
            if i == 0:

                ax[0][i].set_ylabel("mG")
                ax[0][i].set_yticks(np.arange(0, 5000, 250))
                ax[0][i].set_ylim(-20, )

                ax[1][i].set_ylabel("mG")
                ax[2][i].set_ylabel("G")

            # wrist AVM w/ cutpoints
            if not avm_bar:
                #ax[1][i].plot(np.arange(0, len(avm))*epoch_len, avm, color='black',
                #              marker='o' if avm_markers else None, markeredgecolor='black', markerfacecolor='black')
                for epoch, a in enumerate(avm):
                    ax[1][i].plot([epoch_len * epoch, epoch_len * (epoch + 1) - .1], [a, a], color='black', lw=2)

            if avm_bar:
                ax[1][i].bar(np.arange(0, len(avm))*epoch_len, avm, align='edge', width=epoch_len, color='darkgrey', edgecolor='black')
            ax[1][i].axhline(y=92.5, color='red', linestyle='dashed', label='Mod.')
            ax[1][i].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Light')
            ax[1][i].set_title(f"Wrist AVM ({epoch_len}-second epochs)")

            if None not in force_avm_ylim:
                ax[1][i].set_ylim(force_avm_ylim)

            # raw ankle data
            c_dict = {'Vertical': 'black', 'AP': 'red', 'ML': 'dodgerblue'}

            if rem_ankle_base:
                mean_a = np.mean(ankle_axis['data'][ankle_idx[0]:ankle_idx[1]+1])
                ax[2][i].plot(t, [i - mean_a for i in ankle_axis['data'][ankle_idx[0]:ankle_idx[1]+1]],
                              color=c_dict[ankle_axis['label']])

            if not rem_ankle_base:
                ax[2][i].plot(t, ankle_axis['data'][ankle_idx[0]:ankle_idx[1]], color=c_dict[ankle_axis['label']])

            ax[2][i].set_title("{} Ankle Axis ({}Hz) - {} steps/min".format(ankle_axis['label'], int(ankle_fs), round(cad, 1)))
            ax[2][i].set_xlabel("Seconds")
            ax[2][i].set_xlim(0, (pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds() + .1)

            if i == 0:
                ax[2][i].set_ylabel("G")

            if use_grid:
                ax[0][i].grid()
                ax[1][i].grid()
                ax[2][i].grid()

        if colors is not None:
            for row in range(3):

                ymins = []
                ymaxes = []
                for cols in range(len(regions)):
                    ymins.append(ax[row][cols].get_ylim()[0])
                    ymaxes.append(ax[row][cols].get_ylim()[1])

                ylims = [min(ymins), max(ymaxes)]

                if row == 2 and force_ankle_ylim[0] is not None and force_ankle_ylim[1] is not None:
                    ylims = force_ankle_ylim

                for cols in range(len(regions)):
                    ax[row][cols].axvspan(xmin=ax[row][cols].get_xlim()[0], xmax=ax[row][cols].get_xlim()[1],
                                          ymin=0, ymax=1,
                                          color=colors[cols], alpha=alpha[cols])
                    ax[row][cols].set_ylim(ylims)

        if show_legend2:
            ax[1][1].legend(bbox_to_anchor=[1.09, .38], loc='center')

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir + full_id + "Plot2.tiff", dpi=125)

        return fig

    def fig2_plot2_raw(wrist, ankle, wrist_up, wrist_ant, wrist_med, ankle_axis, regions=(), fig_width=7,
                       force_wrist_ylim=(None, None), force_ankle_ylim=(None, None), show_legend2=False,
                       force_raw_vm_ylim=(None, None), force_avm_ylim=(None, None), epoch_len=1,
                       full_id="", colors=None, alpha=(.33), save_dir=None, use_grid=False, rem_ankle_base=True):

        wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
        ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

        fig, ax = plt.subplots(4, len(regions), sharex='col', sharey='row', figsize=(fig_width, 8))

        # Loops through rows of data
        for i, region in enumerate(regions):

            # Indexes for whole data regions
            wrist_idx = [int((pd.to_datetime(region[0]) - wrist.header['startdate']).total_seconds() * wrist_fs),
                         int((pd.to_datetime(region[1]) - wrist.header['startdate']).total_seconds() * wrist_fs)]
            ankle_idx = [int((pd.to_datetime(region[0]) - ankle.header['startdate']).total_seconds() * ankle_fs),
                         int((pd.to_datetime(region[1]) - ankle.header['startdate']).total_seconds() * ankle_fs)]

            # time in seconds
            t = np.arange(wrist_idx[-1] + 1 - wrist_idx[0]) / wrist_fs

            # magnitude and AVM data
            wrist_vm = np.sqrt(
                np.square([wrist.signals[wrist.get_signal_index('Accelerometer x')][
                           wrist_idx[0]:wrist_idx[1] + int(wrist_fs)],
                           wrist.signals[wrist.get_signal_index('Accelerometer y')][
                           wrist_idx[0]:wrist_idx[1] + int(wrist_fs)],
                           wrist.signals[wrist.get_signal_index('Accelerometer z')][
                           wrist_idx[0]:wrist_idx[1] + int(wrist_fs)]]).sum(
                    axis=0)) - 1
            wrist_vm[wrist_vm < 0] = 0
            avm = [np.mean(wrist_vm[i:i + int(wrist_fs*epoch_len)] * 1000) for i in np.arange(0, len(wrist_vm), int(wrist_fs*epoch_len))]

            # raw data
            ax[0][i].plot(t[:-1], wrist_up[wrist_idx[0]:wrist_idx[1]], color='black')
            ax[0][i].plot(t[:-1], wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red')
            ax[0][i].plot(t[:-1], wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue')
            ax[0][i].set_title(f"Raw Wrist Acceleration ({wrist_fs}Hz)")

            if None not in force_raw_vm_ylim:
                ax[1][i].set_ylim(force_raw_vm_ylim)

            # wrist VM
            ax[1][i].plot(t, wrist_vm[:len(t)], color='black')
            ax[1][i].set_title("{} Wrist VM ({}Hz)".format("Right" if "RWrist" in wrist.header['patient_additional'] else 'Left',
                                                           int(wrist.signal_headers[wrist.get_signal_index("Accelerometer x")]['sample_rate'])))

            # y-axis formatting for left-most column
            if i == 0:

                if force_wrist_ylim[0] is not None or force_wrist_ylim[1] is not None:
                    ax[0][i].set_ylim(force_wrist_ylim)

                ax[0][i].set_ylabel("G")
                ax[1][i].set_ylabel("G")
                ax[1][i].set_yticks([0, .5, 1])
                ax[1][i].set_ylim(-.02, )

                ax[2][i].set_ylabel("mG")
                ax[3][i].set_ylabel("G")

            # wrist AVM w/ cutpoints
            ax[2][i].plot(np.arange(0, len(avm))*epoch_len, avm, color='black', label=f"{epoch_len}-sec AVM")
            ax[2][i].axhline(y=92.5, color='red', linestyle='dashed', label='Mod.')
            ax[2][i].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Light')
            ax[2][i].set_title(f"Wrist AVM ({epoch_len}-second epochs)")

            if show_legend:
                ax[2][i].legend()

            if None not in force_avm_ylim:
                ax[2][i].set_ylim(force_avm_ylim)

            # raw ankle data
            c_dict = {'Vertical': 'black', 'AP': 'red', 'ML': 'dodgerblue'}

            if rem_ankle_base:
                mean_a = np.mean(ankle_axis['data'][ankle_idx[0]:ankle_idx[1]+1])
                ax[3][i].plot(t, [i - mean_a for i in ankle_axis['data'][ankle_idx[0]:ankle_idx[1]+1]],
                              color=c_dict[ankle_axis['label']])

            if not rem_ankle_base:
                ax[3][i].plot(t, ankle_axis['data'][ankle_idx[0]:ankle_idx[1]+1], color=c_dict[ankle_axis['label']])

            ax[3][i].set_title("Ankle {} axis".format(ankle_axis['label']))
            ax[3][i].set_xlabel("Seconds")
            ax[3][i].set_xlim(0, (pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds() + .1)

            if i == 0:
                ax[1][i].set_ylabel("G")

            if use_grid:
                ax[0][i].grid()
                ax[1][i].grid()
                ax[2][i].grid()
                ax[3][i].grid()

        if colors is not None:
            for row in range(4):

                ymins = []
                ymaxes = []
                for cols in range(len(regions)):
                    ymins.append(ax[row][cols].get_ylim()[0])
                    ymaxes.append(ax[row][cols].get_ylim()[1])

                ylims = [min(ymins), max(ymaxes)]

                if row == 3 and force_ankle_ylim[0] is not None and force_ankle_ylim[1] is not None:
                    ylims = force_ankle_ylim

                for cols in range(len(regions)):
                    ax[row][cols].axvspan(xmin=ax[row][cols].get_xlim()[0], xmax=ax[row][cols].get_xlim()[1],
                                          ymin=0, ymax=1,
                                          color=colors[cols], alpha=alpha[cols])
                    ax[row][cols].set_ylim(ylims)

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir + full_id + "Plot2.tiff", dpi=125)

        return fig

    def fig2_plot2_v2(wrist, ankle, ankle_axis, regions=(), colors=None, full_id="", epoch_len=1, show_legend2=False,
                      force_wrist_ylim=(None, None), force_ankle_ylim=(None, None),
                      force_raw_vm_ylim=(None, None), force_avm_ylim=(None, None),
                      alpha=(.33), fig_width=7, save_dir=None, use_grid=False, rem_ankle_base=True, avm_bar=False,
                      region_name="", raw_wrist=True, norm_ankle=False):

        wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
        ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

        fig, ax = plt.subplots(3 if not raw_wrist else 4, len(regions),
                               sharex='col', sharey='row', figsize=(fig_width, 6 if not raw_wrist else 8))

        # Loops through regions (columns) of data
        for i, region in enumerate(regions):

            if len(regions) == 1:
                col_ax = ax
            if len(regions) > 1:
                col_ax = ax[:, i]

            if 'timestamp' in steps.columns:
                window_steps = steps.loc[(steps['timestamp'] >= pd.to_datetime(region[0])) &
                                         (steps['timestamp'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['timestamp'] - window_steps.iloc[0]['timestamp']).total_seconds()
            if 'step_time' in steps.columns:
                window_steps = steps.loc[(steps['step_time'] >= pd.to_datetime(region[0])) &
                                         (steps['step_time'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['step_time'] - window_steps.iloc[0]['step_time']).total_seconds()

            n_steps = window_steps.shape[0] - 1
            cad = 60 * 2 * (n_steps / step_dur) if "OND09" in full_id else 60 * (n_steps / step_dur)

            print(f"Window #{i + 1} ({region[0]} to {region[1]}): cadence = {cad:.1f} steps/min")

            # Indexes for whole data regions
            wrist_idx = [int((pd.to_datetime(region[0]) - wrist.header['startdate']).total_seconds() * wrist_fs),
                         int((pd.to_datetime(region[1]) - wrist.header['startdate']).total_seconds() * wrist_fs)]

            ankle_idx = [int((pd.to_datetime(region[0]) - ankle.header['startdate']).total_seconds() * ankle_fs),
                         int((pd.to_datetime(region[1]) - ankle.header['startdate']).total_seconds() * ankle_fs)]

            # time in seconds
            t = np.arange(wrist_idx[-1] + 1 - wrist_idx[0]) / wrist_fs

            # raw triaxial wrist
            ax_offset = 0  # index for axes depending on if showing raw or not

            if raw_wrist:
                col_ax[0].plot(t[:-1], wrist_up[wrist_idx[0]:wrist_idx[1]], color='black')
                col_ax[0].plot(t[:-1], wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red')
                col_ax[0].plot(t[:-1], wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue')
                col_ax[0].set_title(f"Raw Wrist Acceleration ({int(wrist_fs)}Hz)")
                ax_offset = 1

            # magnitude and AVM data
            wrist_vm = nw_act(x=wrist.signals[wrist.get_signal_index('Accelerometer x')][wrist_idx[0]:wrist_idx[1]+int(wrist_fs)],
                              y=wrist.signals[wrist.get_signal_index('Accelerometer y')][wrist_idx[0]:wrist_idx[1]+int(wrist_fs)],
                              z=wrist.signals[wrist.get_signal_index('Accelerometer z')][wrist_idx[0]:wrist_idx[1]+int(wrist_fs)],
                              epoch_length=epoch_len, sample_rate=wrist_fs,
                              start_datetime=pd.to_datetime(region[0]), quiet=True)[3]

            wrist_vm *= 1000
            avm = [np.mean(wrist_vm[i:i + int(wrist_fs*epoch_len)]) for i in
                   np.arange(0, len(wrist_vm), int(wrist_fs*epoch_len))]

            # wrist VM
            col_ax[ax_offset].plot(t, wrist_vm[:len(t)], color='black')
            col_ax[ax_offset].set_title("{} Wrist VM ({}Hz)".format("Right" if "RWrist" in wrist.header['patient_additional'] else 'Left',
                                                                    int(wrist_fs)))

            if None not in force_raw_vm_ylim:
                col_ax[ax_offset].set_ylim(force_raw_vm_ylim)

            # y-axis formatting for left-most column (i=0)
            if i == 0:
                col_ax[0].set_ylabel("G")  # overridden if ax_offset = 0 (when not raw_wrist)
                col_ax[0+ax_offset].set_ylabel("mG")

                col_ax[1+ax_offset].set_ylabel("mG")
                col_ax[2+ax_offset].set_ylabel("G{}".format(" (normalized)" if norm_ankle else ""))

            # wrist AVM w/ cutpoints
            if not avm_bar:
                for epoch, a in enumerate(avm):
                    col_ax[1+ax_offset].plot([epoch_len * epoch, epoch_len * (epoch + 1) - .1], [a, a],
                                             color='black', lw=2)

            if avm_bar:
                col_ax[1+ax_offset].bar(np.arange(0, len(avm))*epoch_len, avm,
                                        align='edge', width=epoch_len, color='darkgrey', edgecolor='black')

            col_ax[1+ax_offset].axhline(y=92.5, color='red', linestyle='dashed', label='Mod.')
            col_ax[1+ax_offset].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Light')
            col_ax[1+ax_offset].set_title(f"Wrist AVM ({epoch_len}-second epochs)")

            if None not in force_avm_ylim:
                col_ax[1+ax_offset].set_ylim(force_avm_ylim)

            col_ax[1+ax_offset].set_ylim(0, 150)

            # raw ankle data
            c_dict = {'Vertical': 'black', 'AP': 'red', 'ML': 'dodgerblue'}

            use_ankle_data = ankle_axis['data'][ankle_idx[0]:ankle_idx[1]]

            if rem_ankle_base:
                mean_a = np.mean(use_ankle_data)
                use_ankle_data = [i - mean_a for i in use_ankle_data]

            if norm_ankle:
                ankle_min = min(use_ankle_data)
                ankle_max = max(use_ankle_data)
                ankle_range = ankle_max - ankle_min
                use_ankle_data = [(i - ankle_min) / (ankle_range) * 2 - 1 for i in use_ankle_data]

            col_ax[2+ax_offset].plot(t[:min([len(t), len(use_ankle_data)])],
                                     use_ankle_data[:min([len(t), len(use_ankle_data)])],
                                     color=c_dict[ankle_axis['label']])

            col_ax[2+ax_offset].set_title("{} Ankle Axis ({}Hz) - {} steps/min".format(ankle_axis['label'],
                                                                                       int(ankle_fs), round(cad, 1)))
            col_ax[2+ax_offset].set_xlabel("Seconds")
            col_ax[2+ax_offset].set_xlim(0, (pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds() + .1)

            if use_grid:
                col_ax[0].grid()
                col_ax[1].grid()
                col_ax[2].grid()

                if raw_wrist:
                    col_ax[3].grid()

        if len(regions) == 1:
            if None not in force_avm_ylim:
                col_ax[1+ax_offset].set_ylim(force_avm_ylim)

        if colors is not None:
            # loops through rows
            for row in range(3 if not raw_wrist else 4):

                if len(regions) == 1:
                    row_ax = ax
                if len(regions) > 1:
                    row_ax = ax[row, :]

                ymins = []
                ymaxes = []

                if len(regions) > 1:
                    for cols in range(len(regions)):
                        ymins.append(row_ax[cols].get_ylim()[0])
                        ymaxes.append(row_ax[cols].get_ylim()[1])
                if len(regions) == 1:
                    ymins = [row_ax[row].get_ylim()[0]]
                    ymaxes = [row_ax[row].get_ylim()[1]]
                    cols = 0

                ylims = [min(ymins), max(ymaxes)]

                # formatting for raw wrist
                if len(regions) == 1 and row == 0 and None not in force_wrist_ylim:
                    ylims = force_wrist_ylim

                # formatting for AVM
                if len(regions) == 1 and row == 1+ax_offset and None not in force_avm_ylim:
                    ylims = force_avm_ylim

                # formatting for ankle data
                if row == 2+ax_offset and force_ankle_ylim[0] is not None and force_ankle_ylim[1] is not None:
                    if not norm_ankle:
                        ylims = force_ankle_ylim
                    if norm_ankle:
                        ylims = (-1.05, 1.05)

                if len(regions) > 1:
                    for cols in range(len(regions)):
                        row_ax[cols].axvspan(xmin=row_ax[cols].get_xlim()[0], xmax=row_ax[cols].get_xlim()[1],
                                             ymin=0, ymax=1, color=colors[cols], alpha=alpha[cols])
                        row_ax[cols].set_ylim(ylims)

                        # formatting for normalized ankle data
                        if row == 2+ax_offset and norm_ankle:
                            row_ax[cols].set_yticks([-1, 0, 1])

                if len(regions) == 1:
                    row_ax[row].axvspan(xmin=row_ax[row].get_xlim()[0], xmax=row_ax[row].get_xlim()[1],
                                        ymin=0, ymax=1, color=colors[0], alpha=alpha[0])
                    row_ax[row].set_ylim(ylims)

                    # formatting for normalized ankle data
                    if row == 2+ax_offset and norm_ankle:
                        row_ax[cols].set_yticks([-1, 0, 1])

        if show_legend2:
            col_ax[1+ax_offset].legend(bbox_to_anchor=[1.09, .38], loc='center')

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir + full_id + f"Plot2_{region_name}.tiff", dpi=125)

        return fig

    def fig2_plot2_v3(wrist, ankle, ankle_axis, regions=(), colors=None, full_id="", epoch_len=1, show_legend2=False,
                      force_wrist_ylim=(None, None), wrist_yticks=None,
                      force_ankle_ylim=(None, None), ankle_yticks=None,
                      force_raw_vm_ylim=(None, None), force_avm_ylim=(None, None),
                      alpha=(.33), fig_width=7, save_dir=None, use_grid=False, rem_ankle_base=True, avm_bar=False,
                      region_name="", raw_wrist=True, norm_ankle=False):

        wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
        ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

        fig, ax = plt.subplots(3 if not raw_wrist else 4, len(regions),
                               sharex='col', sharey='row', figsize=(fig_width, 6 if not raw_wrist else 8))

        # Loops through regions (columns) of data
        for i, region in enumerate(regions):

            if len(regions) == 1:
                col_ax = ax
            if len(regions) > 1:
                col_ax = ax[:, i]

            if 'timestamp' in steps.columns:
                window_steps = steps.loc[(steps['timestamp'] >= pd.to_datetime(region[0])) &
                                         (steps['timestamp'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['timestamp'] - window_steps.iloc[0]['timestamp']).total_seconds()
            if 'step_time' in steps.columns:
                window_steps = steps.loc[(steps['step_time'] >= pd.to_datetime(region[0])) &
                                         (steps['step_time'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['step_time'] - window_steps.iloc[0]['step_time']).total_seconds()

            n_steps = window_steps.shape[0] - 1
            cad = 60 * 2 * (n_steps / step_dur) if "OND09" in full_id else 60 * (n_steps / step_dur)

            print(f"Window #{i + 1} ({region[0]} to {region[1]}): cadence = {cad:.1f} steps/min")

            # Indexes for whole data regions
            wrist_idx = [int((pd.to_datetime(region[0]) - wrist.header['startdate']).total_seconds() * wrist_fs),
                         int((pd.to_datetime(region[1]) - wrist.header['startdate']).total_seconds() * wrist_fs)]

            ankle_idx = [int((pd.to_datetime(region[0]) - ankle.header['startdate']).total_seconds() * ankle_fs),
                         int((pd.to_datetime(region[1]) - ankle.header['startdate']).total_seconds() * ankle_fs)]

            # time in seconds
            t = np.arange(wrist_idx[-1] + 1 - wrist_idx[0]) / wrist_fs

            # raw triaxial wrist
            ax_offset = 0  # index for axes depending on if showing raw or not

            if raw_wrist:
                col_ax[0].plot(t[:-1], wrist_up[wrist_idx[0]:wrist_idx[1]], color='black')
                col_ax[0].plot(t[:-1], wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red')
                col_ax[0].plot(t[:-1], wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue')
                col_ax[0].set_title(f"Raw Wrist Acceleration ({int(wrist_fs)}Hz)")
                ax_offset = 1

            # magnitude and AVM data
            wrist_vm = nw_act(x=wrist.signals[wrist.get_signal_index('Accelerometer x')][wrist_idx[0]:wrist_idx[1]],
                              y=wrist.signals[wrist.get_signal_index('Accelerometer y')][wrist_idx[0]:wrist_idx[1]],
                              z=wrist.signals[wrist.get_signal_index('Accelerometer z')][wrist_idx[0]:wrist_idx[1]],
                              epoch_length=epoch_len, sample_rate=wrist_fs,
                              start_datetime=pd.to_datetime(region[0]), quiet=True)[3]

            wrist_vm *= 1000
            avm = [np.mean(wrist_vm[i:i + int(wrist_fs*epoch_len)]) for i in
                   np.arange(0, len(wrist_vm), int(wrist_fs*epoch_len))]

            # wrist VM
            col_ax[ax_offset].plot(t[:min([len(t), len(wrist_vm)])], wrist_vm[:min([len(t), len(wrist_vm)])], color='black')
            col_ax[ax_offset].set_title("{} Wrist VM ({}Hz)".format("Right" if "RWrist" in wrist.header['patient_additional'] else 'Left',
                                                                    int(wrist_fs)))
            col_ax[0].set_xlim(-.1, (pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds() +.1)

            if None not in force_raw_vm_ylim:
                col_ax[ax_offset].set_ylim(force_raw_vm_ylim)

            # y-axis formatting for left-most column (i=0)
            if i == 0:
                col_ax[0].set_ylabel("G")  # overridden if ax_offset = 0 (when not raw_wrist)
                col_ax[0+ax_offset].set_ylabel("mG")
                col_ax[1+ax_offset].set_ylabel("mG")
                col_ax[2+ax_offset].set_ylabel("G{}".format(" (normalized)" if norm_ankle else ""))

            # wrist AVM w/ cutpoints
            if not avm_bar:
                for epoch, a in enumerate(avm):
                    col_ax[1+ax_offset].plot([epoch_len * epoch, epoch_len * (epoch + 1) - .1], [a, a],
                                             color='black', lw=2)

            if avm_bar:
                col_ax[1+ax_offset].bar(np.arange(0, len(avm))*epoch_len, avm,
                                        align='edge', width=epoch_len, color='darkgrey', edgecolor='black')

            col_ax[1+ax_offset].axhline(y=92.5, color='red', linestyle='dashed', label='Mod.')
            col_ax[1+ax_offset].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Light')
            col_ax[1+ax_offset].set_title(f"Wrist AVM ({epoch_len}-second epochs)")

            if None not in force_avm_ylim:
                col_ax[1+ax_offset].set_ylim(force_avm_ylim)

            # raw ankle data
            c_dict = {'Vertical': 'black', 'AP': 'red', 'ML': 'dodgerblue'}

            use_ankle_data = ankle_axis['data'][ankle_idx[0]:ankle_idx[1]]

            if rem_ankle_base:
                mean_a = np.mean(use_ankle_data)
                use_ankle_data = [i - mean_a for i in use_ankle_data]

            if norm_ankle:
                ankle_min = min(use_ankle_data)
                ankle_max = max(use_ankle_data)
                ankle_range = ankle_max - ankle_min
                use_ankle_data = [(i - ankle_min) / (ankle_range) * 2 - 1 for i in use_ankle_data]

            col_ax[2+ax_offset].plot(t[:min([len(t), len(use_ankle_data)])],
                                     use_ankle_data[:min([len(t), len(use_ankle_data)])],
                                     color=c_dict[ankle_axis['label']])

            col_ax[2+ax_offset].set_title("{} Ankle Axis ({}Hz) - {} steps/min".format(ankle_axis['label'],
                                                                                       int(ankle_fs), round(cad, 1)))
            col_ax[2+ax_offset].set_xlabel("Seconds")

            if use_grid:
                col_ax[0].grid()
                col_ax[1].grid()
                col_ax[2].grid()

                if raw_wrist:
                    col_ax[3].grid()

        """ ========================= axis formatting for one-region data segments ========================= """
        if len(regions) == 1:
            # raw wrist axis formatting ------
            if raw_wrist:
                if None in force_wrist_ylim:
                    ylims = [ax[0].get_ylim()[0], ax[0].get_ylim()[1]]
                if None not in force_wrist_ylim:
                    ylims = force_wrist_ylim

                ax[0].set_ylim(ylims)

                if wrist_yticks is not None:
                    ax[0].set_yticks(wrist_yticks)

                ax[0].axvspan(xmin=ax[0].get_xlim()[0], xmax=ax[0].get_xlim()[1],
                              ymin=0, ymax=1, color=colors[0], alpha=alpha[0])

            # raw VM axis formatting ------
            if None in force_raw_vm_ylim:
                ylims = [ax[ax_offset].get_xlim()[0], ax[ax_offset].get_xlim()[1]]
            if None not in force_raw_vm_ylim:
                ylims = force_raw_vm_ylim

            ax[ax_offset].axvspan(xmin=ax[ax_offset].get_xlim()[0], xmax=ax[ax_offset].get_xlim()[1],
                                  ymin=0, ymax=1, color=colors[0], alpha=alpha[0])

            # AVM axis formatting -----
            if None in force_avm_ylim:
                ylims = [ax[1+ax_offset].get_xlim()[0], ax[1+ax_offset].get_xlim()[1]]
            if None not in force_avm_ylim:
                ylims = force_avm_ylim

            ax[1+ax_offset].axvspan(xmin=ax[1+ax_offset].get_xlim()[0], xmax=ax[1+ax_offset].get_xlim()[1],
                                    ymin=0, ymax=1, color=colors[0], alpha=alpha[0])

            # Ankle axis formatting -----
            if norm_ankle:
                ylims = [-1.05, 1.05]

            if not norm_ankle:
                if None in force_ankle_ylim:
                    ylims = [ax[2+ax_offset].get_xlim()[0], ax[2+ax_offset].get_xlim()[1]]
                if None not in force_ankle_ylim:
                    ylims = force_ankle_ylim

            ax[2+ax_offset].set_ylim(ylims)

            if ankle_yticks is not None and not norm_ankle:
                ax[2+ax_offset].set_yticks(ankle_yticks)

            if norm_ankle:
                ax[2+ax_offset].set_yticks([-1, 0, 1])

            ax[2+ax_offset].axvspan(xmin=ax[2+ax_offset].get_xlim()[0], xmax=ax[2+ax_offset].get_xlim()[1],
                                    ymin=0, ymax=1, color=colors[0], alpha=alpha[0])

        """ ========================= axis formatting for multi-region data segments ========================= """
        if len(regions) >= 2:
            # loops through rows looking at all columns
            for row in range(3 if not raw_wrist else 4):
                row_ax = ax[row, :]

                ymins = []
                ymaxes = []

                # gets axis limits for each column of data for each given row
                for cols in range(len(regions)):
                    ymins.append(row_ax[cols].get_ylim()[0])
                    ymaxes.append(row_ax[cols].get_ylim()[1])

                ylims = [min(ymins), max(ymaxes)]

                if raw_wrist and row == 0:
                    if None not in force_wrist_ylim:
                        ylims = force_wrist_ylim

                    if wrist_yticks is not None:
                        row_ax[0].set_yticks(wrist_yticks)

                if row == ax_offset:
                    if None not in force_raw_vm_ylim:
                        ylims = force_raw_vm_ylim
                    if None in force_raw_vm_ylim:
                        ylims = row_ax[ax_offset].get_ylim()

                if row == 1 + ax_offset:
                    if None not in force_avm_ylim:
                        ylims = force_avm_ylim

                if row == 2 + ax_offset:
                    if None not in force_ankle_ylim:
                        ylims = force_ankle_ylim
                    if norm_ankle:
                        ylims = (-1.05, 1.05)

                for cols in range(len(regions)):
                    row_ax[cols].set_ylim(ylims)
                    row_ax[cols].axvspan(xmin=row_ax[cols].get_xlim()[0], xmax=row_ax[cols].get_xlim()[1],
                                         ymin=0, ymax=1, color=colors[cols], alpha=alpha[cols])

        if show_legend2:
            col_ax[1+ax_offset].legend(bbox_to_anchor=[1.09, .38], loc='center')

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir + full_id + f"Plot2_{region_name}.tiff", dpi=125)

        return fig

    if edf_folders is None:
        edf_folders = {"OND06": 'W:/NiMBaLWEAR/OND06/processed/cropped_device_edf/GNAC/OND06_{}_01_GNAC_{}{}.edf',
                       'OND09': 'W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_{}_01_AXV6_{}{}.edf'}
    study_code = full_id.split("_")[0]
    subj = full_id.split("_")[1]

    try:
        dom_hand = df_demos.loc[df_demos['full_id'] == full_id]['Hand'].iloc[0]
    except (KeyError, IndexError):
        print("No hand dominance data found. Defaulting to Right.")
        dom_hand = 'R'

    if wrist is None:
        print("-Importing wrist data...")
        wrist = nwdata.NWData()
        wrist.import_edf(edf_folders[study_code].format(subj, dom_hand, 'Wrist'))

    if ankle is None:
        print("-Importing ankle data...")
        ankle = nwdata.NWData()
        ankle.import_edf(edf_folders[study_code].format(subj, dom_hand, 'Ankle'))

    plot1 = fig2_plot1(wrist=wrist, ankle=ankle, colors=colors, alpha=alpha, fig_width=fig_width,
                       window_start=window_start, window_end=window_end,
                       rem_ankle_base=rem_ankle_base, ankle_axis=ankle_axis, norm_ankle=norm_ankle,
                       force_wrist_ylim=force_wrist_ylim, wrist_yticks=wrist_yticks,
                       force_ankle_ylim=force_ankle_ylim, ankle_yticks=ankle_yticks,
                       pad_window=pad_window, save_dir=save_dir, use_grid=use_grid,
                       shade_regions=regions, show_legend=show_legend, full_id=full_id, region_name=region_name,
                       wrist_up=wrist_up, wrist_ant=wrist_ant, wrist_med=wrist_med)

    # if not raw_wrist:
    plot2 = fig2_plot2_v3(wrist=wrist, ankle=ankle, colors=colors, alpha=alpha, fig_width=fig_width,
                          regions=regions, save_dir=save_dir, use_grid=use_grid, full_id=full_id, ankle_axis=ankle_axis,
                          force_wrist_ylim=force_wrist_ylim, wrist_yticks=wrist_yticks,
                          force_ankle_ylim=force_ankle_ylim, ankle_yticks=ankle_yticks,
                          force_avm_ylim=force_avm_ylim, epoch_len=epoch_len,
                          force_raw_vm_ylim=force_raw_vm_ylim, rem_ankle_base=rem_ankle_base, show_legend2=show_legend2,
                          avm_bar=avm_bar, region_name=region_name, raw_wrist=raw_wrist, norm_ankle=norm_ankle)
    """if raw_wrist:
        plot2 = fig2_plot2_raw(wrist=wrist, ankle=ankle, colors=colors, alpha=alpha, fig_width=fig_width,
                               regions=regions, save_dir=save_dir, use_grid=use_grid, full_id=full_id,
                               force_wrist_ylim=force_wrist_ylim, force_ankle_ylim=force_ankle_ylim,
                               force_avm_ylim=force_avm_ylim, force_raw_vm_ylim=force_raw_vm_ylim,
                               wrist_up=wrist_up, wrist_ant=wrist_ant, rem_ankle_base=rem_ankle_base,
                               wrist_med=wrist_med, ankle_axis=ankle_axis, epoch_len=epoch_len,
                               show_legend2=show_legend2)"""

    return plot1, plot2, wrist, ankle


def data_sections2(full_id, window_start, window_end, df_demos,
                   wrist_up, wrist_ant, wrist_med, ankle_axis, epoch_len=1,
                   edf_folders=None, norm_ankle=False,
                   linewidth=1, tick_fontsize=9, title_fontsize=10,
                   force_wrist_ylim=(None, None), wrist_yticks=None, zoomed_xticks=None,
                   force_ankle_ylim=(None, None), ankle_yticks=None,
                   force_raw_vm_ylim=(None, None), force_avm_ylim=(None, None),
                   fig_width=7, fig_height=10, include_yaxis=True,
                   show_legend=False, show_legend2=True,
                   steps=None, colors=None, alpha=1.0, avm_bar=False,
                   raw_wrist=False,
                   use_grid=True, pad_window=15, regions=(), rem_ankle_base=True,
                   wrist=None, ankle=None):

    # https://stackoverflow.com/questions/66880397/how-can-i-increase-horizontal-space-hspace-between-two-specific-matplotlib-sub
    # https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots

    """ ================================================================== """
    """ ========================== Data set-up =========================== """
    """ ================================================================== """
    if edf_folders is None:
        edf_folders = {"OND06": 'W:/NiMBaLWEAR/OND06/processed/cropped_device_edf/GNAC/OND06_{}_01_GNAC_{}{}.edf',
                       'OND09': 'W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_{}_01_AXV6_{}{}.edf'}
    study_code = full_id.split("_")[0]
    subj = full_id.split("_")[1]

    try:
        dom_hand = df_demos.loc[df_demos['full_id'] == full_id]['Hand'].iloc[0]
    except (KeyError, IndexError):
        print("No hand dominance data found. Defaulting to Right.")
        dom_hand = 'R'

    if wrist is None:
        print("-Importing wrist data...")
        wrist = nwdata.NWData()
        wrist.import_edf(edf_folders[study_code].format(subj, dom_hand, 'Wrist'))

    if ankle is None:
        print("-Importing ankle data...")
        ankle = nwdata.NWData()
        ankle.import_edf(edf_folders[study_code].format(subj, dom_hand, 'Ankle'))

    """ =============================== Plotting set up =============================== """
    # plt.rc('ytick', labelsize=tick_fontsize)
    # plt.rc('xtick', labelsize=tick_fontsize)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # subplot config
    subfigs = fig.subfigures(2, 1, height_ratios=[2, 4], hspace=.5)
    plotx = subfigs[0].subplots(2, subplot_kw={"zorder": 1}, sharex='col')
    plot0 = plotx[0]
    plot1 = plotx[1]
    plot2 = subfigs[1].subplots(4, len(regions), sharey='row', sharex='col', squeeze=True, subplot_kw={"zorder": 0})

    # ylab_box = dict(pad=8, facecolor='white', edgecolor='white', alpha=0)

    """ ================================================================== """
    """ ========================== Full window =========================== """
    """ ================================================================== """
    if colors is None:
        colors = ['grey'] * len(regions)

    wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
    ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

    wrist_start = wrist.header['startdate']
    ankle_start = ankle.header['startdate']
    window_start = pd.to_datetime(window_start)
    window_end = pd.to_datetime(window_end)

    wrist_idx = [int((window_start - wrist_start).total_seconds() * wrist_fs),
                 int((window_end - wrist_start).total_seconds() * wrist_fs)]
    ankle_idx = [int((window_start - ankle_start).total_seconds() * ankle_fs),
                 int((window_end - ankle_start).total_seconds() * ankle_fs)]

    t = np.arange(wrist_idx[-1] - wrist_idx[0]) / wrist_fs
    t = [i - pad_window for i in t]

    plot0.plot(t, wrist_up[wrist_idx[0]:wrist_idx[1]], color='black', label='Up', lw=linewidth)
    plot0.plot(t, wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red', label='Uln/Rad', lw=linewidth)
    plot0.plot(t, wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue', label='Med.', lw=linewidth)

    if force_wrist_ylim[0] is None and force_wrist_ylim[1] is None:
        ylim0 = plot0.get_ylim()
    if force_wrist_ylim[0] is not None or force_wrist_ylim[1] is not None:
        ylim0 = force_wrist_ylim

    for region, color, a in zip(regions, colors, alpha):
        plot0.axvspan(xmin=(pd.to_datetime(region[0]) - window_start).total_seconds() - pad_window,
                      xmax=(pd.to_datetime(region[1]) - window_start).total_seconds() - pad_window,
                      ymin=0, ymax=1, color=color, alpha=a)

    plot0.set_ylim(ylim0)

    if wrist_yticks is not None:
        plot0.set_yticks(wrist_yticks)

    if not include_yaxis:
        plot0.set_yticklabels([])

    if include_yaxis:
        plot0.set_ylabel("G", fontsize=title_fontsize, bbox=ylab_box)

    if show_legend:
        plot0.legend(loc='lower right')

    plot0.set_title("Raw {} Wrist Accelerometer ({}Hz)".format("Right" if "RWrist" in
                                                                wrist.header['patient_additional'] else 'Left',
                                                               int(wrist_fs)),
                    fontsize=title_fontsize)

    use_ankle_data = ankle_axis['data'][ankle_idx[0]:ankle_idx[1]]

    if rem_ankle_base:
        mean_a = np.mean(use_ankle_data)
        use_ankle_data = [i - mean_a for i in use_ankle_data]

    if norm_ankle:
        ankle_min = min(use_ankle_data)
        ankle_max = max(use_ankle_data)
        ankle_range = ankle_max - ankle_min
        use_ankle_data = [(i - ankle_min) / (ankle_range) * 2 - 1 for i in use_ankle_data]

    plot1.plot(t, use_ankle_data, color='red', lw=linewidth)

    if force_ankle_ylim[0] is None and force_ankle_ylim[1] is None:
        ylim1 = plot1.get_ylim()
    if force_ankle_ylim[0] is not None or force_ankle_ylim[1] is not None:
        ylim1 = force_ankle_ylim

    for region, color, a in zip(regions, colors, alpha):
        plot1.axvspan(xmin=(pd.to_datetime(region[0]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                      xmax=(pd.to_datetime(region[1]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                      ymin=0, ymax=1, color=color, alpha=a)

    if not norm_ankle:
        plot1.set_ylabel("G", fontsize=title_fontsize, bbox=ylab_box)
        plot1.set_ylim(ylim1)

        if ankle_yticks is not None:
            plot1.set_yticks(ankle_yticks, fontsize=tick_fontsize)

    if norm_ankle:
        plot1.set_ylabel("G (normalized)", fontsize=title_fontsize, bbox=ylab_box)
        plot1.set_ylim(-1.05, 1.05)
        plot1.set_yticks([-1, 0, 1])

    if not include_yaxis:
        plot1.set_yticks([])
        plot1.set_ylabel("")

    plot1.set_xlim(-pad_window, max(t) + pad_window + .05)

    plot1.set_title("{} Ankle Accelerometer Axis ({} Hz)".format(ankle_axis['label'], int(ankle_fs)),
                    fontsize=title_fontsize)

    if use_grid:
        plot0.grid()
        plot1.grid()

    """ ================================================================== """
    """ ========================= Zoomed window ========================== """
    """ ================================================================== """

    # zoomed windows ===================================================

    # Loops through regions (columns) of data
    for i, region in enumerate(regions):

        if len(regions) == 1:
            col_ax = plot2
        if len(regions) > 1:
            col_ax = plot2[:, i]

        for row in range(len(col_ax)):
            col_ax[row].patch.set_facecolor(color=colors[i])
            col_ax[row].patch.set_alpha(alpha[i])

        if 'timestamp' in steps.columns:
            window_steps = steps.loc[(steps['timestamp'] >= pd.to_datetime(region[0])) &
                                     (steps['timestamp'] < pd.to_datetime(region[1]))]
            step_dur = (window_steps.iloc[-1]['timestamp'] - window_steps.iloc[0]['timestamp']).total_seconds()
        if 'step_time' in steps.columns:
            window_steps = steps.loc[(steps['step_time'] >= pd.to_datetime(region[0])) &
                                     (steps['step_time'] < pd.to_datetime(region[1]))]
            step_dur = (window_steps.iloc[-1]['step_time'] - window_steps.iloc[0]['step_time']).total_seconds()

        n_steps = window_steps.shape[0] - 1
        cad = 60 * 2 * (n_steps / step_dur) if "OND09" in full_id else 60 * (n_steps / step_dur)

        print(f"Window #{i + 1} ({region[0]} to {region[1]}): cadence = {cad:.1f} steps/min")

        # Indexes for whole data regions
        wrist_idx = [int((pd.to_datetime(region[0]) - wrist.header['startdate']).total_seconds() * wrist_fs),
                     int((pd.to_datetime(region[1]) - wrist.header['startdate']).total_seconds() * wrist_fs)]

        ankle_idx = [int((pd.to_datetime(region[0]) - ankle.header['startdate']).total_seconds() * ankle_fs),
                     int((pd.to_datetime(region[1]) - ankle.header['startdate']).total_seconds() * ankle_fs)]

        # time in seconds
        t = np.arange(wrist_idx[-1] + 1 - wrist_idx[0]) / wrist_fs

        # raw triaxial wrist
        ax_offset = 0  # index for axes depending on if showing raw or not

        if raw_wrist:
            col_ax[0].plot(t[:-1], wrist_up[wrist_idx[0]:wrist_idx[1]], color='black', lw=linewidth)
            col_ax[0].plot(t[:-1], wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red', lw=linewidth)
            col_ax[0].plot(t[:-1], wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue', lw=linewidth)
            col_ax[0].set_title(f"Raw Wrist Acceleration ({int(wrist_fs)}Hz)", fontsize=title_fontsize)

            if force_wrist_ylim is not None:
                col_ax[0].set_ylim(force_wrist_ylim)

            ax_offset = 1

        # magnitude and AVM data
        wrist_vm = nw_act(x=wrist.signals[wrist.get_signal_index('Accelerometer x')][wrist_idx[0]:wrist_idx[1]],
                          y=wrist.signals[wrist.get_signal_index('Accelerometer y')][wrist_idx[0]:wrist_idx[1]],
                          z=wrist.signals[wrist.get_signal_index('Accelerometer z')][wrist_idx[0]:wrist_idx[1]],
                          epoch_length=epoch_len, sample_rate=wrist_fs,
                          start_datetime=pd.to_datetime(region[0]), quiet=True)[3]

        wrist_vm *= 1000
        avm = [np.mean(wrist_vm[i:i + int(wrist_fs * epoch_len)]) for i in
               np.arange(0, len(wrist_vm), int(wrist_fs * epoch_len))]

        # wrist VM
        col_ax[ax_offset].plot(t[:min([len(t), len(wrist_vm)])], wrist_vm[:min([len(t), len(wrist_vm)])],
                               color='black', lw=linewidth)
        col_ax[ax_offset].set_title("{} Wrist VM ({}Hz)".format("Right" if "RWrist" in wrist.header['patient_additional'] else 'Left',
                                                                int(wrist_fs)), fontsize=title_fontsize)
        col_ax[0].set_xlim(-.1, (pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds() + .1)

        if None not in force_raw_vm_ylim:
            col_ax[ax_offset].set_ylim(force_raw_vm_ylim)

        # y-axis formatting for left-most column (i=0)
        if i == 0:
            if include_yaxis:
                col_ax[0].set_ylabel("G", fontsize=title_fontsize, bbox=ylab_box)  # overridden if ax_offset = 0 (when not raw_wrist)
                col_ax[0 + ax_offset].set_ylabel("mG", fontsize=title_fontsize, bbox=ylab_box)
                col_ax[1 + ax_offset].set_ylabel("mG", fontsize=title_fontsize, bbox=ylab_box)
                col_ax[2 + ax_offset].set_ylabel("G{}".format(" (normalized)" if norm_ankle else ""),
                                                 fontsize=title_fontsize, bbox=ylab_box)

            if not include_yaxis:
                col_ax[0].set_yticklabels([])
                col_ax[1].set_yticklabels([])
                col_ax[2].set_yticklabels([])
                col_ax[3].set_yticklabels([])

        # wrist AVM w/ cutpoints
        if not avm_bar:
            for epoch, a in enumerate(avm):
                col_ax[1 + ax_offset].plot([epoch_len * epoch, epoch_len * (epoch + 1) - .1], [a, a],
                                           color='black', lw=2)

        if avm_bar:
            col_ax[1 + ax_offset].bar(np.arange(0, len(avm)) * epoch_len, avm,
                                      align='edge', width=epoch_len, color='darkgrey', edgecolor='black')

        col_ax[1 + ax_offset].axhline(y=92.5, color='red', linestyle='dashed', label='Mod.')
        col_ax[1 + ax_offset].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Light')
        col_ax[1 + ax_offset].set_title(f"Wrist AVM ({epoch_len}-second epochs)", fontsize=title_fontsize)

        if None not in force_avm_ylim:
            col_ax[1 + ax_offset].set_ylim(force_avm_ylim)

        # raw ankle data
        c_dict = {'Vertical': 'black', 'AP': 'red', 'ML': 'dodgerblue'}

        use_ankle_data = ankle_axis['data'][ankle_idx[0]:ankle_idx[1]]

        if rem_ankle_base:
            mean_a = np.mean(use_ankle_data)
            use_ankle_data = [i - mean_a for i in use_ankle_data]

        if norm_ankle:
            ankle_min = min(use_ankle_data)
            ankle_max = max(use_ankle_data)
            ankle_range = ankle_max - ankle_min
            use_ankle_data = [(i - ankle_min) / (ankle_range) * 2 - 1 for i in use_ankle_data]

        col_ax[2 + ax_offset].plot(t[:min([len(t), len(use_ankle_data)])],
                                   use_ankle_data[:min([len(t), len(use_ankle_data)])],
                                   color=c_dict[ankle_axis['label']], lw=linewidth)

        col_ax[2 + ax_offset].set_title("{} Ankle Axis ({}Hz) - {} steps/min".format(ankle_axis['label'],
                                                                                     int(ankle_fs), round(cad, 1)),
                                        fontsize=title_fontsize)
        col_ax[2 + ax_offset].set_xlabel("Seconds", fontsize=title_fontsize)

        if zoomed_xticks is not None:
            col_ax[2 + ax_offset].set_xticks(zoomed_xticks)

        if use_grid:
            col_ax[0].grid()
            col_ax[1].grid()
            col_ax[2].grid()

            if raw_wrist:
                col_ax[3].grid()

    for i, region in enumerate(regions):

        if len(regions) == 1:
            col_ax = plot2
        if len(regions) > 1:
            col_ax = plot2[:, i]

        con = ConnectionPatch(
            xyA=[(pd.to_datetime(region[0]) - window_start).total_seconds(), plot1.get_ylim()[0]],
            xyB=[col_ax[0].get_xlim()[0], col_ax[0].get_ylim()[1]],
            coordsA='data', coordsB='data',
            axesA=plot1, axesB=col_ax[0], color=colors[i], linestyle='dashed')
        col_ax[0].add_artist(con)

        con = ConnectionPatch(
            xyA=[(pd.to_datetime(region[1]) - window_start).total_seconds(), plot1.get_ylim()[0]],
            xyB=[col_ax[0].get_xlim()[1], col_ax[0].get_ylim()[1]],
            coordsA='data', coordsB='data',
            axesA=plot1, axesB=col_ax[0], color=colors[i], linestyle='dashed')
        col_ax[0].add_artist(con)

    """ ================================================================== """
    """ ======================= Final formatting ========================= """
    """ ================================================================== """

    subfigs[0].subplots_adjust(top=.925, hspace=.2, left=.15, right=.975, bottom=.1)
    subfigs[1].subplots_adjust(top=.925, hspace=.25, wspace=.1, left=.15, right=.975, bottom=.075)

    return fig, wrist, ankle


def data_sections3(full_id, window_start, window_end, df_demos,
                   wrist_up, wrist_ant, wrist_med, ankle_axis, epoch_len=1,
                   edf_folders=None, norm_ankle=False,
                   linewidth=1, tick_fontsize=9, title_fontsize=10,
                   force_wrist_ylim=(None, None), wrist_yticks=None, zoomed_xticks=None,
                   force_ankle_ylim=(None, None), ankle_yticks=None,
                   force_raw_vm_ylim=(None, None), force_avm_ylim=(None, None),
                   fig_width=7, fig_height=10, include_yaxis=True,
                   show_legend=False, show_legend2=True,
                   steps=None, colors=None, alpha=1.0, avm_bar=False,
                   raw_wrist=False,
                   use_grid=True, pad_window=15, regions=(), rem_ankle_base=True,
                   wrist=None, ankle=None):

    # https://stackoverflow.com/questions/66880397/how-can-i-increase-horizontal-space-hspace-between-two-specific-matplotlib-sub
    # https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots

    """ ================================================================== """
    """ ========================== Data set-up =========================== """
    """ ================================================================== """
    if edf_folders is None:
        edf_folders = {"OND06": 'W:/NiMBaLWEAR/OND06/processed/cropped_device_edf/GNAC/OND06_{}_01_GNAC_{}{}.edf',
                       'OND09': 'W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_{}_01_AXV6_{}{}.edf'}
    study_code = full_id.split("_")[0]
    subj = full_id.split("_")[1]

    try:
        dom_hand = df_demos.loc[df_demos['full_id'] == full_id]['Hand'].iloc[0]
    except (KeyError, IndexError):
        print("No hand dominance data found. Defaulting to Right.")
        dom_hand = 'R'

    if wrist is None:
        print("-Importing wrist data...")
        wrist = nwdata.NWData()
        wrist.import_edf(edf_folders[study_code].format(subj, dom_hand, 'Wrist'))

    if ankle is None:
        print("-Importing ankle data...")
        ankle = nwdata.NWData()
        ankle.import_edf(edf_folders[study_code].format(subj, dom_hand, 'Ankle'))

    """ =============================== Plotting set up =============================== """
    plt.rc('ytick', labelsize=tick_fontsize)
    plt.rc('xtick', labelsize=tick_fontsize)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # subplot config
    subfigs = fig.subfigures(2, 1, height_ratios=[2, 4], hspace=.5)
    plotx = subfigs[0].subplots(2, subplot_kw={"zorder": 1}, sharex='col')
    plot0 = plotx[0]
    plot1 = plotx[1]
    plot2 = subfigs[1].subplots(4, len(regions), sharey='row', sharex='col', squeeze=True, subplot_kw={"zorder": 0})

    ylab_box = dict(pad=8, facecolor='white', edgecolor='white', alpha=0)

    """ ================================================================== """
    """ ========================== Full window =========================== """
    """ ================================================================== """
    if colors is None:
        colors = ['grey'] * len(regions)

    wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
    ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

    wrist_start = wrist.header['startdate']
    ankle_start = ankle.header['startdate']
    window_start = pd.to_datetime(window_start)
    window_end = pd.to_datetime(window_end)

    wrist_idx = [int((window_start - wrist_start).total_seconds() * wrist_fs),
                 int((window_end - wrist_start).total_seconds() * wrist_fs)]
    ankle_idx = [int((window_start - ankle_start).total_seconds() * ankle_fs),
                 int((window_end - ankle_start).total_seconds() * ankle_fs)]

    t = np.arange(wrist_idx[-1] - wrist_idx[0]) / wrist_fs
    t = [i - pad_window for i in t]

    plot0.plot(t, wrist_up[wrist_idx[0]:wrist_idx[1]], color='black', label='Up', lw=linewidth)
    plot0.plot(t, wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red', label='Uln/Rad', lw=linewidth)
    plot0.plot(t, wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue', label='Med.', lw=linewidth)

    if force_wrist_ylim[0] is None and force_wrist_ylim[1] is None:
        ylim0 = plot0.get_ylim()
    if force_wrist_ylim[0] is not None or force_wrist_ylim[1] is not None:
        ylim0 = force_wrist_ylim

    for region, color, a in zip(regions, colors, alpha):
        plot0.axvspan(xmin=(pd.to_datetime(region[0]) - window_start).total_seconds() - pad_window,
                      xmax=(pd.to_datetime(region[1]) - window_start).total_seconds() - pad_window,
                      ymin=0, ymax=1, color=color, alpha=a)

    plot0.set_ylim(ylim0)

    if wrist_yticks is not None:
        plot0.set_yticks(wrist_yticks)

    if not include_yaxis:
        plot0.set_yticklabels([])

    if include_yaxis:
        plot0.set_ylabel("G", fontsize=title_fontsize, bbox=ylab_box)

    if show_legend:
        plot0.legend(loc='lower right')

    plot0.set_title("Raw {} Wrist Accelerometer ({}Hz)".format("Right" if "RWrist" in
                                                                wrist.header['patient_additional'] else 'Left',
                                                               int(wrist_fs)),
                    fontsize=title_fontsize)

    use_ankle_data = ankle_axis['data'][ankle_idx[0]:ankle_idx[1]]

    if rem_ankle_base:
        mean_a = np.mean(use_ankle_data)
        use_ankle_data = [i - mean_a for i in use_ankle_data]

    if norm_ankle:
        ankle_min = min(use_ankle_data)
        ankle_max = max(use_ankle_data)
        ankle_range = ankle_max - ankle_min
        use_ankle_data = [(i - ankle_min) / (ankle_range) * 2 - 1 for i in use_ankle_data]

    plot1.plot(t, use_ankle_data, color='red', lw=linewidth)

    if force_ankle_ylim[0] is None and force_ankle_ylim[1] is None:
        ylim1 = plot1.get_ylim()
    if force_ankle_ylim[0] is not None or force_ankle_ylim[1] is not None:
        ylim1 = force_ankle_ylim

    for region, color, a in zip(regions, colors, alpha):
        plot1.axvspan(xmin=(pd.to_datetime(region[0]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                      xmax=(pd.to_datetime(region[1]) - pd.to_datetime(window_start)).total_seconds() - pad_window,
                      ymin=0, ymax=1, color=color, alpha=a)

    if not norm_ankle:
        plot1.set_ylabel("G", fontsize=title_fontsize, bbox=ylab_box)
        plot1.set_ylim(ylim1)

        if ankle_yticks is not None:
            plot1.set_yticks(ankle_yticks, fontsize=tick_fontsize)

    if norm_ankle:
        plot1.set_ylabel("G (normalized)", fontsize=title_fontsize, bbox=ylab_box)
        plot1.set_ylim(-1.05, 1.05)
        plot1.set_yticks([-1, 0, 1])

    if not include_yaxis:
        plot1.set_yticks([])
        plot1.set_ylabel("")

    plot1.set_xlim(-pad_window, max(t) + pad_window + .05)

    plot1.set_title("{} Ankle Accelerometer Axis ({} Hz)".format(ankle_axis['label'], int(ankle_fs)),
                    fontsize=title_fontsize)

    if use_grid:
        plot0.grid()
        plot1.grid()

    """ ================================================================== """
    """ ========================= Zoomed window ========================== """
    """ ================================================================== """

    # zoomed windows ===================================================

    # Loops through regions (columns) of data
    for i, region in enumerate(regions):

        if len(regions) == 1:
            col_ax = plot2
        if len(regions) > 1:
            col_ax = plot2[:, i]

        for row in range(len(col_ax)):
            col_ax[row].patch.set_facecolor(color=colors[i])
            col_ax[row].patch.set_alpha(alpha[i])

        if 'timestamp' in steps.columns:
            window_steps = steps.loc[(steps['timestamp'] >= pd.to_datetime(region[0])) &
                                     (steps['timestamp'] < pd.to_datetime(region[1]))]
            step_dur = (window_steps.iloc[-1]['timestamp'] - window_steps.iloc[0]['timestamp']).total_seconds()
        if 'step_time' in steps.columns:
            window_steps = steps.loc[(steps['step_time'] >= pd.to_datetime(region[0])) &
                                     (steps['step_time'] < pd.to_datetime(region[1]))]
            step_dur = (window_steps.iloc[-1]['step_time'] - window_steps.iloc[0]['step_time']).total_seconds()

        n_steps = window_steps.shape[0] - 1
        cad = 60 * 2 * (n_steps / step_dur) if "OND09" in full_id else 60 * (n_steps / step_dur)

        print(f"Window #{i + 1} ({region[0]} to {region[1]}): cadence = {cad:.1f} steps/min")

        # Indexes for whole data regions
        wrist_idx = [int((pd.to_datetime(region[0]) - wrist.header['startdate']).total_seconds() * wrist_fs),
                     int((pd.to_datetime(region[1]) - wrist.header['startdate']).total_seconds() * wrist_fs)]

        ankle_idx = [int((pd.to_datetime(region[0]) - ankle.header['startdate']).total_seconds() * ankle_fs),
                     int((pd.to_datetime(region[1]) - ankle.header['startdate']).total_seconds() * ankle_fs)]

        # time in seconds
        t = np.arange(wrist_idx[-1] + 1 - wrist_idx[0]) / wrist_fs

        # raw triaxial wrist
        ax_offset = 0  # index for axes depending on if showing raw or not

        if raw_wrist:
            col_ax[0].plot(t[:-1], wrist_up[wrist_idx[0]:wrist_idx[1]], color='black', lw=linewidth)
            col_ax[0].plot(t[:-1], wrist_ant[wrist_idx[0]:wrist_idx[1]], color='red', lw=linewidth)
            col_ax[0].plot(t[:-1], wrist_med[wrist_idx[0]:wrist_idx[1]], color='dodgerblue', lw=linewidth)
            col_ax[0].set_title(f"Raw Wrist Acceleration ({int(wrist_fs)}Hz)", fontsize=title_fontsize)

            if force_wrist_ylim is not None:
                col_ax[0].set_ylim(force_wrist_ylim)

            ax_offset = 1

        # magnitude and AVM data
        wrist_vm = nw_act(x=wrist.signals[wrist.get_signal_index('Accelerometer x')][wrist_idx[0]:wrist_idx[1]],
                          y=wrist.signals[wrist.get_signal_index('Accelerometer y')][wrist_idx[0]:wrist_idx[1]],
                          z=wrist.signals[wrist.get_signal_index('Accelerometer z')][wrist_idx[0]:wrist_idx[1]],
                          epoch_length=epoch_len, sample_rate=wrist_fs,
                          start_datetime=pd.to_datetime(region[0]), quiet=True)[3]

        wrist_vm *= 1000
        avm = [np.mean(wrist_vm[i:i + int(wrist_fs * epoch_len)]) for i in
               np.arange(0, len(wrist_vm), int(wrist_fs * epoch_len))]

        # wrist VM
        col_ax[ax_offset].plot(t[:min([len(t), len(wrist_vm)])], wrist_vm[:min([len(t), len(wrist_vm)])],
                               color='black', lw=linewidth)
        col_ax[ax_offset].set_title("{} Wrist VM ({}Hz)".format("Right" if "RWrist" in wrist.header['patient_additional'] else 'Left',
                                                                int(wrist_fs)), fontsize=title_fontsize)
        col_ax[0].set_xlim(-.1, (pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds() + .1)

        if None not in force_raw_vm_ylim:
            col_ax[ax_offset].set_ylim(force_raw_vm_ylim)

        # y-axis formatting for left-most column (i=0)
        if i == 0:
            if include_yaxis:
                col_ax[0].set_ylabel("G", fontsize=title_fontsize, bbox=ylab_box)  # overridden if ax_offset = 0 (when not raw_wrist)
                col_ax[0 + ax_offset].set_ylabel("mG", fontsize=title_fontsize, bbox=ylab_box)
                col_ax[1 + ax_offset].set_ylabel("mG", fontsize=title_fontsize, bbox=ylab_box)
                col_ax[2 + ax_offset].set_ylabel("G{}".format(" (normalized)" if norm_ankle else ""),
                                                 fontsize=title_fontsize, bbox=ylab_box)

            if not include_yaxis:
                col_ax[0].set_yticklabels([])
                col_ax[1].set_yticklabels([])
                col_ax[2].set_yticklabels([])
                col_ax[3].set_yticklabels([])

        # wrist AVM w/ cutpoints
        if not avm_bar:
            for epoch, a in enumerate(avm):
                col_ax[1 + ax_offset].plot([epoch_len * epoch, epoch_len * (epoch + 1) - .1], [a, a],
                                           color='black', lw=2)

        if avm_bar:
            col_ax[1 + ax_offset].bar(np.arange(0, len(avm)) * epoch_len, avm,
                                      align='edge', width=epoch_len, color='darkgrey', edgecolor='black')

        col_ax[1 + ax_offset].axhline(y=92.5, color='red', linestyle='dashed', label='Mod.')
        col_ax[1 + ax_offset].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Light')
        col_ax[1 + ax_offset].set_title(f"Wrist AVM ({epoch_len}-second epochs)", fontsize=title_fontsize)

        if None not in force_avm_ylim:
            col_ax[1 + ax_offset].set_ylim(force_avm_ylim)

        # raw ankle data
        c_dict = {'Vertical': 'black', 'AP': 'red', 'ML': 'dodgerblue'}

        use_ankle_data = ankle_axis['data'][ankle_idx[0]:ankle_idx[1]]

        if rem_ankle_base:
            mean_a = np.mean(use_ankle_data)
            use_ankle_data = [i - mean_a for i in use_ankle_data]

        if norm_ankle:
            ankle_min = min(use_ankle_data)
            ankle_max = max(use_ankle_data)
            ankle_range = ankle_max - ankle_min
            use_ankle_data = [(i - ankle_min) / (ankle_range) * 2 - 1 for i in use_ankle_data]

        col_ax[2 + ax_offset].plot(t[:min([len(t), len(use_ankle_data)])],
                                   use_ankle_data[:min([len(t), len(use_ankle_data)])],
                                   color=c_dict[ankle_axis['label']], lw=linewidth)

        col_ax[2 + ax_offset].set_title("{} Ankle Axis ({}Hz) - {} steps/min".format(ankle_axis['label'],
                                                                                     int(ankle_fs), round(cad, 1)),
                                        fontsize=title_fontsize)
        col_ax[2 + ax_offset].set_xlabel("Seconds", fontsize=title_fontsize)

        if zoomed_xticks is not None:
            col_ax[2 + ax_offset].set_xticks(zoomed_xticks)

        if use_grid:
            col_ax[0].grid()
            col_ax[1].grid()
            col_ax[2].grid()

            if raw_wrist:
                col_ax[3].grid()

    for i, region in enumerate(regions):

        if len(regions) == 1:
            col_ax = plot2
        if len(regions) > 1:
            col_ax = plot2[:, i]

        con = ConnectionPatch(
            xyA=[(pd.to_datetime(region[0]) - window_start).total_seconds(), plot1.get_ylim()[0]],
            xyB=[col_ax[0].get_xlim()[0], col_ax[0].get_ylim()[1]],
            coordsA='data', coordsB='data',
            axesA=plot1, axesB=col_ax[0], color=colors[i], linestyle='dashed')
        col_ax[0].add_artist(con)

        con = ConnectionPatch(
            xyA=[(pd.to_datetime(region[1]) - window_start).total_seconds(), plot1.get_ylim()[0]],
            xyB=[col_ax[0].get_xlim()[1], col_ax[0].get_ylim()[1]],
            coordsA='data', coordsB='data',
            axesA=plot1, axesB=col_ax[0], color=colors[i], linestyle='dashed')
        col_ax[0].add_artist(con)

    """ ================================================================== """
    """ ======================= Final formatting ========================= """
    """ ================================================================== """

    subfigs[0].subplots_adjust(top=.925, hspace=.2, left=.15, right=.975, bottom=.1)
    subfigs[1].subplots_adjust(top=.925, hspace=.25, wspace=.1, left=.15, right=.975, bottom=.075)

    return fig, wrist, ankle


def plot_comparison_barplot_diffs(df_cp_totals, figsize=(13, 8),
                                  binary_mvpa=False, binary_activity=False, greyscale_diff=True, greyscale=False,
                                  label_fontsize=12, fontsize=10, legend_fontsize=10):

    fig, ax = plt.subplots(1, 5, figsize=figsize, sharey='row', gridspec_kw={'width_ratios': [1, 1, .67, .67, .67]})

    intensity_barplot(df=df_cp_totals, cp_author='fraysse', figsize=(8, 8),
                      df_sig=None, sig_icon="*", lw=1.5,
                      label_fontsize=label_fontsize, fontsize=fontsize, legend_fontsize=legend_fontsize,
                      ytick_subjs='cohort_id', greyscale=greyscale, incl_legend=False,
                      binary_mvpa=binary_mvpa, binary_activity=binary_activity, ax=ax[0])
    intensity_barplot(df=df_cp_totals, cp_author='powell', figsize=(8, 8),
                      df_sig=None, sig_icon="*", lw=1.5,
                      label_fontsize=label_fontsize, fontsize=fontsize, legend_fontsize=legend_fontsize,
                      ytick_subjs='cohort_id', greyscale=greyscale, incl_legend=False,
                      binary_mvpa=binary_mvpa, binary_activity=binary_activity, ax=ax[1])

    ax[1].set_ylabel("")

    c = [['lightgrey', 'dimgrey'], ['limegreen', 'green'], ['orange', 'chocolate']]
    for i, col in enumerate(['diff_sedp', 'diff_lightp', 'diff_modp']):

        if not greyscale_diff:
            ax[2+i].barh(y=np.arange(1, df_cp_totals.shape[0] + 1), width=df_cp_totals[col], alpha=.8,
                         color=[c[i][0] if j >= 0 else c[i][1] for j in df_cp_totals[col]], edgecolor='black')
        if greyscale_diff:
            ax[2+i].barh(y=np.arange(1, df_cp_totals.shape[0] + 1), width=df_cp_totals[col],
                         color=['dimgrey' if j >= 0 else 'white' for j in df_cp_totals[col]], edgecolor='black')

        ax[2+i].set_title("Δ {}\n(Fraysse - Powell)".format(col.split("_")[1][:-1].capitalize()), fontsize=label_fontsize)
        ax[2+i].axvline(x=0, color='black', linestyle='dashed', lw=2)
        ax[2+i].set_xlim(-100, 100)
        ax[2+i].set_xlabel("Difference (%)", fontsize=label_fontsize)

        ax[2+i].set_yticks(np.arange(1, df_cp_totals.shape[0] + 1))
        # ax[2+i].set_xticklabels([int(i) for i in ax[2+i].get_xticks()], fontsize=fontsize)

        ax[2+i].set_yticklabels(df_cp_totals['cohort_id'], fontsize=10)

        ax[2+i].set_ylim(.4, df_cp_totals.shape[0] + .6)

    plt.tight_layout()

    return fig


def plot_comparison_barplot(df_cp_totals, figsize=(13, 8), sharex='col', incl_legend=False,
                            binary_mvpa=False, binary_activity=False, greyscale=False,
                            label_fontsize=12, fontsize=10, legend_fontsize=10,
                            incl_mean=False, incl_median=True):

    df = df_cp_totals.copy()
    df = df[["subject_id", 'cohort_id', 'fraysse_sedp', 'fraysse_lightp', 'fraysse_modp',
             'powell_sedp', 'powell_lightp', 'powell_modp']]

    if not incl_mean and not incl_median:
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey='row')
        ax1 = ax[0]
        ax2 = ax[1]

        ax3, ax4 = ax[0], ax[1]

    if incl_mean or incl_median:
        fig, ax = plt.subplots(2, 2, figsize=figsize, sharey='row', sharex=sharex,
                               gridspec_kw={'height_ratios': [1, 3/df.shape[0]*1.2]})
        ax1 = ax[0][0]
        ax2 = ax[0][1]

        ax3, ax4 = ax[1][0], ax[1][1]

    intensity_barplot(df=df, cp_author='fraysse', figsize=(8, 8),
                      df_sig=None, sig_icon="*", lw=1.5,
                      label_fontsize=label_fontsize, fontsize=fontsize, legend_fontsize=legend_fontsize,
                      ytick_subjs='cohort_id', greyscale=greyscale, incl_legend=False,
                      binary_mvpa=binary_mvpa, binary_activity=binary_activity, ax=ax1)

    intensity_barplot(df=df, cp_author='powell', figsize=(8, 8),
                      df_sig=None, sig_icon="*", lw=1.5,
                      label_fontsize=label_fontsize, fontsize=fontsize, legend_fontsize=legend_fontsize,
                      ytick_subjs='cohort_id', greyscale=greyscale, incl_legend=incl_legend,
                      binary_mvpa=binary_mvpa, binary_activity=binary_activity, ax=ax2)

    if incl_mean or incl_median:

        c = ['grey', 'limegreen', 'orange']
        intensity = ['sedp', 'lightp', 'modp']
        ymin = [.7, .35, 0]

        for i in range(3):
            if incl_mean:
                m = df[f'fraysse_{intensity[i]}'].mean()
                s = df[f'fraysse_{intensity[i]}'].std()
                e = [m - s, m + s]
            if incl_median:
                m = df[f'fraysse_{intensity[i]}'].median()
                e = [df[f'fraysse_{intensity[i]}'].describe()['25%'], df[f'fraysse_{intensity[i]}'].describe()['75%']]

            ax3.plot([m, m], [ymin[i], ymin[i] + .3], color=c[i], lw=3)
            ax3.fill_between(x=e, y1=ymin[i], y2=ymin[i] + .3, color=c[i], alpha=.33)

            if incl_mean:
                m = df[f'powell_{intensity[i]}'].mean()
                s = df[f'powell{intensity[i]}'].std()
                e = [m - s, m + s]
            if incl_median:
                m = df[f'powell_{intensity[i]}'].median()
                e = [df[f'powell_{intensity[i]}'].describe()['25%'], df[f'powell_{intensity[i]}'].describe()['75%']]

            ax4.plot([m, m], [ymin[i], ymin[i] + .3], color=c[i], lw=3)
            ax4.fill_between(x=e, y1=ymin[i], y2=ymin[i] + .3, color=c[i], alpha=.33)

        ax3.set_yticks([.15, .5, .85])
        ax3.set_yticklabels(['MVPA', 'Light', 'Sedentary'], fontsize=fontsize)

        ax4.set_yticks([.15, .5, .85])
        ax4.set_yticklabels(['MVPA', 'Light', 'Sedentary'], fontsize=fontsize)

    if not sharex:
        ax1.set_xlabel("Intensity classification\n(% of 15-second during gait)", fontsize=label_fontsize)
        ax2.set_xlabel("Intensity classification\n(% of 15-second epochs during gait)", fontsize=label_fontsize)

        if incl_mean or incl_median:
            ax3.set_xlabel(f"{'Mean (SD)' if incl_mean else 'Median (IQR)'} intensity classification\n(% of 15-second epochs during gait)", fontsize=label_fontsize)
            ax4.set_xlabel(f"{'Mean (SD)' if incl_mean else 'Median (IQR)'} intensity classification\n(% of 15-second epochs during gait)", fontsize=label_fontsize)
            ax3.set_xlim(0, 100)
            ax3.set_xticks(np.arange(0, 101, 20))
            ax3.set_xticks(np.arange(0, 101, 10), minor=True)
            ax3.set_xticklabels([int(i) for i in ax3.get_xticks()], fontsize=fontsize)

            ax4.set_xlim(0, 100)
            ax4.set_xticks(ax2.get_xticks())
            ax4.set_xticks(np.arange(0, 101, 20))
            ax4.set_xticks(np.arange(0, 101, 10), minor=True)
            ax4.set_xticklabels([int(i) for i in ax4.get_xticks()], fontsize=fontsize)

    if sharex == 'col':
        ax3.set_xlabel("Intensity classification\n(% of 15-second epochs during gait)", fontsize=label_fontsize)
        ax4.set_xlabel("Intensity classification\n(% of 15-second epochs during gait)", fontsize=label_fontsize)

    ax2.set_ylabel("")
    plt.tight_layout()

    return fig


def plot_cp_diff_density(df_cp_totals):

    fig, ax = plt.subplots(1)
    df_cp_totals['diff_sedp'].plot.density(color='grey', label='Sed.', ax=ax)
    df_cp_totals['diff_lightp'].plot.density(color='limegreen', label='Light', ax=ax)
    df_cp_totals['diff_modp'].plot.density(color='orange', label='Mod.', ax=ax)
    ax.axvline(x=0, color='black', linestyle='dashed')
    ax.legend()
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, )
    ax.set_xlabel("Difference (Fraysse - Powell)")
    plt.tight_layout()

    return fig


def cp_diff_hist(df_cp_totals, incl_density=False):

    fig, ax = plt.subplots(3, 1, sharex='col', sharey='col')
    c = ['grey', 'limegreen', 'orange']
    labels = ['Sedentary', 'Light', "Moderate"]
    for i, col in enumerate(['diff_sedp', 'diff_lightp', 'diff_modp']):
        ax[i].hist(df_cp_totals[col], color=c[i], edgecolor='black', bins=np.arange(-100, 101, 10), alpha=.75, label=labels[i])
        if incl_density:
            sec_ax = ax[i].twinx()
            df_cp_totals[col].plot.density(color=c[i], ax=sec_ax)
            sec_ax.set_ylim(0, )
            sec_ax.set_yticks([])
            sec_ax.set_yticklabels([])
            sec_ax.set_ylabel("")
        ax[i].set_ylabel("n")
        ax[i].axvline(x=0, color='black', linestyle='dashed')
        ax[i].set_xlim(-100, 100)
        ax[i].legend()
    ax[2].set_xlabel("Difference (Fraysse - Powell)")
    plt.tight_layout()

    return fig


def cp_diff_scatter(df_cp_totals):
    fig, ax = plt.subplots(1)
    c = ['grey', 'limegreen', 'orange']
    for i, intensity in enumerate(['diff_sedp', 'diff_lightp', 'diff_modp']):
        ax.scatter(df_cp_totals['age'], df_cp_totals[intensity], color=c[i], label=f"{intensity.split('_')[1][:3].capitalize()}.")
        eq = np.polyfit(df_cp_totals['age'], df_cp_totals[intensity], deg=1)
        r = scipy.stats.pearsonr(df_cp_totals['age'], df_cp_totals[intensity])

        y = [i * eq[0] + eq[1] for i in np.arange(65, 90, 1)]
        ax.plot(np.arange(65, 90, 1), y, color=c[i], label=f"r={r[0]:.3f}")
    ax.legend()
    ax.set_ylabel("Difference (Fraysse - Powell)")
    ax.set_xlabel("Age")
    ax.axhline(y=0, color='black', linestyle='dashed')
    ax.set_ylim(-100, 100)
    ax.set_xlim(64, 95)
    ax.set_title("Difference in % of epochs by intensity")

    return fig


def cp_comp_barplot_all(df_cp_totals):
    df_cp_totals = df_cp_totals.sort_values("fraysse_sedp", ascending=False)
    fig, ax = plt.subplots(3, 1, sharex='col', figsize=(12, 8))

    ax[0].bar(np.arange(0, df_cp_totals.shape[0]) - 1 / 6, df_cp_totals['fraysse_sedp'], color='lightgrey',
              edgecolor='black', width=1 / 3, label='Fraysse')
    ax[0].bar(np.arange(0, df_cp_totals.shape[0]) + 1 / 6, df_cp_totals['powell_sedp'], color='lightgrey',
              edgecolor='black', width=1 / 3, hatch='//', label='Powell')
    ax[0].legend()
    ax[0].set_ylabel("% sed. epochs")

    ax[1].bar(np.arange(0, df_cp_totals.shape[0]) - 1 / 6, df_cp_totals['fraysse_lightp'], color='limegreen',
              edgecolor='black', width=1 / 3, alpha=.75)
    ax[1].bar(np.arange(0, df_cp_totals.shape[0]) + 1 / 6, df_cp_totals['powell_lightp'], color='limegreen',
              edgecolor='black', width=1 / 3, hatch='//', alpha=.75)
    ax[1].set_ylabel("% light epochs")

    ax[2].bar(np.arange(0, df_cp_totals.shape[0]) - 1 / 6, df_cp_totals['fraysse_modp'], color='orange',
              edgecolor='black', width=1 / 3, alpha=.75)
    ax[2].bar(np.arange(0, df_cp_totals.shape[0]) + 1 / 6, df_cp_totals['powell_modp'], color='orange',
              edgecolor='black', width=1 / 3, hatch='//', alpha=.75)
    ax[2].set_ylabel("% mod. epochs")

    ax[2].set_xticks(np.arange(0, df_cp_totals.shape[0], 1))
    ax[2].set_xticklabels(df_cp_totals['cohort_id'], rotation=90)
    ax[-1].set_xlim(-.5, df_cp_totals.shape[0] + 1.5)
    plt.tight_layout()

    return fig


def cp_comp_mean_barplot(df_cp_desc, err_colname='std'):

    df_cp_desc['sem'] = df_cp_desc['std'] / np.sqrt(df_cp_desc['count'])

    fig, ax = plt.subplots(1, figsize=(6, 6))

    ax.bar([-1], df_cp_desc.loc['fraysse_sedp']['mean'], color='white', edgecolor='black', width=.3, label='Fraysse')
    ax.bar([-1], df_cp_desc.loc['powell_sedp']['mean'], color='white', edgecolor='black', hatch='/////', width=.3, label='Powell')

    ax.bar([-1/6], df_cp_desc.loc['fraysse_sedp']['mean'], color='grey', edgecolor='black', width=.3,
           yerr=df_cp_desc.loc['fraysse_sedp'][err_colname], capsize=4)
    ax.bar([1/6], df_cp_desc.loc['powell_sedp']['mean'], color='grey', edgecolor='black', hatch='///', width=.3,
           yerr=df_cp_desc.loc['powell_sedp'][err_colname], capsize=4)

    ax.bar([1 -1/6], df_cp_desc.loc['fraysse_lightp']['mean'], color='limegreen', edgecolor='black', width=.3,
           yerr=df_cp_desc.loc['fraysse_lightp'][err_colname], capsize=4)
    ax.bar([1 + 1/6], df_cp_desc.loc['powell_lightp']['mean'], color='limegreen', edgecolor='black', hatch='///', width=.3,
           yerr=df_cp_desc.loc['powell_lightp'][err_colname], capsize=4)

    ax.bar([2 -1/6], df_cp_desc.loc['fraysse_modp']['mean'], color='orange', edgecolor='black', width=.3,
           yerr=df_cp_desc.loc['fraysse_modp'][err_colname], capsize=4)
    ax.bar([2 + 1/6], df_cp_desc.loc['powell_modp']['mean'], color='orange', edgecolor='black', hatch='///', width=.3,
           yerr=df_cp_desc.loc['powell_modp'][err_colname], capsize=4)

    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['sedentary', 'light', 'moderate'])
    ax.set_ylabel("% of 15-second epochs\nduring walking")
    ax.legend()

    return fig


def cp_comp_meandiff_barplot(df_cp_desc, err_colname='std'):

    df_cp_desc['sem'] = df_cp_desc['std'] / np.sqrt(df_cp_desc['count'])

    fig, ax = plt.subplots(1, figsize=(6, 6))

    ax.bar(['sedentary'], df_cp_desc.loc['diff_sedp']['mean'], color='grey', edgecolor='black', width=.75,
           yerr=df_cp_desc.loc['diff_sedp'][err_colname], capsize=4)

    ax.bar(['light'], df_cp_desc.loc['diff_lightp']['mean'], color='limegreen', edgecolor='black', width=.75,
           yerr=df_cp_desc.loc['diff_lightp'][err_colname], capsize=4)

    ax.bar(['mod'], df_cp_desc.loc['diff_modp']['mean'], color='orange', edgecolor='black', width=.75,
           yerr=df_cp_desc.loc['diff_modp'][err_colname], capsize=4)

    ax.axhline(y=0, color='black', linestyle='dashed')
    ax.set_ylim(-100, 100)
    ax.set_ylabel("Δ % (Fraysse - Powell)")

    return fig


def bland_altman(x, y, xlabel, ylabel):

    fig, ax = plt.subplots(1, figsize=(10, 8))

    bias = np.mean(y)
    ci = 1.96 * np.std(y)
    ax.grid()
    ax.scatter(x, y, color='black')
    ax.axhline(bias, color='black', linestyle='dashed', label='bias')
    ax.axhline(bias + ci, color='red', linestyle='dashed', label='UL')
    ax.axhline(bias - ci, color='red', linestyle='dashed', label='LL')
    ax.axhline(0, color='limegreen')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_scatter(df, x, y, groupby=None, incl_ci=True, incl_reg_line=True, group_level_reg=True,
                 correl_location='legend', legend_fontsize=12, label_fontsize=15, tick_fontsize=12,
                 incl_corr_val=None, ax=None, min_n_regression=4, colors=None, highlight_subj=None):

    if ax is None:
        fig, ax = plt.subplots(1)

    if groupby is not None:
        plt.suptitle(f"Data grouped by {groupby}")
        g = df.groupby(groupby)

        if group_level_reg:

            if incl_corr_val is None:
                all_label = 'All'
            if incl_corr_val == 'spearman':
                rho = scipy.stats.spearmanr(df[x], df[y])[0]
                all_label = f"All (rho = {rho:.3f})"
            if incl_corr_val == 'pearson':
                r = scipy.stats.spearmanr(df[x], df[y])[0]
                all_label = f"All (r = {r:.3f})"

            sns.regplot(data=df, x=x, y=y, ax=ax,
                        ci=95,
                        fit_reg=incl_reg_line, truncate=True,
                        scatter_kws={'color': 'white', 'marker': None, 'zorder': 1},
                        line_kws={'lw': .01, 'color': 'dodgerblue'})

            reg = np.polyfit(df[x], df[y], deg=1)
            lims = [df[x].min(), df[x].max()]
            x_vals = np.arange(lims[0], lims[1]+1)

            if correl_location == 'legend':
                ax.plot(x_vals, [reg[0] * i + reg[1] for i in x_vals], color='dodgerblue', lw=1.5, label=all_label)
            if correl_location == 'plot':
                ax.plot(x_vals, [reg[0] * i + reg[1] for i in x_vals], color='dodgerblue', lw=1.5)
                ax.text(x_vals[-1]-10, y=x_vals[-1] * reg[0] + reg[1] + 5, s=all_label,
                        color='dodgerblue', fontsize=label_fontsize)

        for group_i, group in enumerate(g.groups):

            curr_group = g.get_group(group)

            if incl_ci or incl_reg_line:
                if curr_group.shape[0] >= min_n_regression:
                    sns.regplot(data=curr_group, x=x, y=y, ax=ax,
                                label=group,
                                ci=None if not incl_ci else 95,
                                fit_reg=incl_reg_line and not group_level_reg, truncate=True,
                                scatter_kws={'color': colors[group_i] if colors is not None else None,
                                             'edgecolor': 'black'},
                                line_kws={'color': colors[group_i]} if colors is not None else None)
                    reg = np.polyfit(curr_group[x], curr_group[y], 1)
                    r = scipy.stats.pearsonr(curr_group[x], curr_group[y])[0]
                    rho = scipy.stats.spearmanr(curr_group[x], curr_group[y])[0]
                    val_range = np.arange(ax.get_xlim()[0], ax.get_xlim()[1])

    if groupby is None:

        sns.regplot(data=df, x=x, y=y, ci=None if not incl_ci else 95, fit_reg=incl_reg_line, ax=ax, zorder=1)

    if highlight_subj is not None:
        df_subj = df.loc[df['subject_id'] == highlight_subj]
        ax.scatter(df_subj[x], df_subj[y], edgecolor='red', color='red',
                   zorder=2, label=df_subj['cohort_id'].iloc[0])

    ax.legend(fontsize=legend_fontsize)

    ax.set_xlabel(x, fontsize=label_fontsize)
    ax.set_ylabel(y, fontsize=label_fontsize)

# bland_altman(x=(df_cp_totals['fraysse_modp'] + df_cp_totals['powell_modp'])/2, y=df_cp_totals['fraysse_modp'] - df_cp_totals['powell_modp'], xlabel='mean % moderate (Fraysse + Powell)', ylabel='Fraysse - Powell (% mod)')
# bland_altman(x=df_cp_totals['n_epochs'], y=df_cp_totals['fraysse_mod'] - df_cp_totals['powell_mod'], xlabel='n epochs', ylabel='Fraysse - Powell (mod epochs)')
# bland_altman(x=df_cp_totals['n_epochs'], y=df_cp_totals['diff_modp'], xlabel='n_epochs', ylabel='Fraysse - Powell (% mod)')
