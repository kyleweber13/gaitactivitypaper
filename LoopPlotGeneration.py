import Plotting
import RunLoop
import DataImport
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from nimbalwear.activity import activity_wrist_avm as nw_act
from matplotlib.patches import ConnectionPatch
plt.rcParams.update({'font.serif': 'Times New Roman', 'font.family': 'serif'})


def import_processed_data(df_demos, cutpoints, root_dir, full_ids):

    print(f"\nImporting data from {full_ids}...")

    failed_files, data = RunLoop.run_loop(df_demos=df_demos, full_ids=sorted(list(full_ids)),
                                          cutpoints=cutpoints,
                                          save_files=False, root_dir=root_dir, correct_ond09_cadence=True,
                                          min_cadence=80, min_bout_dur=60, mrp=5,
                                          min_step_time=60/200, remove_edge_low_cadence=True)

    return data


def import_raw_data(df_demos, full_ids):

    data_dict = {}
    for subj in full_ids:
        ankle = DataImport.import_edf(df_demos.loc[df_demos['full_id'] == subj]['ankle_edf'].iloc[0])
        wrist = DataImport.import_edf(df_demos.loc[df_demos['full_id'] == subj]['wrist_edf'].iloc[0])
        df_steps = DataImport.import_steps_file(f"O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Calibrated/Data/Steps/{subj}_01_GAIT_STEPS.csv")

        subj_dict = {'ankle': ankle, 'wrist': wrist, 'steps': df_steps}

        data_dict[subj] = subj_dict

    return data_dict


plot_dict = {"OND09_0037": {"ankle_axis": 'Accelerometer y', 'ankle_mult': 1},
             'OND06_9525': {"ankle_axis": 'Accelerometer x', 'ankle_mult': 1},
             'OND06_3413': {"ankle_axis": 'Accelerometer x', 'ankle_mult': 1}}

data = import_processed_data(df_demos=df_demos, cutpoints=(62.5, 92.5), full_ids=list(plot_dict.keys()),
                             root_dir="O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Calibrated/Data/")

try:
    f = open(f"C:/Users/ksweber/Desktop/wristgait_paper_samples_rawdata.pickle", 'rb')
    raw_data = pickle.load(f)
    f.close()
except FileNotFoundError:
    raw_data = import_raw_data(df_demos=df_demos, full_ids=list(plot_dict.keys()))


def legacy():
    section_dict = {"OND09_0037": {0: {'window': ['2021-11-11 15:40:45', '2021-11-11 15:41:15'],
                                       'description': 'fastMVPA',
                                       'regions': [['2021-11-11 15:40:55', '2021-11-11 15:41:05']],
                                       'colors': ['orange'],
                                       'fig_width': 6, 'fig_height': 10,
                                       'xticks': [0, 2, 4, 6, 8, 10],
                                       'include_yaxis': True},
                                   1: {"window": ['2021-11-10 19:37:30', '2021-11-10 19:38:00'],
                                       'description': 'fastLight',
                                       'regions': [['2021-11-10 19:37:40', '2021-11-10 19:37:50']],
                                       'colors': ['green'],
                                       'fig_width': 6, 'fig_height': 10,
                                       'xticks': [0, 2, 4, 6, 8, 10],
                                       'include_yaxis': True},
                                   2: {"window": ['2021-11-13 11:42:49', '2021-11-13 11:43:20'],
                                       'description': 'fastSed',
                                       'regions': [['2021-11-13 11:42:59', '2021-11-13 11:43:09']],
                                       'colors': ['grey'],
                                       'fig_width': 6, 'fig_height': 10,
                                       'xticks': [0, 2, 4, 6, 8, 10],
                                       'include_yaxis': True}},

                    "OND06_9525": {0: {"window": ['2020-03-06 8:03:00', '2020-03-06 8:04:30'],
                                       "description": 'PDTremor',
                                       'regions': [['2020-03-06 8:03:05', '2020-03-06 8:03:20'],
                                                   ['2020-03-06 8:03:42.5', '2020-03-06 8:03:57.5']],
                                       'colors': ['green', 'orange'],
                                       'fig_width': 10, 'fig_height': 10,
                                       'xticks': [0, 5, 10, 15],
                                       'include_yaxis': True}},

                    'OND06_3413': {0: {'window': ['2020-01-29 12:36:50', '2020-01-29 12:37:20'],
                                       'description': 'walker',
                                       'regions': [['2020-01-29 12:36:53.4', '2020-01-29 12:37:08.4']],
                                       'colors': ['grey'],
                                       'fig_width': 5, 'fig_height': 10,
                                       'xticks': [0, 5, 10, 15],
                                       'include_yaxis': True}}}

    for subj in section_dict.keys():

        print("===========")
        subj_dict = section_dict[subj]
        for window in subj_dict.keys():
            window_dict = subj_dict[window]
            print(subj, window, window_dict['regions'], window_dict['description'])

            wrist = raw_data[subj]['wrist']
            ankle = raw_data[subj]['ankle']

            plt.close('all')
            plot1, wrist, ankle = Plotting.data_sections2(full_id=subj, df_demos=df_demos,
                                                 # data
                                                 wrist=wrist, ankle=ankle,
                                                 wrist_up=wrist.signals[wrist.get_signal_index("Accelerometer y")],
                                                 wrist_ant=wrist.signals[wrist.get_signal_index("Accelerometer x")],
                                                 wrist_med=wrist.signals[wrist.get_signal_index("Accelerometer z")],
                                                 ankle_axis={'data': ankle.signals[ankle.get_signal_index(plot_dict[subj]['ankle_axis'])] * plot_dict[subj]['ankle_mult'],
                                                             'label': 'AP'},
                                                 steps=raw_data[subj]['steps'],

                                                 # data parameters
                                                 raw_wrist=True,
                                                 rem_ankle_base=True,
                                                 norm_ankle=True,
                                                 epoch_len=5,

                                                 # data sections
                                                 window_start=pd.to_datetime(window_dict['window'][0]),
                                                 window_end=pd.to_datetime(window_dict['window'][1]),
                                                 regions=window_dict['regions'],
                                                 pad_window=0,

                                                 # plotting parameters
                                                 include_yaxis=window_dict['include_yaxis'],
                                                 avm_bar=False,
                                                 colors=window_dict['colors'],
                                                 alpha=(.2, .2),
                                                 show_legend=False, show_legend2=False,
                                                 # fig_width=6 * len(window_dict['regions']) + 2,
                                                 fig_width=window_dict['fig_width'],
                                                 fig_height=window_dict['fig_height'],

                                                 # region_name=window_dict['description'],
                                                 force_wrist_ylim=(-2, 1.25), wrist_yticks=[-2, -1, 0, 1],
                                                 zoomed_xticks=window_dict['xticks'],
                                                 force_ankle_ylim=(None, None),
                                                 force_raw_vm_ylim=(-10, 1000),
                                                 force_avm_ylim=(0, 151),
                                                 linewidth=.75,
                                                 tick_fontsize=9, title_fontsize=10)

            plt.savefig(f"C:/Users/ksweber/Desktop/Paper_Images/New/{subj}_{window}_{window_dict['description']}.tiff", dpi=125)

    plt.close('all')


subj_deets_dict = {"OND09_0037": {"ankle_axis": 'Accelerometer y', 'ankle_mult': 1, 'ankle_desc': 'AP',
                                  'wrist_up': 'Accelerometer y', 'wrist_ant': "Accelerometer x", 'wrist_med': "Accelerometer z",
                                  'wrist_up_mult': 1, 'wrist_ant_mult': 1, 'wrist_med_mult': 1},
                   'OND06_9525': {"ankle_axis": 'Accelerometer x', 'ankle_mult': 1, 'ankle_desc': 'AP',
                                  'wrist_up': 'Accelerometer y', 'wrist_ant': "Accelerometer x", 'wrist_med': "Accelerometer z",
                                  'wrist_up_mult': 1, 'wrist_ant_mult': 1, 'wrist_med_mult': 1},
                   'OND06_3413': {"ankle_axis": 'Accelerometer x', 'ankle_mult': 1, 'ankle_desc': 'AP',
                                  'wrist_up': 'Accelerometer y', 'wrist_ant': "Accelerometer x", 'wrist_med': "Accelerometer z",
                                  'wrist_up_mult': 1, 'wrist_ant_mult': 1, 'wrist_med_mult': 1}}

section_dict = {'figure2': {0: {'full_id': 'OND09_0037',
                                'window': ['2021-11-11 15:40:45', '2021-11-11 15:41:15'],
                                'description': 'fastMVPA',
                                'figure_label': None,
                                'regions': [['2021-11-11 15:40:55', '2021-11-11 15:41:05']],
                                'colors': ['orange']},
                            1: {'full_id': 'OND09_0037',
                                "window": ['2021-11-10 19:37:30', '2021-11-10 19:38:00'],
                                'description': 'fastLight', 'figure_label': None,
                                'regions': [['2021-11-10 19:37:40', '2021-11-10 19:37:50']],
                                'colors': ['green']},
                            2: {'full_id': 'OND09_0037',
                                "window": ['2021-11-13 11:42:49', '2021-11-13 11:43:20'],
                                'description': 'fastSed', 'figure_label': None,
                                'regions': [['2021-11-13 11:42:59', '2021-11-13 11:43:09']],
                                'colors': ['grey']}},
                'figure3': {0: {'full_id': 'OND06_3413',
                                'window': ['2020-01-29 12:36:50', '2020-01-29 12:37:20'],
                                'description': 'walker',
                                'figure_label': "A)", 'text_locx': .035, 'text_locy': .925,
                                'regions': [['2020-01-29 12:36:53.4', '2020-01-29 12:37:08.4']],
                                'colors': ['grey']},
                            1: {'full_id': 'OND06_9525',
                                "window": ['2020-03-06 8:03:00', '2020-03-06 8:04:30'],
                                "description": 'PDTremor',
                                'figure_label': "B)", 'text_locx': .075, 'text_locy': .925,
                                'regions': [['2020-03-06 8:03:05', '2020-03-06 8:03:20'],
                                            ['2020-03-06 8:03:42.5', '2020-03-06 8:03:57.5']],
                                'colors': ['green', 'orange']}}}


def plot_sections(raw_data_dict, device_dict, section_dict, plot_key,
                  rem_ankle_base=True, norm_ankle=True, epoch_len=5, pad_window=0,
                  cutpoints=(62.5, 92.5), figsize=(8.5, 11),
                  tick_fontsize=8, title_fontsize=10, use_grid=True, alpha=.2, linewidth=.75,
                  force_wrist_ylim=None, force_avm_ylim=None, force_ankle_ylim=None, force_raw_vm_ylim=None):

    zoom_ylabs = ['G', 'mG', 'mG', "G" if not norm_ankle else "G (normalized)"]

    fig = plt.figure(figsize=figsize)
    plt.rc('ytick', labelsize=tick_fontsize)
    plt.rc('xtick', labelsize=tick_fontsize)

    width_ratios = []
    for col in section_dict[plot_key].keys():
        subplot_dict = section_dict[plot_key][col]
        width_ratios.append(len(subplot_dict['regions']))

    print(f"Generating {2} x {len(section_dict[plot_key])} plot...")
    subfigs = fig.subfigures(2, len(section_dict[plot_key]), height_ratios=[2, 4], width_ratios=width_ratios)

    for panel in section_dict[plot_key].keys():
        subplot_dict = section_dict[plot_key][panel]
        regions = subplot_dict['regions']
        window = subplot_dict['window']
        n_regions = len(regions)
        colors = subplot_dict['colors']

        wrist = raw_data_dict[subplot_dict['full_id']]['wrist']
        ankle = raw_data_dict[subplot_dict['full_id']]['ankle']
        steps = raw_data_dict[subplot_dict['full_id']]['steps']
        wrist_fs = wrist.signal_headers[wrist.get_signal_index("Accelerometer x")]['sample_rate']

        ankle_fs = ankle.signal_headers[ankle.get_signal_index("Accelerometer x")]['sample_rate']
        deets = device_dict[subplot_dict['full_id']]

        plotx = subfigs[0][panel].subplots(2, subplot_kw={"zorder": 1}, sharex='col')
        subfigs[0][panel].subplots_adjust(hspace=.3, wspace=.125, right=.975)

        if panel == 0:
            subfigs[0][0].subplots_adjust(left=.15)

        plot0 = plotx[0]
        plot1 = plotx[1]
        plot2 = subfigs[1][panel].subplots(4, n_regions,
                                           sharey='row', sharex='col', squeeze=True, subplot_kw={"zorder": 0})

        plot0.set_title(f"Raw Wrist Accelerometer ({int(wrist_fs)} Hz)", fontsize=title_fontsize)
        plot1.set_title("{} Ankle Accelerometer Axis ({} Hz)".format(deets['ankle_desc'], int(ankle_fs)),
                        fontsize=title_fontsize)

        if panel == 0:
            plot0.set_ylabel("G", fontsize=title_fontsize)
            plot1.set_ylabel("G" if not norm_ankle else 'G (normalized)', fontsize=title_fontsize)

        if subplot_dict['figure_label'] is not None:
            subfigs[0][panel].text(subplot_dict['text_locx'], subplot_dict['text_locy'],
                                   subplot_dict['figure_label'], size=title_fontsize + 4)

        wrist_fs = wrist.signal_headers[wrist.get_signal_index('Accelerometer x')]['sample_rate']
        ankle_fs = ankle.signal_headers[ankle.get_signal_index('Accelerometer x')]['sample_rate']

        wrist_start = wrist.header['startdate']
        ankle_start = ankle.header['startdate']
        window_start = pd.to_datetime(window[0])
        window_end = pd.to_datetime(window[1])

        wrist_idx = [int((window_start - wrist_start).total_seconds() * wrist_fs),
                     int((window_end - wrist_start).total_seconds() * wrist_fs)]
        ankle_idx = [int((window_start - ankle_start).total_seconds() * ankle_fs),
                     int((window_end - ankle_start).total_seconds() * ankle_fs)]

        t = np.arange(wrist_idx[-1] - wrist_idx[0]) / wrist_fs
        t = [i - pad_window for i in t]

        plot0.plot(t, wrist.signals[wrist.get_signal_index(deets['wrist_up'])][wrist_idx[0]:wrist_idx[1]] *
                   deets['wrist_up_mult'],
                   color='black', label='Up', lw=linewidth)
        plot0.plot(t, wrist.signals[wrist.get_signal_index(deets['wrist_ant'])][wrist_idx[0]:wrist_idx[1]] *
                   deets['wrist_ant_mult'],
                   color='red', label='Uln/Rad', lw=linewidth)
        plot0.plot(t, wrist.signals[wrist.get_signal_index(deets['wrist_med'])][wrist_idx[0]:wrist_idx[1]] *
                   deets['wrist_med_mult'],
                   color='dodgerblue', label='Med.', lw=linewidth)

        if force_wrist_ylim is not None:
            plot0.set_ylim(force_wrist_ylim)

        if use_grid:
            plot0.grid()

        # raw ankle data
        use_ankle_data = ankle.signals[ankle.get_signal_index(deets['ankle_axis'])][ankle_idx[0]:ankle_idx[1]]

        if rem_ankle_base:
            mean_a = np.mean(use_ankle_data)
            use_ankle_data = [i - mean_a for i in use_ankle_data]

        if norm_ankle:
            ankle_min = min(use_ankle_data)
            ankle_max = max(use_ankle_data)
            ankle_range = ankle_max - ankle_min
            use_ankle_data = [(i - ankle_min) / ankle_range * 2 - 1 for i in use_ankle_data]

        plot1.plot(t, use_ankle_data, color='black', lw=linewidth)

        plot1.set_xlim(-pad_window, max(t) + pad_window + .05)

        if force_ankle_ylim is not None:
            plot1.set_ylim(force_ankle_ylim)

        if use_grid:
            plot1.grid()

        for sub_col, region in enumerate(regions):

            if 'timestamp' in steps.columns:
                window_steps = steps.loc[(steps['timestamp'] >= pd.to_datetime(region[0])) &
                                         (steps['timestamp'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['timestamp'] - window_steps.iloc[0]['timestamp']).total_seconds()
            if 'step_time' in steps.columns:
                window_steps = steps.loc[(steps['step_time'] >= pd.to_datetime(region[0])) &
                                         (steps['step_time'] < pd.to_datetime(region[1]))]
                step_dur = (window_steps.iloc[-1]['step_time'] - window_steps.iloc[0]['step_time']).total_seconds()

            n_steps = window_steps.shape[0] - 1
            cad = 60 * 2 * (n_steps / step_dur) if "OND09" in subplot_dict['full_id'] else 60 * (n_steps / step_dur)

            wrist_reg_idx = [int((pd.to_datetime(region[0]) - wrist.header['startdate']).total_seconds() * wrist_fs),
                             int((pd.to_datetime(region[1]) - wrist.header['startdate']).total_seconds() * wrist_fs)]

            ankle_reg_idx = [int((pd.to_datetime(region[0]) - ankle.header['startdate']).total_seconds() * ankle_fs),
                             int((pd.to_datetime(region[1]) - ankle.header['startdate']).total_seconds() * ankle_fs)]

            # magnitude and AVM data
            wrist_vm = nw_act(x=wrist.signals[wrist.get_signal_index('Accelerometer x')][wrist_reg_idx[0]:wrist_reg_idx[1]],
                              y=wrist.signals[wrist.get_signal_index('Accelerometer y')][wrist_reg_idx[0]:wrist_reg_idx[1]],
                              z=wrist.signals[wrist.get_signal_index('Accelerometer z')][wrist_reg_idx[0]:wrist_reg_idx[1]],
                              epoch_length=epoch_len, sample_rate=wrist_fs,
                              start_datetime=pd.to_datetime(region[0]), quiet=True)[3]

            wrist_vm *= 1000
            avm = [np.mean(wrist_vm[i:i + int(wrist_fs * epoch_len)]) for i in
                   np.arange(0, len(wrist_vm), int(wrist_fs * epoch_len))]

            t_reg_wrist = np.arange(wrist_reg_idx[-1] - wrist_reg_idx[0]) / wrist_fs
            t_reg_ankle = np.arange(ankle_reg_idx[-1] - ankle_reg_idx[0]) / ankle_fs

            plot0.axvspan(xmin=(pd.to_datetime(region[0]) - window_start).total_seconds() - pad_window,
                          xmax=(pd.to_datetime(region[1]) - window_start).total_seconds() - pad_window,
                          ymin=0, ymax=1, color=colors[sub_col], alpha=alpha)
            plot1.axvspan(xmin=(pd.to_datetime(region[0]) - window_start).total_seconds() - pad_window,
                          xmax=(pd.to_datetime(region[1]) - window_start).total_seconds() - pad_window,
                          ymin=0, ymax=1, color=colors[sub_col], alpha=alpha)

            col_ax = plot2[:, sub_col] if n_regions > 1 else plot2

            col_ax[0].plot(t_reg_wrist,
                           wrist.signals[wrist.get_signal_index(deets['wrist_up'])][wrist_reg_idx[0]:wrist_reg_idx[1]] *
                           deets['wrist_up_mult'], color='black', label='Up', lw=linewidth)

            col_ax[0].plot(t_reg_wrist,
                           wrist.signals[wrist.get_signal_index(deets['wrist_ant'])][wrist_reg_idx[0]:wrist_reg_idx[1]] *
                           deets['wrist_ant_mult'], color='red', label='Uln/Rad', lw=linewidth)

            col_ax[0].plot(t_reg_wrist,
                           wrist.signals[wrist.get_signal_index(deets['wrist_med'])][wrist_reg_idx[0]:wrist_reg_idx[1]] *
                           deets['wrist_med_mult'], color='dodgerblue', label='Med.', lw=linewidth)

            if force_wrist_ylim is not None:
                col_ax[0].set_ylim(force_wrist_ylim)

            con = ConnectionPatch(
                xyA=[(pd.to_datetime(region[0]) - window_start).total_seconds(), plot1.get_ylim()[0]],
                xyB=[0, col_ax[0].get_ylim()[1]],
                coordsA='data', coordsB='data',
                axesA=plot1, axesB=col_ax[0], color=colors[sub_col], linestyle='dashed')
            col_ax[0].add_artist(con)

            con = ConnectionPatch(
                xyA=[(pd.to_datetime(region[1]) - window_start).total_seconds(), plot1.get_ylim()[0]],
                xyB=[(pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds(), col_ax[0].get_ylim()[1]],
                coordsA='data', coordsB='data',
                axesA=plot1, axesB=col_ax[0], color=colors[sub_col], linestyle='dashed')
            col_ax[0].add_artist(con)

            # wrist VM
            col_ax[1].plot(t_reg_wrist[:min([len(t_reg_wrist), len(wrist_vm)])], wrist_vm[:min([len(t_reg_wrist), len(wrist_vm)])],
                           color='black', lw=linewidth)
            col_ax[1].set_title("Wrist Vector Magnitude ({} Hz)".format(int(wrist_fs)), fontsize=title_fontsize)

            if force_raw_vm_ylim is not None:
                col_ax[1].set_ylim(force_raw_vm_ylim)

            # wrist avm
            col_ax[2].axhline(cutpoints[0], color='green', linestyle='dashed')
            col_ax[2].axhline(cutpoints[1], color='red', linestyle='dashed')

            for epoch, a in enumerate(avm):
                col_ax[2].plot([epoch_len * epoch, epoch_len * (epoch + 1) - .1], [a, a], color='black', lw=2)

            if force_avm_ylim is not None:
                col_ax[2].set_ylim(force_avm_ylim)

            col_ax[3].plot(t_reg_ankle,
                           # ankle.signals[ankle.get_signal_index(deets['ankle_axis'])][ankle_reg_idx[0]:ankle_reg_idx[1]],
                           use_ankle_data[ankle_reg_idx[0]-ankle_idx[0]:ankle_reg_idx[1]-ankle_idx[0]],
                           color='black', lw=linewidth)

            if norm_ankle:
                col_ax[3].set_yticks([-1, 0, 1])

            col_ax[-1].set_xlim(-.1, (pd.to_datetime(region[1]) - pd.to_datetime(region[0])).total_seconds() + .1)

            zoom_titles = [f'Raw Wrist Acceleration ({int(wrist_fs)} Hz)', f'Wrist Vector Magnitude ({int(wrist_fs)} Hz)',
                           f"Wrist Average Vector Magnitude ({epoch_len}-second)",
                           f"{deets['ankle_desc']} Ankle Axis ({int(ankle_fs)} Hz) - {cad:.1f} steps/minute"]

            for row in range(4):
                if panel == 0:
                    col_ax[row].set_ylabel(zoom_ylabs[row], fontsize=title_fontsize)

                col_ax[row].set_title(zoom_titles[row], fontsize=title_fontsize)

                col_ax[row].patch.set_facecolor(color=colors[sub_col])
                col_ax[row].patch.set_alpha(alpha)

                if use_grid:
                    col_ax[row].grid()

            col_ax[-1].set_xlabel("Seconds", fontsize=title_fontsize)

        if panel > 0:
            plot0.set_yticklabels([])
            plot1.set_yticklabels([])

        for row in range(4):
            if n_regions > 1:
                if panel > 0:
                    plot2[row][0].set_yticklabels([])
            if n_regions == 1:
                if panel > 0:
                    plot2[row].set_yticklabels([])

        plt.subplots_adjust(wspace=.1)

    subfigs[1, 0].align_ylabels()

    plt.show()


for figure_key in section_dict.keys():
    plot_sections(raw_data_dict=raw_data, device_dict=subj_deets_dict, section_dict=section_dict, plot_key=figure_key,
                  figsize=(12, 9), alpha=.2, use_grid=True, cutpoints=(62.5, 92.5),
                  tick_fontsize=9, title_fontsize=12,
                  force_wrist_ylim=(-2, 1.25),
                  force_avm_ylim=(0, 151), force_ankle_ylim=None,
                  force_raw_vm_ylim=(-10, 1000),
                  norm_ankle=True)
    # plt.savefig(f"C:/Users/ksweber/Desktop/{figure_key}_small.png", dpi=100)
    plt.savefig(f"O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Calibrated/JMIR mHealth and uHealth submission/Copyediting/Revised Figures/{figure_key}.png", dpi=100)
    plt.close("all")

