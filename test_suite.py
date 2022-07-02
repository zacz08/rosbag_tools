import multiprocessing as mp

import pandas as pd
import pexpect
import time
import os
import re
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats

# dirty way to import eval file from separate project
import sys

sys.path.insert(1, '/home/dom-ubuntu/opt/ORB_SLAM3/eval')
import eval


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def process_tests(testdir, store_name):
    curr_dir = os.getcwd()
    os.chdir(testdir)

    test_result_list = [
        ["type", "intensity", "run", "rmse", "mean", "std_dev", "min", "max", "tracked_points", "total_points",
         "track_start_time", "track_end_time", "sve_rms", "sve_mean", "sve_std_dev"]]

    typelist = [i for i in os.listdir(testdir) if os.path.isdir(i)]
    typelist.sort()
    for type in typelist:
        print(f"Type: {type}")
        dir_list = os.listdir(f"{testdir}/{type}")
        dir_list.sort()
        # trajlist = []
        run_dict = {}
        for filename in dir_list:
            if re.findall(r"sve.*\.txt", filename):
                [run_name, run_number] = re.split("_sve_", filename[:-4])
                run_number = int(run_number)
                [_, run_intensity] = re.split(f"{type}", run_name)
                try:
                    run_intensity = float(run_intensity)
                    if run_intensity > 1:
                        run_intensity = int(run_intensity)
                except ValueError:
                    try:
                        run_intensity = int(re.split(r'\D', run_intensity)[0])
                    except:
                        run_intensity = 0

                print(f"Run: {run_name:20s}\t-\t#{run_number:02d}")

                # TODO: remake eval.py to callable functions
                # THEN: iterate post analysis of each trajectory (eval.py)
                eval_args = {
                    'groundtruth_traj_file': '~/Documents/fyp/datasets/euroc/MH04/mav0/state_groundtruth_estimate0/data.csv',
                    'estimated_traj_file': f"{testdir}/{type}/{run_name}_traj_{run_number}.txt",
                    'verbose': f"{testdir}/{type}/{run_name}_metrics_{run_number}.json",
                    # 'plot': f"{testdir}/{type}/{run_name}_traj_plot_{run_number}",
                    'sve': [f"{testdir}/{type}/{run_name}_sve_{run_number}.txt",
                            f"{testdir}/{type}/{run_name}_sve_plot_{run_number}", False]
                }
                with eval.Evaluator(eval_args) as ev:
                    ev.main()
                    res = ev.get_results()

                    if res["traj"] is not None:
                        # ["type", "intensity", "run", "rmse", "mean", "std_dev", "min", "max", "tracked_points", "total_points", "track_start_time", "track_end_time", "sve_rms", "sve_mean", "sve_std_dev"]
                        result_row = [type,
                                      run_intensity,
                                      run_number,
                                      res['traj']['error']['rmse'],
                                      res['traj']['error']['mean'],
                                      res['traj']['error']['std'],
                                      res['traj']['error']['min'],
                                      res['traj']['error']['max'],
                                      res['traj']['len'],
                                      res['sve']['len'],
                                      res['traj']['time']['start'],
                                      res['traj']['time']['end'],
                                      res['sve']['rms'],
                                      res['sve']['mean'],
                                      res['sve']['std']]
                    else:
                        # ["type", "intensity", "run", "rmse", "mean", "std_dev", "min", "max", "tracked_points", "total_points", "track_start_time", "track_end_time", "sve_rms", "sve_mean", "sve_std_dev"]
                        result_row = [type,
                                      run_intensity,
                                      run_number,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      res['sve']['len'],
                                      np.nan,
                                      np.nan,
                                      res['sve']['rms'],
                                      res['sve']['mean'],
                                      res['sve']['std']]

                    test_result_list.append(result_row)

    column_titles = test_result_list.pop(0)
    result_df = pd.DataFrame(test_result_list, columns=column_titles)
    result_df = result_df.sort_values(by=['type', 'intensity', 'run'])

    os.chdir(curr_dir)
    with pd.HDFStore('results.h5') as store:
        store[f"{store_name}"] = result_df
    result_df.to_pickle(f"{store_name}.pkl")


def process_and_analyse_results(store_name, zscore=3):
    # df = pd.read_pickle("pickledf.pkl")
    with pd.HDFStore('results.h5') as store:
        results_df = store[f"{store_name}"]

    results_df = results_df[
        results_df.rmse < 100.0]  # remove runs that 'actually' failed: rmse whole orders of magnitude greater than anything else - skews results a lot!

    total_runs = np.max(results_df["run"] + 1)
    results_df_dict = {}

    type_list = unique(results_df["type"])
    # print(type_list)
    for test_type in type_list:
        type_df = results_df[results_df["type"] == test_type]
        intensity_list = unique(type_df["intensity"])

        test_data_list = [
            ["intensity", "rmse", "mean", "std_dev", "track_percent", "sve_rms", "sve_mean", "sve_std_dev", "tracking"]]

        for test_intensity in intensity_list:
            inten_df = type_df[type_df["intensity"] == test_intensity]
            # remove outliers by rmse z-score >= 3
            if inten_df["rmse"].size >= 2:
                valid_df = inten_df[(np.abs(stats.zscore(inten_df["rmse"])) < zscore)]
            else:
                continue

            rmse = np.nanmean(valid_df["rmse"])
            mean = np.nanmean(valid_df["mean"])
            std_dev = np.nanmean(valid_df["std_dev"])
            track_percent = np.nansum(valid_df["track_end_time"] - valid_df["track_start_time"]) / float(total_runs)
            sve_rms = np.nanmean(valid_df["sve_rms"])
            sve_mean = np.nanmean(valid_df["sve_mean"])
            sve_std_dev = np.nanmean(valid_df["sve_std_dev"])

            time_arr = np.linspace(0, 0, 1000)
            for t_start, t_end in zip(valid_df["track_start_time"], valid_df["track_end_time"]):
                t_start = int(np.round(t_start * 1000))
                t_end = int(np.round(t_end * 1000))
                time_arr[t_start:t_end + 1] += 1

            test_row = [test_intensity, rmse, mean, std_dev, track_percent, sve_rms, sve_mean, sve_std_dev, time_arr]
            test_data_list.append(test_row)

        col_titles = test_data_list.pop(0)
        results_df_dict[f"{test_type}"] = pd.DataFrame(test_data_list, columns=col_titles)

    """ Semi-Manual data processing stage """

    nan_row = pd.DataFrame(np.nan, columns=results_df_dict["plain"].columns, index=results_df_dict["plain"].index)
    nan_row = nan_row.astype(results_df_dict["plain"].dtypes)

    # add 0 intensity from plain (baseline) test
    results_df_dict["blur"] = pd.concat([results_df_dict["plain"], results_df_dict["blur"]], ignore_index=True)
    # add any tests which completely failed - rmse=nan, tracked=0
    blur_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i in blur_range:
        if i in results_df_dict["blur"]["intensity"].to_list():
            continue
        results_df_dict["blur"] = pd.concat([results_df_dict["blur"], nan_row], ignore_index=True)
        results_df_dict["blur"].iloc[-1, results_df_dict["blur"].columns.get_loc("intensity")] = i
        results_df_dict["blur"].iloc[-1, results_df_dict["blur"].columns.get_loc("track_percent")] = 0.0
        results_df_dict["blur"].iat[-1, results_df_dict["blur"].columns.get_loc("tracking")] = 0 * \
                                                                                               results_df_dict[
                                                                                                   "blur"].iloc[-2][
                                                                                                   "tracking"].copy()
        results_df_dict["blur"] = results_df_dict["blur"].sort_values(by=['intensity'])

    # add 0 size from plain (baseline) test
    results_df_dict["occ"] = pd.concat([results_df_dict["plain"], results_df_dict["occ"]], ignore_index=True)
    occ_range = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    for i in occ_range:
        if i in results_df_dict["occ"]["intensity"].to_list():
            continue
        results_df_dict["occ"] = pd.concat([results_df_dict["occ"], nan_row], ignore_index=True)
        results_df_dict["occ"].iloc[-1, results_df_dict["occ"].columns.get_loc("intensity")] = i
        results_df_dict["occ"].iloc[-1, results_df_dict["occ"].columns.get_loc("track_percent")] = 0.0
        results_df_dict["occ"].iat[-1, results_df_dict["occ"].columns.get_loc("tracking")] = 0 * \
                                                                                             results_df_dict[
                                                                                                 "occ"].iloc[-2][
                                                                                                 "tracking"].copy()
        results_df_dict["occ"] = results_df_dict["occ"].sort_values(by=['intensity'])

    # add 1.0 scale factor from plain (baseline) test
    results_df_dict["res"] = pd.concat([results_df_dict["res"], results_df_dict["plain"]], ignore_index=True)
    results_df_dict["res"].at[results_df_dict["res"]["intensity"] == 0, "intensity"] = 1.0
    # add any tests which completely failed - rmse=nan, tracked=0
    res_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in res_range:
        if i in results_df_dict["res"]["intensity"].to_list():
            continue
        results_df_dict["res"] = pd.concat([results_df_dict["res"], nan_row], ignore_index=True)
        results_df_dict["res"].iloc[-1, results_df_dict["res"].columns.get_loc("intensity")] = i
        results_df_dict["res"].iloc[-1, results_df_dict["res"].columns.get_loc("track_percent")] = 0.0
        results_df_dict["res"].iat[-1, results_df_dict["res"].columns.get_loc("tracking")] = 0 * \
                                                                                             results_df_dict[
                                                                                                 "res"].iloc[-2][
                                                                                                 "tracking"].copy()
        results_df_dict["res"] = results_df_dict["res"].sort_values(by=['intensity'], ascending=False)

    return results_df_dict


def do_plots(results_df_dict, total_runs, additional_string=None, save=False):
    """ Plotting """

    if type(additional_string) == type(""):
        additional_string = f"{additional_string}_"
    else:
        additional_string = ""

    figs = []
    fig_titles = []

    mpl.rcParams.update({'figure.autolayout': True})
    fig_titles.append(f"{additional_string}blur_graph")
    figs.append(plot_data(df=results_df_dict["blur"],
                          title="Effect of Gaussian Blur Kernel Size on\nTracking Accuracy and Scene Visibility Estimate",
                          xlabel=r"Blur Kernel Size ($n \times n$, pixels)"
                          ))

    fig_titles.append(f"{additional_string}res_graph")
    figs.append(plot_data(df=results_df_dict["res"],
                          title="Effect of Image Downsampling on\nTracking Accuracy and Scene Visibility Estimate",
                          xlabel=r"Downsampling Factor ($1 / n$)",
                          reverse_x=True
                          ))

    fig_titles.append(f"{additional_string}occ_graph")
    figs.append(plot_data(df=results_df_dict["occ"],
                          title="Effect of Occluded Area Size on\nTracking Accuracy and Scene Visibility Estimate",
                          xlabel=r"Occluded Area Size ($n \times n$, pixels)"
                          ))

    mpl.rcParams.update({'figure.autolayout': False})
    fig_titles.append(f"{additional_string}blur_heat")
    figs.append(
        plot_heatmap(data=results_df_dict["blur"]["tracking"], title="Heatmap of Tracked Time with varying Blur",
                     label_data=results_df_dict["blur"]["intensity"],
                     label_text=r"Blur Kernel Size ($n \times n$ pixels)",
                     legend_text="Number of Runs Tracking",
                     max=total_runs))

    fig_titles.append(f"{additional_string}res_heat")
    figs.append(
        plot_heatmap(data=results_df_dict["res"]["tracking"], title="Heatmap of Tracked Time with varying Downsampling",
                     label_data=results_df_dict["res"]["intensity"], label_text="Downsampling Factor ($1/n$)",
                     legend_text=r"Number of Runs Tracking",
                     max=total_runs))

    fig_titles.append(f"{additional_string}occ_heat")
    figs.append(
        plot_heatmap(data=results_df_dict["occ"]["tracking"], title="Heatmap of Tracked Time with varying Occlusion",
                     label_data=results_df_dict["occ"]["intensity"],
                     label_text=r"Occluded Area Size ($n \times n$ pixels)",
                     legend_text="Number of Runs Tracking",
                     max=total_runs))

    # figs.append(plt.figure())
    # ax1 = plt.subplot()
    # color = 'tab:blue'
    # ax1.plot(np.linspace(0.0, 0.999, 1000), heat_df_dict["occ"]["data"][5], linewidth=1.0, color=color)
    # ax1.set_xlabel("Normalised Time")
    # ax1.set_ylabel("Number of Runs Tracking", color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_xlim([0, 1.0])
    # ax1.set_ylim([0, 100])

    # df = pd.read_csv("/home/dom-ubuntu/Documents/fyp/datasets/rosbags/test_bags/occ/euroc_MH_04_occ250^2_sve_64.txt")
    # t_min = np.min(df["Time"])
    # t_max = np.max(df["Time"])
    # df["norm_time"] = (df["Time"] - t_min) / (t_max - t_min)
    # ax2 = ax1.twinx()
    # color = 'tab:red'
    # ax2.plot(df["norm_time"], df["SVE"], linewidth=1.0, color=color)
    # ax2.set_ylabel("SVE Magnitude", color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # # ax2.set_ylim([0, np.ceil(np.max(df["SVE"]))])
    #
    # plt.title("Number Tracking Compared with SVE")
    # plt.grid(True)

    if save:
        for title, fig in zip(fig_titles, figs):
            fig.savefig(f"images/results/{title}.eps", format="eps")
            fig.savefig(f"images/results/{title}.png", format="png", dpi=1000)
            fig.savefig(f"images/results/low_res/{title}.eps", format="eps")
            fig.savefig(f"images/results/low_res/{title}.png", format="png", dpi=50)


def plot_data(df, title, xlabel, reverse_x=False):
    fig = plt.figure()
    ax1 = plt.subplot()
    color = 'tab:blue'
    ax1.plot(df["intensity"], df["rmse"], linewidth=1.0, color=color, label="Tracking RMSE")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Tracking RMSE (m)", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, np.ceil(np.max(df["rmse"]))])
    if not reverse_x:
        ax1.set_xlim([0, np.max(df["intensity"])])
    else:
        ax1.set_xlim([np.max(df["intensity"]), 0])

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(df["intensity"], df["sve_mean"], linewidth=1.0, color=color, label="SVE Magnitude")
    ax2.fill_between(df["intensity"], df["sve_mean"] - df["sve_std_dev"], df["sve_mean"] + df["sve_std_dev"],
                     alpha=0.1, color=color)
    ax2.set_ylabel("SVE Metric ($\mu \pm \sigma$)", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, np.ceil(np.max(df["sve_rms"]))])

    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

    plt.title(title)
    plt.grid(True)
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    return fig


def plot_heatmap(data, title, label_data, label_text, legend_text, max, figs=None):
    """
    :param data: np.ndarray of 0-1 range of frequency for pixel
    :return:
    """
    data = data.reset_index(drop=True)
    label_data = label_data.reset_index(drop=True)
    (rows,) = data.shape
    (columns,) = data[0].shape

    # Create figure and adjust figure height to number of colormaps
    nrows = rows
    # figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    # fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    # fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
    #                     left=0.2, right=0.99)
    fig, axs = plt.subplots(nrows=nrows)
    fig.subplots_adjust(hspace=0)
    axs[0].set_title(title, fontsize=14)

    for row in range(rows):
        gradient = np.vstack((data[row], data[row]))
        im = axs[row].imshow(gradient, aspect='auto', cmap=plt.get_cmap('plasma'),
                             norm=mpl.colors.Normalize(vmin=0, vmax=max))
        axs[row].text(-0.01, 0.5, label_data[row], va='center', ha='right', fontsize=10,
                      transform=axs[row].transAxes)
        axs[row].grid(True)
        axs[row].xaxis.set_ticks(np.linspace(0.0, columns, 5))
        axs[row].yaxis.set_visible(False)
        for tick in axs[row].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

    for tick in axs[-1].xaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(True)
        tick.label2.set_visible(False)
    axs[-1].xaxis.set_ticks(np.linspace(0.0, columns, 5))
    axs[-1].xaxis.set_ticklabels([str(i) for i in np.linspace(0.0, 1.0, 5)])
    axs[-1].set_xlabel("Normalised Time")

    fig.text(x=-0.125, y=0.5, s=label_text, va='center', ha='right', fontsize=10,
             transform=axs[int(rows / 2)].transAxes, rotation=90)

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)

    cbar.set_ticks(np.arange(0, max + 1, max / 3))
    cbar.set_ticklabels([0.0, np.round(max / 3), np.round(2 * max / 3), max])
    cbar.set_label(legend_text, rotation=90, va="top")

    return fig


def plot_sves(sve_list, label_list, title, xlabel, reverse_x=False, colour_list=None, do_std_dev=True):
    fig = plt.figure()
    ax1 = plt.subplot()

    for sve, color, label_text in zip(sve_list, colour_list, label_list):
        if sve is not None:
            ax1.plot(sve["intensity"], sve["sve_mean"], linewidth=1.0, color=color, label=f"{label_text} $\mu$")
            if do_std_dev:
                ax1.fill_between(sve["intensity"], sve["sve_mean"] - sve["sve_std_dev"],
                                 sve["sve_mean"] + sve["sve_std_dev"], alpha=0.1, color=color,
                                 label=f'{label_text} $\mu\pm\sigma$')

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("SVE Magnitude")
    # ax1.set_ylabel("SVE Metric ($\mu \pm \sigma$)", color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, np.ceil(max([np.max(sve["sve_mean"]) for sve in sve_list]))])
    if not reverse_x:
        ax1.set_xlim([0, np.max(sve_list[0]["intensity"])])
    else:
        ax1.set_xlim([np.max(sve_list[0]["intensity"]), 0])

    plt.title(title)
    plt.grid(True)
    plt.legend()

    return fig


def sve_compare(results_list, label_list, colour_list, prepend_title=None, do_std_dev=True, save=False):
    if prepend_title is None:
        prepend_title = ""
    else:
        prepend_title = f"{prepend_title}_"

    figs = []
    titles = []

    for type, title, xlabel in zip(["blur", "occ", "res"], ["Effect of SVE Metric Modification\nwhen subjected to Blur",
                                                            "Effect of SVE Metric Modification\nwhen subjected to Occlusion",
                                                            "Effect of SVE Metric Modification\nwhen subjected to Downsampling"],
                                   ["Blur Kernel Size ($n x n$, pixels)", "Occluded Area Size ($n x n$, pixels)",
                                    "Downsampling Factor ($1/n$)"]):
        titles.append(type)
        sve_list = [result[type] for result in results_list]
        figs.append(plot_sves(sve_list=sve_list,
                              label_list=label_list,
                              title=title,
                              xlabel=xlabel,
                              do_std_dev=do_std_dev,
                              colour_list=colour_list,
                              reverse_x=(type == "res")
                              ))

        if save:
            for title, fig in zip(titles, figs):
                fig.savefig(f"images/results/{prepend_title}{title}_sve_comp.png", format="png", dpi=1000)
                fig.savefig(f"images/results/low_res/{prepend_title}{title}_sve_comp.png", format="png", dpi=50)

                fig.savefig(f"images/results/{prepend_title}{title}_sve_comp.eps", format="eps")
                fig.savefig(f"images/results/low_res/{prepend_title}{title}_sve_comp.eps", format="eps")

def plot_trajs(true_df, traj1_df, traj2_df=None, title="", xlabel="", reverse_x=False):
    fig = plt.figure()
    ax1 = plt.subplot()
    color = 'k'
    ax1.plot(true_df["intensity"], true_df["rmse"], linewidth=1.0, color=color, label='Original')
    # ax1.fill_between(sve1_df["intensity"], traj1_df["rmse"] - sve1_df["sve_std_dev"],
    #                  sve1_df["sve_mean"] + traj1_df["sve_std_dev"], alpha=0.1, color=color,
    #                  label='Un-Modified ($\mu \pm \sigma$)')

    color = 'tab:blue'
    ax1.plot(traj1_df["intensity"], traj1_df["rmse"], linewidth=1.0, color=color, label='Opt 1')

    if traj2_df is not None:
        color = 'tab:red'
        ax1.plot(traj2_df["intensity"], traj2_df["rmse"], linewidth=1.0, color=color, label='Opt 2')

    ax1.set_xlabel(xlabel)

    ind_max = []
    ind_max.append(np.max(true_df["rmse"]))
    ind_max.append(np.max(traj1_df["rmse"]))
    if traj2_df is not None:
        ind_max.append(np.max(traj2_df["rmse"]))

    ax1.set_ylim([0, np.ceil(max(ind_max))])
    if not reverse_x:
        ax1.set_xlim([0, np.max(true_df["intensity"])])
    else:
        ax1.set_xlim([np.max(true_df["intensity"]), 0])

    ax1.set_ylabel("Tracking RMSE (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    return fig

def traj_compare(results_df_dict, track_results_df_dict, track2_results_df_dict, prepend_title=None, save=False):
    if prepend_title is None:
        prepend_title = ""
    else:
        prepend_title = f"{prepend_title}_"

    figs = []
    fig_titles = []

    mpl.rcParams.update({'figure.autolayout': True})
    fig_titles.append(f"switch_comp_blur_graph")
    figs.append(plot_trajs(true_df=results_df_dict["blur"],
                           traj1_df=track_results_df_dict["blur"],
                           traj2_df=track2_results_df_dict["blur"],
                           title="Effect of Gaussian Blur Kernel Size on\nTracking Accuracy Across Switching Behaviours",
                           xlabel=r"Blur Kernel Size ($n \times n$, pixels)"
                           ))

    fig_titles.append(f"switch_comp_res_graph")
    figs.append(plot_trajs(true_df=results_df_dict["res"],
                           traj1_df=track_results_df_dict["res"],
                           traj2_df=track2_results_df_dict["res"],
                           title="Effect of Image Downsampling on\nTracking Accuracy and Scene Visibility Estimate",
                           xlabel=r"Downsampling Factor ($1 / n$)",
                           reverse_x=True
                           ))

    fig_titles.append(f"switch_comp_occ_graph")
    figs.append(plot_trajs(true_df=results_df_dict["occ"],
                           traj1_df=track_results_df_dict["occ"],
                           traj2_df=track2_results_df_dict["occ"],
                           title="Effect of Occluded Area Size on\nTracking Accuracy and Scene Visibility Estimate",
                           xlabel=r"Occluded Area Size ($n \times n$, pixels)"
                           ))

    if save:
        for title, fig in zip(fig_titles, figs):
            fig.savefig(f"images/results/{prepend_title}{title}.eps", format="eps")
            fig.savefig(f"images/results/low_res/{prepend_title}{title}.eps", format="eps")
            fig.savefig(f"images/results/{prepend_title}{title}.png", format="png", dpi=1000)
            fig.savefig(f"images/results/low_res/{prepend_title}{title}.png", format="png", dpi=50)

if __name__ == "__main__":
    # process_tests(testdir="/home/dom-ubuntu/Documents/fyp/datasets/rosbags/test_bags", store_name="sve_unmod")
    # process_tests(testdir="/home/dom-ubuntu/Documents/fyp/datasets/rosbags/test2_bags", store_name="sve_mod")
    # process_tests(testdir="/home/dom-ubuntu/Documents/fyp/datasets/rosbags/test3_bags", store_name="sve_track")
    # process_tests(testdir="/home/dom-ubuntu/Documents/fyp/datasets/rosbags/test4_bags", store_name="sve_track2")
    # process_tests(testdir="/home/dom-ubuntu/Documents/fyp/datasets/rosbags/test5_bags", store_name="sve_track3")
    results_df_dict = process_and_analyse_results(store_name="sve_unmod")
    # mod_results_df_dict = process_and_analyse_results(store_name="sve_mod")  # same sve as track1
    track_results_df_dict = process_and_analyse_results(store_name="sve_track")
    # track2_results_df_dict = process_and_analyse_results(store_name="sve_track2")  # without sqrt
    track2_results_df_dict = process_and_analyse_results(store_name="sve_track3")  # with sqrt

    z1_results_df_dict = process_and_analyse_results(store_name="sve_unmod", zscore=1)
    # z1_mod_results_df_dict = process_and_analyse_results(store_name="sve_mod", zscore=1)  # same sve as track1
    z1_track_results_df_dict = process_and_analyse_results(store_name="sve_track", zscore=1)
    # z1_track2_results_df_dict = process_and_analyse_results(store_name="sve_track2", zscore=1)
    z1_track2_results_df_dict = process_and_analyse_results(store_name="sve_track3", zscore=1)

    do_plots(results_df_dict, 100, "unmod", save=True)
    # do_plots(mod_results_df_dict, 100, "mod", save=False)
    do_plots(track_results_df_dict, 100, "track", save=True)
    do_plots(track2_results_df_dict, 100, "track2", save=True)
    plt.close('all')

    do_plots(z1_results_df_dict, 100, "z1_unmod", save=True)
    # do_plots(z1_mod_results_df_dict, 100, "z1_mod", save=False)
    do_plots(z1_track_results_df_dict, 100, "z1_track", save=True)
    do_plots(z1_track2_results_df_dict, 100, "z1_track2", save=True)
    plt.close('all')

    sve_compare(results_list=[results_df_dict, track_results_df_dict], colour_list=["k", "tab:blue"],
                label_list=["Original", "Opt 1"], prepend_title="org_opt1", save=True)
    sve_compare(results_list=[results_df_dict, track2_results_df_dict], colour_list=["k", "tab:red"],
                label_list=["Original", "Opt 2"], prepend_title="org_opt2", save=True)
    plt.close('all')

    traj_compare(results_df_dict, track_results_df_dict, track2_results_df_dict, save=True)
    traj_compare(z1_results_df_dict, z1_track_results_df_dict, z1_track2_results_df_dict, prepend_title="z1", save=True)
    plt.close('all')

    # plt.show(block=True)
    plt.close('all')
