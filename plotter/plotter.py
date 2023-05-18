#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

import matplotlib.pylab as pl
import matplotlib.ticker as plticker
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from matplotlib.colors import LinearSegmentedColormap

def pareto2():
    filename = "/tmp/pareto"
    if not os.path.isfile(filename):
        return

    data = np.genfromtxt(filename, delimiter=",")

    fig = plt.figure(figsize=(9.6, 7.2))
    plt.rcParams["svg.fonttype"] = "none"
    # fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = data[:, 0]
    # print(xs.max(), xs.min())
    ys = data[:, 1]
    # print(ys.max(), ys.min())
    zs = data[:, 2]
    # print(zs.max(), zs.min())
    ax.scatter(xs, ys, zs, marker="o")

    # plot the shadows
    # ax.plot(xs, zs, 'r+', zdir='y', zs=30000)
    # ax.plot(ys, zs, 'g+', zdir='x', zs=2300)
    # ax.plot(xs, ys, 'k+', zdir='z', zs=18.0)

    # define limits
    # ax.set_xlim([2300, 2600])
    # ax.set_ylim([20000, 30000])
    # ax.set_zlim([18.0, 18.21])

    ax.set_xlabel("Resilience")
    ax.set_ylabel("Performance")
    ax.set_zlabel("Cost")

    ax.view_init(elev=24, azim=-57)
    # plt.show()
    plt.savefig(f"/tmp/pareto.svg")


def fitness(objective: int, filename: str):
    # plt.close("all")
    LABELS = ["Model Performance", "Cost", "Network Cost", "Network Failure Probability"]
    COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    file = f"tmp/data/fitnesses/fitnesses[{filename}]"
    if not os.path.isfile(file):
        logging.error(f"There is no {file} file.")
        return

    data = np.genfromtxt(file, delimiter=",")

    #ylabel = ["Resilience", "Model Performance", "Cost", "Network Performance", "Net Cost", "Net Fail Prob"]
    # color = ["red", "green", "blue", "yellow"]
    #color = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#9b19f5", "#ffa300"]

    ylabel = LABELS[:]
    color = COLORS[:]


    plt.figure()
    # from matplotlib import rcParams
    # import matplotlib.font_manager as fm
    # fm._rebuild()
    #hfont = {"fontname": "Courier"}
    plt.rcParams["svg.fonttype"] = "none"
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams["font.fontname"] = "Courier"
    # rcParams['font.sans-serif'] = ['Tahoma']
    plt.plot(data[:, objective], color=color[objective])
    plt.ylabel(ylabel=ylabel[objective])
    #plt.xlabel("Number of generations", **hfont)
    plt.xlabel("Number of generations")

    # red_patch = mpatches.Patch(color='red', label='The red data')
    # plt.legend(handles=[red_patch], loc='upper left')

    # plt.show()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"tmp/plots/fitnesses/Fitness({ylabel[objective]})[{filename}].svg")


def read_times(filename: str):
    if not os.path.isfile(filename):
        return
    data = pd.read_csv(filename, encoding='cp1252')
    print(data.head())
    means = []
    stds = []
    times = data['avg±std_constraints_time']
    for time in times:
        time_splits = time.split("±")
        means.append(float(time_splits[0].strip()))
        stds.append(float(time_splits[1].replace('s', ' ').strip()))
    return means, stds


def times():

    means_big, stds_big = read_times("./tmp/data/testbed/testbed_big_mipc.csv")
    means_small, stds_small = read_times("./tmp/data/testbed/testbed_small_mipc.csv")
    means_small.append(0)
    stds_small.append(0)
    xlabels = pd.read_csv("./tmp/data/testbed/testbed_big_mipc.csv", encoding='cp1252')['pipeline']
    print(xlabels)
    x = np.arange(len(xlabels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams["svg.fonttype"] = "none"
    rects1 = ax.bar(
        x - width,
        means_big,
        width,
        yerr=stds_big,
        label="Big Infrastructure",
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
        color="g",
    )
    rects2 = ax.bar(
        x,
        means_small,
        width,
        yerr=stds_small,
        label="Small Infrastructure",
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
        color="b",
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Execution time (s)")
    ax.set_xlabel("Number of models in the pipeline")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.yaxis.grid(True)
    ax.legend()

    # Save the figure and show
    plt.yscale("log")
    plt.title("Time to met constraints")
    plt.tight_layout()
    plt.savefig(f"./tmp/plots/times/times.svg")


def _memory():
    plt.figure()
    plt.rcParams["svg.fonttype"] = "none"

    year_n_1 = [1.5, 3, 10, 13, 22, 36, 30, 33, 24.5, 15, 6.5, 1.2]
    year_n = [2, 7, 14, 17, 20, 27, 30, 38, 25, 18, 6, 1]

    plt.fill_between(
        np.arange(12), year_n_1, color="lightpink", alpha=0.5, label="year N-1"
    )
    plt.fill_between(np.arange(12), year_n, color="skyblue", alpha=0.5, label="year N")

    plt.legend()


def memory():
    import matplotlib.pylab as pl
    from mpl_toolkits.mplot3d import Axes3D

    m1 = np.genfromtxt("massif_5.csv") / 1024**2
    m2 = np.genfromtxt("massif_10.csv") / 1024**2
    m3 = np.genfromtxt("massif_20.csv") / 1024**2
    m4 = np.genfromtxt("massif_40.csv") / 1024**2
    m5 = np.genfromtxt("massif_80.csv") / 1024**2

    x = np.arange(m1.size)

    y1 = np.ones(x.size)
    y2 = np.ones(x.size) * 2
    y3 = np.ones(x.size) * 3
    y4 = np.ones(x.size) * 4
    y5 = np.ones(x.size) * 5

    pl.figure()
    pl.rcParams["svg.fonttype"] = "none"
    ax = pl.subplot(projection="3d")
    ax.plot(x, y1, m1, color="r", label="5 models")
    ax.plot(x, y2, m2, color="g", label="10 models")
    ax.plot(x, y3, m3, color="b", label="20 models")
    ax.plot(x, y4, m4, color="c", label="40 models")
    ax.plot(x, y5, m5, color="m", label="80 models")

    ax.legend()

    ax.add_collection3d(pl.fill_between(x, m1, color="r", alpha=0.1), zs=1, zdir="y")
    ax.add_collection3d(pl.fill_between(x, m2, color="g", alpha=0.1), zs=2, zdir="y")
    ax.add_collection3d(pl.fill_between(x, m3, color="b", alpha=0.1), zs=3, zdir="y")
    ax.add_collection3d(pl.fill_between(x, m4, color="c", alpha=0.1), zs=4, zdir="y")
    ax.add_collection3d(pl.fill_between(x, m5, color="m", alpha=0.1), zs=5, zdir="y")

    ax.set_xlabel("Time (s)")
    ax.set_zlabel("Memory (Megabytes)")

    ax.view_init(elev=22, azim=-114)

    plt.savefig(f"/tmp/memory.svg")


def get_single_color(data, i: int, j: int, x: float, y: float):
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    cm = LinearSegmentedColormap.from_list("", [colors[i], colors[j]])

    if i == j:
        return colors[i]

    y_max = data[:, i].max()
    y_min = data[:, i].min()
    x_max = data[:, j].max()
    x_min = data[:, j].min()
    for x2, y2 in zip(data[:, j], data[:, i]):
        if i == 0 and y2 > y and x2 < x:
            return "lightgrey"
        elif j == 0 and x2 > x and y2 < y:
            return "lightgrey"
        elif i != 0 and j != 0 and x2 < x and y2 < y:
            return "lightgrey"
    distance1 = sqrt((x_min - x) ** 2 + (y_max - y) ** 2)
    distance2 = sqrt((x_max - x) ** 2 + (y_min - y) ** 2)
    ratio = int((distance1 / (distance1 + distance2)) * 255)
    return cm(ratio)


def get_colors(data, i: int, j: int):
    for x, y in zip(data[:, j], data[:, i]):
        yield get_single_color(data=data, i=i, j=j, x=x, y=y)


def pareto(filename: str):
    OBJECTIVES_LABELS = ["Model Perf", "Cost", "Net Cost", "Net Fail Prob"]
    file = f"tmp/data/paretos/pareto[{filename}].txt"
    if not os.path.isfile(file):
        logging.error(f"There is no {file} file.")
        return

    data = np.genfromtxt(file, delimiter=",")
    num_objectives = data.shape[1]

    plt.close("all")
    plt.rcParams["svg.fonttype"] = "none"
    _, axis = plt.subplots(num_objectives, num_objectives, figsize=(15, 15))

    for i in range(num_objectives):
        for j in range(num_objectives):
            my_colors = list(get_colors(data=data, j=j, i=i))
            y_max = data[:, i].max()
            y_min = data[:, i].min()
            x_max = data[:, j].max()
            x_min = data[:, j].min()
            axis[i, j].set_xlim([x_min-0.01, x_max+0.01])
            axis[i, j].set_ylim([y_min-0.01, y_max+0.01])
            axis[i, j].scatter(data[:, j], data[:, i], c=my_colors)
            #axis[i, j].set_xticklabels([label for label in np.arange(x_min, x_max)])
            #axis[i, j].set_yticklabels([label for label in np.arange(y_min, y_max)])
            if j == 0:
                axis[i, j].set_ylabel(OBJECTIVES_LABELS[i])
            if i == num_objectives-1:
                axis[i, j].set_xlabel(OBJECTIVES_LABELS[j])
    plt.tight_layout()
    # Combine all the operations and display
    plt.savefig(f"tmp/plots/paretos/Pareto[{filename}].svg")

def main():
    text = "This is the plotter."
    parser = argparse.ArgumentParser(description=text)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-f",
        "--fitnesses",
        type=str,
        default=None,
        #action="store_true",
        help="Generate Fitness Plots.",
        required=False,
    )
    required.add_argument(
        "-p",
        "--pareto",
        type=str,
        default=None,
        #action="store_true",
        help="Generate Pultiplot.",
        required=False,
    )
    required.add_argument(
        "-t",
        "--times",
        action="store_true",
        help="Generate Times Plot.",
        required=False,
    )

    args = parser.parse_args()

    if args.fitnesses:
        for o in range(5):
            fitness(objective=o, filename=args.fitnesses)

    if args.pareto:
        pareto(filename=args.pareto)

    if args.times:
        times()

    # memory()

    plt.show()


if __name__ == "__main__":
    main()
