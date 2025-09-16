#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:18:50 2025

@author: sasson
"""

import os
from traject import *
from visu_traject import *
from score_traject import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd # type: ignore
import Tools
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator

mpl.rcParams["figure.dpi"] = 200
mpl.rcParams.update({
    "text.usetex": True,
    "font.size": 14
})
mpl.use('Agg')


# Directories
workspaceFolder = '/Users/nicolassasso/Documents/Python_projects/Track'

repin = os.path.join(workspaceFolder,'inputs')
repout = os.path.join(workspaceFolder,'outputs')

cutoff = 'fc'
cyclone_name = ''
tag = 'dep_atln'
algo = 'VDG'
# xp = ['GRX6','GTUW','GX9Y']
# labels = ['NR','Ctrl','WIVERN err2']

xp = ['GTDJ', 'GV8T', 'GXCE']
labels = ['NR', 'Ctrl', 'WIVERN_err2']

# dom={"lonmin":100,"latmin":-40,"lonmax":120,"latmax":-10}
dom = {"lonmin": 20, "latmin": -40, "lonmax": 180, "latmax": -2}

timetraj = {'start': "2021121000", 'final': "2022031000", 'step': "6"}

basetime = timetraj['start']
endtime = timetraj['final']

traj_path = {}
ltraj = []
indef = []
for i_exp, exp in enumerate(xp):

    indef_file = open(repin + f'/indef_{exp}.json')
    indef.append(json.load(indef_file))
    indef_file.close()

    origin = indef[i_exp]['origin']

    if (origin == 'an') | (exp == 'GRX6') | (exp == 'GTDJ'):  # if Nature Run
        # .json file for traj
        traj_path[exp] = f'{repout}/{exp}/track_{tag}_VDGfree_{exp}_{origin}_{basetime}_to_{endtime}_v0.92.json'
    else:
        # .json file for traj
        traj_path[exp] = f'{repout}/{exp}/track_{tag}_{algo}_{exp}_{origin}_{basetime}_to_{endtime}_v0.92.json'

    if os.path.exists(traj_path[exp]) == True: # If the trajectories exist :
        if cyclone_name == '':
            a = Read(traj_path[exp])
            if len(xp) > 1:
                ltraj.append(a)
            else:
                ltraj = a
        else:
            a = Read(traj_path[exp])
            a2 = Select(a, {"name": cyclone_name})
            ltraj.append(a2[0])
    else:
        print(f'# WARNING : No trajectory file for exp {exp}')

# Scores computation
diags = ["TTE", "mslp_min_p", "ff10m_max_s4",
         "ff925_max_s4", "ff300_max_s4", "z500_p"]
# diags = ["TTE","mslp_min_p","ff10m_max_s4"]
diags = ['TTE']


df = {}

for i_exp, exp in enumerate(xp[1:]):
    if os.path.exists(f'{repout}/score_cyclones_{exp}.csv') == True:
        print(f'# File {repout}/score_cyclones_{exp}.csv exists. No score computation needed')
        
    else:
        df[exp] = score_comp_diff(ltraj[0], ltraj[i_exp+1], timetraj,
                                diags, 1, f'{repout}/score_cyclones_{exp}.csv', proj="merc")


# %%

for i_exp, exp in enumerate(xp[1:]):
    score_dirfig = f'{repout}/{exp}/figs'
    os.makedirs(score_dirfig, exist_ok=True)

    csv_file = f'{repout}/score_cyclones_{exp}.csv'

    for diag in diags:
        fig, ax = start_figure_score()
        plot_boxplot_score(ax, csv_file, diag, 6, 96, 10, 90)
        fig.suptitle(
            f'{exp} vs {xp[0]}.\n{labels[i_exp+1]} vs {labels[0]}.\n{basetime} to {endtime}.')
        fig.savefig(score_dirfig +
                    f'/{tag}_boxplot_{diag}.png', format='png', dpi=300)
        # plt.show()

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax = axes.flat

labels = ['(a) Control', '(b) WIVERN']

diag = 'ff300_max_s4'
for i_exp, exp in enumerate(xp[1:]):
    csv_file = f'{repout}/score_cyclones_{exp}.csv'
    plot_boxplot_score(ax[i_exp], csv_file, diag, 6, 96, 10, 90)
    ax[i_exp].set_title(rf'{labels[i_exp]}')
    ax[i_exp].set_ylim(-18, 7)

ax[1].set_ylabel('')


fig.tight_layout
fig.savefig(repout +
            f'/{tag}_boxplot_{diag}_2exp.png', format='png', bbox_inches = 'tight', dpi=300)


# %%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
ax = axes.flat
diags = ["mslp_min_p", "TTE",
         "ff10m_max_s4", "ff300_max_s4"]

suptitles = ['(a) Min. mslp','(b) Total Track Error','(c) Max. 10m wind speed','(d) Max. 300hPa wind speed']
ylabels = ['(hPa)','(km)','(m/s)','(m/s)']
for idiag, diag in enumerate(diags):
    colors = ['blue', 'orange', 'magenta']
    # ax=plt.axes()
    for i_exp, exp in enumerate(xp[1:]):
        plot_line_score(ax[idiag], f'{repout}/score_cyclones_{xp[i_exp+1]}.csv', diag, 6,
                        96, "mean", linewidth=2, color=colors[i_exp], label=labels[i_exp]+' mean')
        # plot_line_score(ax, f'{repout}/score_cyclones_{xp[i_exp+1]}.csv', diag, 6, 96, "std",color=colors[i_exp],linestyle="--", label=labels[i_exp+1]+' std')
    # plot_line_score(ax, f'{repout}/score_cyclones_{xp[2]}.csv', diag, 6, 96, "mean",color="orange")
    # plot_line_score(ax, f'{repout}/score_cyclones_{xp[2]}.csv', diag, 6, 96, "std",color="orange",linestyle="--")
    # plot_line_score(fig, "score_PanguERA5.csv", diag, 6, 102, "mean",color="red")
    # plot_line_score(fig, "score_PanguERA5.csv", diag, 6, 102, "q90",color="red",linestyle="--")
    ax[idiag].hlines(y=0, xmin=-10, xmax=150, linewidth=1, color='k')
    ax[idiag].legend(ncol=len(xp[1:]))
    ax[idiag].grid(which='major', linewidth=1.5,
            color='gray', alpha=0.5, linestyle='-')
    ax[idiag].grid(which='minor', linewidth=1.5,
            color='gray', alpha=0.5, linestyle=':')
    ax[idiag].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[idiag].xaxis.set_tick_params(length=5, width=1, direction='out')
    ax[idiag].yaxis.set_tick_params(length=5, width=1, direction='out')
    ax[idiag].set_ylabel(ylabels[idiag])
    ax[idiag].set_xlim(-1, 97)
    ax[idiag].set_title(suptitles[idiag])

    ax[idiag].set_xlabel("Forecast term [h]")  # , ha='left')

#fig.title(diag)
fig.tight_layout
fig.savefig(f"{repout}/{tag}_scores_.png", format='png', bbox_inches='tight', dpi=300)
