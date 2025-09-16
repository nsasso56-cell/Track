#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:38:32 2025

@author: sasson
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import epygram
import json
import cartopy.crs as ccrs
import logging
import get_orbits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.mpl.ticker as cticker
from traject import *
from netCDF4 import Dataset
from visu_traject import plot_single_track, plot_single_diag
from Tools import diagdef, guess_diag

import matplotlib.dates as mdates


matplotlib.rcParams["figure.dpi"] = 200
# matplotlib.use('Agg')


# Directories
output_dir = '/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/outputs'
input_dir = '/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/inputs'


track_name = 'VDG-2021121000-6'
tag = 'dep_atln_v2'
xp_nr = 'GTDJ'
color_nr = 'b'

diags = ['mslp_min_p', 'ff10m_max_s4']

listxp = ['GV8T', 'GXCE']
labels = ['Ctrl', 'WIVERN err2']
colors = ['orange', 'green']

# domtraj={"lonmin":20,"latmin":-40,"lonmax":180,"latmax":-2} #Indian ocean
# domtraj = {"lonmin": -60, "latmin": 40,
#            "lonmax": -20, "latmax": 65}  # ATLN
domtraj = {"lonmin": -50, "latmin": 45,
           "lonmax": 0, "latmax": 60}  # ATLN

begin_date = '2021121000'
end_date = '2022031000'

# %%

savefig_dir = output_dir + f'/{track_name}'
os.makedirs(savefig_dir, exist_ok=True)

# Get cyclone from the NR :
algo = 'VDGfree'
origin = 'an'
input_path = f'{output_dir}/{xp_nr}/track_{track_name}_{tag}_{algo}_{xp_nr}_{origin}_{begin_date}_to_{end_date}_v0.92.json'
assert os.path.isfile(input_path), f"{input_path} doesn't exist."

ref_traj = Read(input_path)
print(f'\n# Read file {input_path}.\n')

# Récupération des trajectoires des xps assim
algo = 'VDG'
origin = 'fc'
trajs = {}
list_basetime = {}
for i_exp, xp in enumerate(listxp):

    input_path = f'{output_dir}/{xp}/track_{tag}_{algo}_{xp}_{origin}_{begin_date}_to_{end_date}_v0.92.json'
    ltraj = Read(input_path)
    trajs[xp] = Select(ltraj, {"name": track_name})
    list_basetime[xp] = []
    for i in range(len(trajs[xp])):
        list_basetime[xp].append(trajs['GV8T'][i].basetime)


for i_b, basetime in enumerate(list_basetime[listxp[0]]):
    print(f'\n# {basetime} :')

    savefig_path = f'{savefig_dir}/tracks_multiexp_{track_name}_{tag}_{basetime}'

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.PlateCarree()})

    title = f'{track_name}. {tag}. Spaghetti plot. {basetime}'
    ax.set_title(title)
    y_txt = 0.99
    plot_single_track(ref_traj[0], col=color_nr)
    text = f'Start: {ref_traj[0].traj[0].__dict__["time"]}'
    plt.text(.01, y_txt, text, ha='left', va='top',
             transform=ax.transAxes, color=color_nr)

    traj2plot = []
    for i_exp, xp in enumerate(listxp):
        traj2plot.append(Select(trajs[xp], {"basetime": basetime})[0])
        plot_single_track(traj2plot[i_exp], col=colors[i_exp])
        text = f'Start: {traj2plot[i_exp].traj[0].__dict__["time"]}'
        plt.text(.01, y_txt-0.07 * (i_exp+1), text, ha='left', va='top',
                 transform=ax.transAxes, color=colors[i_exp])

    ax.legend(['Ref'] + labels, loc='lower left')

    ax.coastlines()
    ax.set_xticks(np.linspace(
        domtraj['lonmin'], domtraj['lonmax'], 6), crs=projection)
    ax.set_yticks(np.linspace(
        domtraj['latmin'], domtraj['latmax'], 6), crs=projection)
    ax.set_xticklabels(np.linspace(
        domtraj['lonmin'], domtraj['lonmax'], 6))
    ax.set_yticklabels(np.linspace(
        domtraj['latmin'], domtraj['latmax'], 6))
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.grid(linewidth=1, color='lightgray', linestyle='--')
    ax.set_extent([domtraj['lonmin'], domtraj['lonmax'],
                  domtraj['latmin'], domtraj['latmax']])
    fig.tight_layout()

    for save_format in ['png']:
        fig.savefig(savefig_path + '.' + save_format,
                    format=save_format, dpi=200)
        print(f'# Figure saved in {savefig_path}.{save_format}')
    plt.show()

    for diag in diags:
        savefig_path = f'{savefig_dir}/diag_{diag}_multiexp_{track_name}_{tag}_{basetime}'

        fig, ax = plt.subplots()

        title = f'{track_name}. {tag}.\n {diag}. {basetime}'
        ax.set_title(title)
        fct = 1
        if diag == "mslp_min_p":
            fct = 0.01  # (conversion to hPa)

        plot_single_diag(ref_traj[0], diag, factor=fct, col=color_nr)

        diag2plot = []
        for i_exp, xp in enumerate(listxp):
            diag2plot.append(Select(trajs[xp], {"basetime": basetime})[0])
            plot_single_diag(traj2plot[i_exp], diag,
                             factor=fct, col=colors[i_exp])

        ax.legend(['Ref'] + labels, loc='best')

        ax.grid(linewidth=1, color='lightgray', linestyle='--')
        # set graphical parameters
        # plt.xticks(rotation=45)
        myFmt = mdates.DateFormatter('%Y/%m/%d %H:00')
        ax.xaxis.set_major_formatter(myFmt)

        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        ax.set_label('Time')
        ax.set_ylabel(diag)
        fig.tight_layout()

        for save_format in ['png']:
            fig.savefig(savefig_path + '.' + save_format,
                        format=save_format, dpi=200)
            print(f'# Figure saved in {savefig_path}.{save_format}')
        plt.show()
