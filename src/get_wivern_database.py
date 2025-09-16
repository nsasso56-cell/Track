#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:40:54 2025

Tool to select a single track in a track file, and write it in a separate .json file.

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
epygram.init_env()
os.environ["ECCODES_SAMPLES_PATH"] = (
    "/home/common/epygram/ext/eccodes/share/eccodes/samples")

# logging.basicConfig(​filename='track_and_wivern_plots.log', ​level=logging.DEBUG, ​format=' %(asctime)s -  %(levelname)s -  %(message)s'​)​

matplotlib.rcParams["figure.dpi"] = 200
matplotlib.use('Agg')


output_dir = '/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/outputs'
input_dir = '/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/inputs'


# Input variables from user
# Name of the track to put in the new .json.
track_name = 'VDG-2021121000-6'

xp = 'GTDJ'
tag = 'dep_atln_v2'
begin_date = '2021121000'
end_date = '2022031000'

domtraj = {"lonmin": -60.0, "latmin": 30.0,
           "lonmax": 30.0, "latmax": 75.0}  # ATLN
altitude_obs = [3750, 4250]

# %% Setup variables and directories.
algo = 'VDG'
cutoff = 'fc'
if xp == 'GTDJ':
    algo = 'VDGfree'
    cutoff = 'an'

savefig_dir = output_dir + f'/{track_name}'
os.makedirs(savefig_dir, exist_ok=True)

algodef = f'{input_dir}/namtraject_{tag}_{algo}.json'
indef = f'{input_dir}/indef_{xp}.json'

input_path = f'{output_dir}/{xp}/track_{track_name}_{tag}_{algo}_{xp}_{cutoff}_{begin_date}_to_{end_date}_v0.92.json'

assert os.path.isfile(input_path), f"{input_path} doesn't exist."
assert os.path.isfile(algodef), f"{algodef} doesn't exist."
assert os.path.isfile(indef), f"{indef} doesn't exist."

# Opening of the .json :
ltraj = Read(input_path)
print(f'\n# Read file {input_path}.\n')

ltraj2 = []

print(f'\n# Select track named {track_name}.\n')
traj = Select(ltraj, {"name": track_name})

# Read variables from trajfile
latc = []
lonc = []
time = []
hour = []
mslp = []
if len(ltraj) == 1:
    # read variables
    traj = ltraj[0]
    for ivi in range(traj.nobj):
        latc.append(traj.traj[ivi].__dict__['latc'])
        lonc.append(traj.traj[ivi].__dict__['lonc'])
        time.append(traj.traj[ivi].__dict__['time'])
        hour.append(traj.traj[ivi].__dict__['time'][-2:])
        mslp.append(traj.traj[ivi].__dict__['mslp_min_p'])

date_str = []
for date in time:
    date_str.append(date[:8] + 'T' + date[8:] + '00A')


# Récupération des données
Z = []
lon = []
lat = []
odb_key_val_dic = {}

i = len(date_str)
# i = 1
for i_date, date in enumerate(date_str[0:i]):

    # Get ODB database of WIVERN observations :
    odb_name = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/GWNM/{date}/screening/odb-ecma.screening.wivern/sensor_wivern_SID_ID.nc'
    assert os.path.isfile(odb_name), f"{odb_name} does not exist !"

    myfile = Dataset(odb_name, 'r')
    for variables in myfile.variables.keys():
        key_int = myfile[variables].odb_name[:myfile[variables].odb_name.index(
            '@')]
        odb_key_val_dic[key_int] = myfile[variables][:]

    myfile.close()

    # Filtering the observations around 3km height and in the domtraj domain
    ind_domain = np.where((odb_key_val_dic['vertco_reference_2'] > altitude_obs[0]) & (
        odb_key_val_dic['vertco_reference_2'] < altitude_obs[1]) &
        (odb_key_val_dic['lon'] > domtraj['lonmin']) & (odb_key_val_dic['lon'] < domtraj['lonmax']) &
        (odb_key_val_dic['lat'] > domtraj['latmin']) & (odb_key_val_dic['lat'] < domtraj['latmax']+10))

    retrtype = odb_key_val_dic['retrtype']
    Z.append(10*np.log10(retrtype[ind_domain]/1e6))
    lon.append(odb_key_val_dic['lon'][ind_domain])
    lat.append(odb_key_val_dic['lat'][ind_domain])

    #


for i_date, date in enumerate(date_str[0:i]):

    # # Get orbits of WIVERN on the specific assim time :
    # df = get_orbits.get_orbits(int(date[:4]), int(
    #     date[4:6]), int(date[6:8]), int(date[9:11]))
    # print('\n# Reading orbits file : ok')
    # ind_scanline_1 = np.where((df.Scanline_position_id == 15) & (
    #     df.longitude > domtraj["lonmin"]) & (df.longitude < domtraj["lonmax"]) & (df.latitude > domtraj["latmin"]) & (df.latitude < domtraj["latmax"] + 10))
    # ind_scanline_2 = np.where((df.Scanline_position_id == 45) & (
    #     df.longitude > domtraj["lonmin"]) & (df.longitude < domtraj["lonmax"]) & (df.latitude > domtraj["latmin"]) & (df.latitude < domtraj["latmax"]+10))

    savefig_path = savefig_dir + f'/track_and_WIVERN_{track_name}_{tag}_{date}'
    # Figure
    cmap = plt.get_cmap('hot_r', 10)
    latmin = domtraj
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.PlateCarree()})

    title = f'{date}. Altitude of obs [{altitude_obs[0]/1000}: {altitude_obs[1]/1000}] km'
    ax.set_title(title)
    # Scatter plot des obs WIVERN de reflectivité :
    im = ax.scatter(lon[i_date], lat[i_date], c=Z[i_date],
                    s=0.25, alpha=1, cmap=cmap, vmin=-15, vmax=15)

    # Plot de la traj du cyclone
    ax.plot(lonc[0:i_date+1], latc[0:i_date+1], marker='o',
            linewidth=1, markersize=1, color='k', label=f'{track_name}')
    ax.plot(lonc[0], latc[0], marker='x', markersize=4, color='k')

    # ax.scatter(df.longitude[ind_scanline_1[0]],
    #            df.latitude[ind_scanline_1[0]], s=0.5, color='k')
    # ax.scatter(df.longitude[ind_scanline_2[0]],
    #            df.latitude[ind_scanline_2[0]], s=0.5, color='k')

    ax.legend(loc='lower left')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%",
                              pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Reflect. Z [dBz]')

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
        print(f'# Fig saved : {savefig_path}.{save_format}')
        fig.savefig(savefig_path + f'.{save_format}',
                    format=save_format, dpi=200)
