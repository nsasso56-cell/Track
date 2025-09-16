#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:22:09 2024

@author: sasson
"""

import os
import epygram
from epygram.extra import usevortex as vtx
from traject import *
from visu_traject import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import uniform_filter
from datetime import datetime, timedelta
import json

import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable


epygram.init_env()
matplotlib.rcParams["figure.dpi"] = 200
# matplotlib.rc('font', size=18)
matplotlib.use('Agg')
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11
})


# %%

# Directories
repin = './inputs/'
repout = f'./outputs/'

track_name = 'VDG-2021121000-6'
algo = 'VDGfree'
tag = 'dep_atln_v2'
xp = 'GTDJ'
label = 'WIVERN err2'

begin_date = '2021121000'
end_date = '2022031000'


if track_name != '':
    repout = os.path.join(repout, track_name)

dom = {"lonmin": 360-50, "latmin": 30, "lonmax": 359, "latmax": 70}


list_traj = {}
# read indef of experiment !
indef_file = open(repin + f'indef_{xp}.json')
indef = json.load(indef_file)
indef_file.close()

# origin = indef['origin']
origin = 'fc'

traj_path = f'/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/outputs/{xp}/track_{tag}_{algo}_{xp}_an_{begin_date}_to_{end_date}_v0.92.json'


ltraj = Read(traj_path)
if track_name != '':
    ltraj = Select(ltraj, {"name": track_name})[0]
else:
    # on prend la 1ere trajectoire
    ltraj = ltraj[0]

# Mise en forme des vecteurs lon, lat, etc
lonc = []
latc = []
timec = []
for i in range(len(ltraj.traj)):
    lonc.append(ltraj.traj[i].lonc)
    latc.append(ltraj.traj[i].latc)
    timec.append(ltraj.traj[i].time)
list_traj = {'time': timec, 'lonc': lonc, 'latc': latc}

target_index = 2
time_traj = timec[target_index + 1]
target_time = timec[target_index]


fc_term = int(time_traj[-2:])
# Réucpération du GRIB

list_xp = [xp, 'GV8T', 'GXCE']
grib_paths = []
vortex_times = []
if origin == 'an':
    for exp in list_xp:
        vortex_target_time = f'{target_time[:8]}T{target_time[-2:]}00A'
        vortex_dir = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/{exp}/{vortex_target_time}/forecast'
        grib_path = vortex_dir + \
            '/grid.arpege-forecast.glob025+0006:00.grib'
        grib_paths.append(grib_path)
        dt_hours = 6

elif origin == 'fc':
    for exp in list_xp:
        if exp == xp:
            vortex_target_time = f'{target_time[:8]}T{target_time[-2:]}00A'
            vortex_dir = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/{exp}/{vortex_target_time}/forecast'
            grib_path = vortex_dir + \
                '/grid.arpege-forecast.glob025+0006:00.grib'
        else:
            vortex_target_time = f'{target_time[:8]}T0000P'
            vortex_dir = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/{exp}/{vortex_target_time}/forecast'
            grib_path = vortex_dir + \
                '/grid.arpege-forecast.glob025+00%.2i:00.grib' % (fc_term)

        vortex_times.append(vortex_target_time)
        grib_paths.append(grib_path)
        dt_hours = 6


# %% Get epygram files !

# Epygram setup

# Paramètres de la commande requète Vortex
block = 'forecast'
remote = 'hendrix'

vapp = 'arpege'
vconf = '4dvarfr'
model = 'arpege'
role = 'historic'
instru = 'wivern'
nativefmt = 'grib'
kind = 'gridpoint'
geom = 'GLOB025'

for i_exp, exp in enumerate(list_xp):
    print(f'# Expe {exp}')
    if exp == xp:
        geom = 'GLOB025'
        term = 6
        cutoff = 'assim'
    else:
        geom = 'GLOB025'
        term = fc_term
        cutoff = 'production'

    grib = vtx.get_resources(experiment=exp,
                             namespace='vortex.multi.fr',
                             origin='hst',
                             kind=kind,
                             nativefmt=nativefmt,
                             date=vortex_times[i_exp],
                             term=term,
                             cutoff=cutoff,
                             geometry=geom,
                             model=vapp,
                             block=block,
                             local=grib_paths[i_exp],
                             vconf=vconf,
                             vapp=vapp,
                             getmode='epygram',
                             shouldfly=False,
                             uselocalcache=True)

# Vortex request :

rr6h = {}
crr6h = {}
u500 = {}
v500 = {}
u850 = {}
v850 = {}
u300 = {}
v300 = {}
hu500 = {}
hu850 = {}
T500 = {}
z500 = {}
av850 = {}
mslp = {}
for i, grib_path in enumerate(grib_paths):
    myfile = epygram.formats.resource(
        filename=grib_path, openmode='r', fmt="GRIB")
    if i == 0:
        print('# Get .grib geometry...')
        toread = 'name:Temperature, typeOfFirstFixedSurface: 100, level:1000'
        champref = myfile.readfield(toread)
        # refgeom=champref.geometry
        # nbptsx=int(resol_calc/champref.geometry.grid['X_resolution'].get('degrees'))
        # nbptsy=int(resol_calc/champref.geometry.grid['Y_resolution'].get('degrees'))
        # newgeom=champref.geometry.make_subsample_geometry(nbptsx,nbptsy,1)
        # (lons,lats)=newgeom.get_lonlat_grid()
        lon, lat = champref.geometry.get_lonlat_grid()

    if list_xp[i] == xp:
        print('# Get reference .grib geometry...')
        toread = 'name:Temperature, typeOfFirstFixedSurface: 103, level:100'
        champref = myfile.readfield(toread)
        # refgeom=champref.geometry
        # nbptsx=int(resol_calc/champref.geometry.grid['X_resolution'].get('degrees'))
        # nbptsy=int(resol_calc/champref.geometry.grid['Y_resolution'].get('degrees'))
        # newgeom=champref.geometry.make_subsample_geometry(nbptsx,nbptsy,1)
        # (lons,lats)=newgeom.get_lonlat_grid()
        lon_nr, lat_nr = champref.geometry.get_lonlat_grid()

    print('# OK\n\n')

    print('# Large scale rain rate 6h...')
    dict = {'discipline': 0, 'parameterCategory': 1,
            'parameterNumber': 77}
    TMP = myfile.readfield(dict)
    rr6h[list_xp[i]] = TMP.data

    print('# Convective rain rate 6h...')
    dict = {'discipline': 0, 'parameterCategory': 1,
            'parameterNumber': 76}
    TMP = myfile.readfield(dict)
    crr6h[list_xp[i]] = TMP.data

    print('# U wind 300hPa...')
    dict = {'shortName': 'u', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 2, 'level': 300, 'typeOfFirstFixedSurface': 100}
    TMP = myfile.readfield(dict)
    u300[list_xp[i]] = TMP.data

    print('# V wind 300hPa...')
    dict = {'shortName': 'v', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 3, 'level': 300, 'typeOfFirstFixedSurface': 100}
    TMP = myfile.readfield(dict)
    v300[list_xp[i]] = TMP.data

    print('# U wind 500hPa...')
    dict = {'shortName': 'u', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 2, 'level': 500, 'typeOfFirstFixedSurface': 103}
    TMP = myfile.readfield(dict)
    u500[list_xp[i]] = TMP.data

    print('# V wind 500hPa...')
    dict = {'shortName': 'v', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 3, 'level': 500, 'typeOfFirstFixedSurface': 103}
    TMP = myfile.readfield(dict)
    v500[list_xp[i]] = TMP.data

    print('# U wind 850hPa...')
    dict = {'shortName': 'u', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 2, 'level': 850, 'typeOfFirstFixedSurface': 100}
    TMP = myfile.readfield(dict)
    u850[list_xp[i]] = TMP.data

    print('# V wind 850hPa...')
    dict = {'shortName': 'v', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 3, 'level': 850, 'typeOfFirstFixedSurface': 100}
    TMP = myfile.readfield(dict)
    v850[list_xp[i]] = TMP.data

    print('# Relative humidity 500hPa...')
    dict = {'shortName': 'r', 'discipline': 0, 'parameterCategory': 1,
            'parameterNumber': 1, 'level': 500, 'typeOfFirstFixedSurface': 103}
    TMP = myfile.readfield(dict)
    hu500[list_xp[i]] = TMP.data

    print('# Relative humidity 850hPa...')
    dict = {'shortName': 'r', 'discipline': 0, 'parameterCategory': 1,
            'parameterNumber': 1, 'level': 850, 'typeOfFirstFixedSurface': 100}
    TMP = myfile.readfield(dict)
    hu850[list_xp[i]] = TMP.data

    print('# Temperature 500hPa...')
    dict = {'shortName': 't', 'discipline': 0, 'parameterCategory': 0,
            'parameterNumber': 0, 'level': 500, 'typeOfFirstFixedSurface': 103}
    TMP = myfile.readfield(dict)
    T500[list_xp[i]] = TMP.data

    print('# Z 500hPa...')
    dict = {'shortName': 'z', 'discipline': 0, 'parameterCategory': 3,
            'parameterNumber': 4, 'level': 500, 'typeOfFirstFixedSurface': 100}
    TMP = myfile.readfield(dict)
    z500[list_xp[i]] = TMP.data

    print('# MSLP...')
    dict = {'shortName': 'sp', 'discipline': 0, 'parameterCategory': 3,
            'parameterNumber': 0, 'level': 0, 'typeOfFirstFixedSurface': 1}
    TMP = myfile.readfield(dict)
    mslp[list_xp[i]] = TMP.data

    print('# Abs. vorticity 850hPa...')
    dict = {'shortName': 'absv', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 10, 'level': 850, 'typeOfFirstFixedSurface': 100}
    TMP = myfile.readfield(dict)
    av850[list_xp[i]] = TMP.data

# %  Data processing

# Considering only the domain :

# Index of borns

# lon = lon-180
i_min = np.where(lat[:, 0] >= dom['latmin'])[0][0]
i_max = np.where(lat[:, 0] <= dom['latmax'])[0][-1]+1

j_min = np.where(lon[0, :] >= dom['lonmin'])[0][0]
j_max = np.where(lon[0, :] <= dom['lonmax'])[0][-1]+1

lon2 = lon[i_min:i_max, j_min:j_max]
lat2 = lat[i_min:i_max, j_min:j_max]

# index_domain = np.where((lon_nr > dom['lonmin']) & (
#     lon_nr < dom['lonmax']) & (lat_nr > dom['latmin']) & (lat_nr < dom['latmax']))
# size1 = np.unique(index_domain[1]).shape[0]
# size2 = np.unique(index_domain[0]).shape[0]


# lon = lon_nr[index_domain].reshape(size1, size2)
# lat = lat_nr[index_domain].reshape(size1, size2)
for exp in list_xp:
    if rr6h[exp].shape != lon2.shape:
        rr6h[exp] = rr6h[exp][i_min:i_max, j_min:j_max]
        crr6h[exp] = crr6h[exp][i_min:i_max, j_min:j_max]
        u500[exp] = u500[exp][i_min:i_max, j_min:j_max]
        v500[exp] = v500[exp][i_min:i_max, j_min:j_max]
        u850[exp] = u850[exp][i_min:i_max, j_min:j_max]
        v850[exp] = v850[exp][i_min:i_max, j_min:j_max]

        T500[exp] = T500[exp][i_min:i_max, j_min:j_max]
        u300[exp] = u300[exp][i_min:i_max, j_min:j_max]
        v300[exp] = v300[exp][i_min:i_max, j_min:j_max]

        hu500[exp] = hu500[exp][i_min:i_max, j_min:j_max]
        hu850[exp] = hu850[exp][i_min:i_max, j_min:j_max]

        av850[exp] = av850[exp][i_min:i_max, j_min:j_max]

        z500[exp] = z500[exp][i_min:i_max, j_min:j_max]

        mslp[exp] = mslp[exp][i_min:i_max, j_min:j_max]
# %%
# Rain rate errors :
window_size = 8
total_rainrate = {}
total_rr_moy = {}
windmodule300 = {}
windmodule500 = {}
err_u = {}
err_v = {}
err_u300 = {}
err_v300 = {}
err_u850 = {}
err_v850 = {}
err_T500 = {}
err_hu500 = {}
err_hu850 = {}
err_wind300 = {}
err_wind300_v2 = {}
err_wind500 = {}
err_wind850 = {}
err_ang850 = {}
err_ang300 = {}
err_av850 = {}
err_z500 = {}
err_mslp = {}
for exp in list_xp:
    total_rainrate[exp] = crr6h[exp] + rr6h[exp]
    total_rr_moy[exp] = uniform_filter(
        total_rainrate[exp], size=window_size, mode='reflect')
    windmodule500[exp] = np.sqrt(u500[exp]**2 + v500[exp]**2)
    windmodule300[exp] = np.sqrt(u300[exp]**2 + v300[exp]**2)


a = crr6h[list_xp[1]] + rr6h[list_xp[1]] - crr6h[xp] - rr6h[xp]
b = crr6h[list_xp[0]] + rr6h[list_xp[0]] - crr6h[xp] - rr6h[xp]

c = np.abs(a) - np.abs(b)

diffrel_rainrate = np.divide(c, b, out=np.zeros_like(c), where=b != 0)

# Diffrel with filtered rain rate fields

a_filt = total_rr_moy[list_xp[1]] - total_rr_moy[xp]  # Error XP
b_filt = total_rr_moy[list_xp[0]] - total_rr_moy[xp]  # error Ctrl

c_filt = np.abs(a_filt) - np.abs(b_filt)

diffrel_rr_filt = np.divide(
    c_filt, np.abs(b_filt), out=np.zeros_like(c_filt), where=b_filt != 0)

for exp in list_xp[1:]:
    err_u[exp] = u500[exp] - u500[xp]
    err_v[exp] = v500[exp] - v500[xp]
    err_u300[exp] = u300[exp] - u300[xp]
    err_v300[exp] = v300[exp] - v300[xp]
    err_u850[exp] = u850[exp] - u850[xp]
    err_v850[exp] = v850[exp] - v850[xp]
    err_hu500[exp] = hu500[exp] - hu500[xp]
    err_hu850[exp] = hu850[exp] - hu850[xp]
    err_T500[exp] = T500[exp] - T500[xp]
    err_wind300[exp] = np.sqrt(
        u300[exp]**2 + v300[exp]**2) - np.sqrt(u300[xp]**2 + v300[xp]**2)
    err_wind850[exp] = np.sqrt(
        u850[exp]**2 + v850[exp]**2) - np.sqrt(u850[xp]**2 + v850[xp]**2)

    err_wind500[exp] = np.sqrt(
        u500[exp]**2 + v500[exp]**2) - np.sqrt(u500[xp]**2 + v500[xp]**2)

    err_ang850[exp] = np.arccos((u850[exp]*u850[xp] + v850[exp]*v850[xp]) /
                                (np.sqrt(u850[exp]**2 + v850[exp]**2)
                                 * np.sqrt(u850[xp]**2 + v850[xp]**2)))

    err_ang300[exp] = np.arccos((u300[exp]*u300[xp] + v300[exp]*v300[xp]) /
                                (np.sqrt(u300[exp]**2 + v300[exp]**2)
                                 * np.sqrt(u300[xp]**2 + v300[xp]**2)))
    err_av850[exp] = av850[exp] - av850[xp]

    err_wind300_v2[exp] = np.sqrt((err_u300[exp])**2 + (err_v300[exp])**2)

    err_z500[exp] = z500[exp] - z500[xp]
    err_mslp[exp] = mslp[exp] - mslp[xp]

    # %%
matplotlib.rc('font', size=7)

fig, axes = plt.subplots(nrows=1, ncols=3, subplot_kw={
    'projection': ccrs.PlateCarree()}, sharey=True)
ax = axes.flat

# gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], figure=fig)


# for i in range(3):
#     ax[i] = fig.add_subplot(gs[i])
# cax = fig.add_subplot(gs[-1])

cmap = plt.get_cmap('jet', 51)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['(a) Nature Run', '(b) Control', '(c) WIVERN err2']

vmin = 5
vmax = 70
levels = np.linspace(vmin, vmax, 50)

for i_exp, exp in enumerate(list_xp):

    ax[i_exp].set_title(labels[i_exp])
    # Plot traj :
    index_stop = list_traj['time'].index(time_traj)
    ax[i_exp].plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
                   marker='o', markersize=2, linewidth=1, color='k', label='System track')
    ax[i_exp].plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
                   marker='d', markersize=4, linewidth=1, color='k', label="_no_label")

    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)

    if exp == xp:
        r = 2
        im = ax[i_exp].contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                                windmodule300[exp][0:-1:r, 0:-1:r],
                                levels, cmap=cmap, extend='both')
        r = 8
        im2 = ax[i_exp].quiver(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                               u300[exp][0:-1:r, 0:-1:r], v300[exp][0:-1:r, 0:-1:r], pivot='middle', linewidth=0.2)

    else:
        r = 2
        im = ax[i_exp].contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                                windmodule300[exp][0:-1:r, 0:-1:r],
                                levels, cmap=cmap, extend='both')
        r = 8
        im2 = ax[i_exp].quiver(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                               u300[exp][0:-1:r, 0:-1:r], v300[exp][0:-1:r, 0:-1:r], pivot='middle', linewidth=0.2)

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

# Colorbar :
# divider = make_axes_locatable(ax[2])
# cax = divider.append_axes(
#     "right", size="4%", pad=0.05, axes_class=plt.Axes)
# fig.add_axes(cax)

cbar_ax = fig.add_axes([0.2, 0.29, 0.6, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal')
# cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 9))
cbar.set_label('Wind module (m/s)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)

ncol_lgd = 1
handles, plot_labels = ax[0].get_legend_handles_labels()
lgd = fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.2, 0.17),
                 fancybox=True, shadow=False, ncol=ncol_lgd)
for i in range(len(ax)):
    # set graphical parameters
    ax[i].coastlines(resolution='50m')
    gl = ax[i].gridlines(draw_labels=True, linestyle='--',
                         linewidth=0.5, color='grey')

    ax[i].set_extent([-45, 0, 35, 65])
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels

    if i > 0:
        gl.left_labels = False

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

fig.subplots_adjust(wspace=0.1)

# fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/winds300hPa_and_track{track_name}_all1x3_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, bbox_inches='tight', format='png', dpi=300)
# plt.show()

# %%
matplotlib.rc('font', size=11)


fig, axes = plt.subplots(nrows=1, ncols=2, subplot_kw={
    'projection': ccrs.PlateCarree()}, sharey=True, constrained_layout=True)
ax = axes.flat

vmin = 0
vmax = 7
cmap = plt.get_cmap('Reds', (vmax-vmin)*1+1)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['(a) Control', '(b) WIVERN']


levels = np.linspace(vmin, vmax, (vmax-vmin)*1+1)

for i_exp, exp in enumerate(list_xp[1:]):

    ax[i_exp].set_title(labels[i_exp])

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax[i_exp].plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
                   marker='o', markersize=3, linewidth=1.5, color='k', label='System track')
    ax[i_exp].plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
                   marker='d', markersize=5, linewidth=1.5, color='k', label="_no_label")

    r = 1
    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)

    im_diff = ax[i_exp].contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                                 err_wind300_v2[exp][0:-1:r, 0:-1:r],
                                 levels, cmap=cmap, extend='max')

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

    # Colorbar :
cbar_ax = fig.add_axes([0.2, 0.17, 0.6, 0.02])
cbar = fig.colorbar(im_diff, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, vmax-vmin+1))
cbar.set_label('Error on 300hPa wind speed (m/s)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis


ncol_lgd = 1
handles, plot_labels = ax[0].get_legend_handles_labels()
lgd = fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.15, 0.0),
                 fancybox=True, shadow=False, ncol=ncol_lgd)


for i in range(len(ax)):
    # set graphical parameters
    ax[i].coastlines(resolution='50m')
    gl = ax[i].gridlines(draw_labels=True, linestyle='--',
                         linewidth=0.5, color='grey')

    ax[i].set_extent([-45, 0, 35, 65])
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    if i > 0:
        gl.left_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

fig.subplots_adjust(wspace=0.1)

# fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/error_wind300v2_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
# plt.show()


# %%
matplotlib.rc('font', size=8)
fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                         'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)
ax = axes.flat

cmap = plt.get_cmap('seismic', 21)
cmap.set_over('w')
cmap.set_under('k')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -8
vmax = 8
levels = np.linspace(vmin, vmax, 50)

# ax.set_title(
#     f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa wind module [m/s].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
for i in range(len(ax)):
    ax[i].plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
               marker='o', markersize=2, linewidth=1, color='k', label='System track')
    ax[i].plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
               marker='d', markersize=3, linewidth=1, color='k', label="_no_label")

r = 1
# diffrel_u500 = np.divide(
#     (np.abs(err_u[list_xp[1]][0:-1:r, 0:-1:r]) -
#      np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r])),
#     np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     out=np.zeros_like(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     where=np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]) != 0)
ax[0].set_title('(a) 300hPa wind module')
im_diff = ax[0].contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                         (np.abs(err_wind300_v2[list_xp[-1]][0:-1:r, 0:-1:r]) -
                          np.abs(err_wind300_v2[list_xp[-2]][0:-1:r, 0:-1:r])),
                         levels, cmap=cmap, extend='both')

# Colorbar :
divider = make_axes_locatable(ax[0])
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('(m/s)')
# cbar.formatter.set_powerlimits((0, 0))


vmin = -0.0003
vmax = -vmin
levels = np.linspace(vmin, vmax, 50)
ax[1].set_title('(b) 850hPa absolute vorticity')
im_diff = ax[1].contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                         (np.abs(err_av850[list_xp[-1]][0:-1:r, 0:-1:r]) -
                          np.abs(err_av850[list_xp[-2]][0:-1:r, 0:-1:r])),
                         levels, cmap=cmap, extend='both')

# Colorbar :
divider = make_axes_locatable(ax[1])
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('($s^{-1}$)')
cbar.formatter.set_powerlimits((0, 0))
# cbar.formatter.set_powerlimits((0, 0))


vmin = -0.25
vmax = -vmin
levels = np.linspace(vmin, vmax, 50)
ax[2].set_title('(c) 500hPa Geopotential height')
im_diff = ax[2].contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                         (np.abs(err_z500[list_xp[-1]][0:-1:r, 0:-1:r]) -
                          np.abs(err_z500[list_xp[-2]][0:-1:r, 0:-1:r]))/1000,
                         levels, cmap=cmap, extend='both')

# Colorbar :
divider = make_axes_locatable(ax[2])
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('(km)')
# cbar.formatter.set_powerlimits((0, 0))


vmin = -5
vmax = -vmin
levels = np.linspace(vmin, vmax, 50)
ax[3].set_title('(d) Surface pressure')
im_diff = ax[3].contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                         (np.abs(err_mslp[list_xp[-1]][0:-1:r, 0:-1:r]) -
                          np.abs(err_mslp[list_xp[-2]][0:-1:r, 0:-1:r]))/100,
                         levels, cmap=cmap, extend='both')

# Colorbar :
divider = make_axes_locatable(ax[3])
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('(hPa)')
# cbar.formatter.set_powerlimits((0, 0))


fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
           fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
for i in range(len(ax)):
    ax[i].coastlines(resolution='50m')
    gl = ax[i].gridlines(draw_labels=True, linestyle='--',
                         linewidth=0.5, color='grey')

    # ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
    ax[i].set_extent([-45, 0, 35, 65])

    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

fig.subplots_adjust(wspace=0.2)

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_2x2_{track_name}_{list_xp[0]}vs{list_xp[1]}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.close()
# %%


matplotlib.rc('font', size=11)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 21)
cmap.set_over('w')
cmap.set_under('k')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -8
vmax = 8
levels = np.linspace(vmin, vmax, 50)

# ax.set_title(
#     f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa wind module [m/s].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=3, linewidth=2, color='k', label='System track')
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=7, linewidth=2, color='k', label="_no_label")

r = 1
# diffrel_u500 = np.divide(
#     (np.abs(err_u[list_xp[1]][0:-1:r, 0:-1:r]) -
#      np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r])),
#     np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     out=np.zeros_like(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     where=np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]) != 0)

im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      (np.abs(err_wind300[list_xp[-1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_wind300[list_xp[-2]][0:-1:r, 0:-1:r])),
                      levels, cmap=cmap, extend='both')

# im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
#                       diffrel_u500,
#                       levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on 300hPa\n wind module (m/s)')
# cbar.formatter.set_powerlimits((0, 0))


ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

# ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-45, 0, 35, 65])

# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
fig.subplots_adjust(wspace=0.1)
fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_wind300hPa_{track_name}_{list_xp[0]}vs{list_xp[1]}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.close()

# %%
matplotlib.rc('font', size=11)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 21)
cmap.set_over('w')
cmap.set_under('k')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -8
vmax = 8
levels = np.linspace(vmin, vmax, 50)

# ax.set_title(
#     f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa wind module [m/s].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=3, linewidth=2, color='k', label='System track')
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=7, linewidth=2, color='k', label="_no_label")

r = 1
# diffrel_u500 = np.divide(
#     (np.abs(err_u[list_xp[1]][0:-1:r, 0:-1:r]) -
#      np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r])),
#     np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     out=np.zeros_like(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     where=np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]) != 0)

im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      (np.abs(err_wind300_v2[list_xp[-1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_wind300_v2[list_xp[-2]][0:-1:r, 0:-1:r])),
                      levels, cmap=cmap, extend='both')

# im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
#                       diffrel_u500,
#                       levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on 300hPa\n wind module (m/s)')
# cbar.formatter.set_powerlimits((0, 0))


ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

# ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-45, 0, 35, 65])

# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
fig.subplots_adjust(wspace=0.1)
fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_wind300hPav2_{track_name}_{list_xp[0]}vs{list_xp[1]}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.close()


# %%
matplotlib.rc('font', size=11)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 21)
cmap.set_over('w')
cmap.set_under('k')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -0.0003
vmax = -vmin
levels = np.linspace(vmin, vmax, 50)

# ax.set_title(
#     f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa wind module [m/s].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=3, linewidth=2, color='k', label='System track')
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=7, linewidth=2, color='k', label="_no_label")

r = 1
# diffrel_u500 = np.divide(
#     (np.abs(err_u[list_xp[1]][0:-1:r, 0:-1:r]) -
#      np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r])),
#     np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     out=np.zeros_like(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     where=np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]) != 0)

im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      (np.abs(err_av850[list_xp[-1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_av850[list_xp[-2]][0:-1:r, 0:-1:r])),
                      levels, cmap=cmap, extend='both')

# im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
#                       diffrel_u500,
#                       levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on 850hPa absolute vorticity ($s^{-1}$)')
# cbar.formatter.set_powerlimits((0, 0))


ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

# ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-45, 0, 35, 65])

# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
fig.subplots_adjust(wspace=0.1)
fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_av850_{track_name}_{list_xp[0]}vs{list_xp[1]}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.close()
# %%
matplotlib.rc('font', size=11)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 21)
cmap.set_over('w')
cmap.set_under('k')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN']

vmin = -3
vmax = 3
levels = np.linspace(vmin, vmax, 50)

# ax.set_title(
#     f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa wind module [m/s].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=3, linewidth=2, color='k', label='System track')
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=7, linewidth=2, color='k', label="_no_label")

r = 1
# diffrel_u500 = np.divide(
#     (np.abs(err_u[list_xp[1]][0:-1:r, 0:-1:r]) -
#      np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r])),
#     np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     out=np.zeros_like(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     where=np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]) != 0)

im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      (np.abs(err_mslp[list_xp[-1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_mslp[list_xp[-2]][0:-1:r, 0:-1:r]))/100,
                      levels, cmap=cmap, extend='both')

# im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
#                       diffrel_u500,
#                       levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on mslp (hPa)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

# ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-45, 0, 35, 65])

# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
fig.subplots_adjust(wspace=0.1)
fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_mslp_{track_name}_{list_xp[0]}vs{list_xp[1]}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)

# %%
matplotlib.rc('font', size=11)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 21)
cmap.set_over('w')
cmap.set_under('k')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN']

vmin = -0.25
vmax = -vmin
levels = np.linspace(vmin, vmax, 50)

# ax.set_title(
#     f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa wind module [m/s].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=3, linewidth=2, color='k', label='System track')
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=7, linewidth=2, color='k', label="_no_label")

r = 1
# diffrel_u500 = np.divide(
#     (np.abs(err_u[list_xp[1]][0:-1:r, 0:-1:r]) -
#      np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r])),
#     np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     out=np.zeros_like(err_u[list_xp[0]][0:-1:r, 0:-1:r]),
#     where=np.abs(err_u[list_xp[0]][0:-1:r, 0:-1:r]) != 0)

im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      (np.abs(err_z500[list_xp[-1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_z500[list_xp[-2]][0:-1:r, 0:-1:r]))/1000,
                      levels, cmap=cmap, extend='both')

# im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
#                       diffrel_u500,
#                       levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on 500hPa\n geopotential height (km)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

# ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-45, 0, 35, 65])

# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
fig.subplots_adjust(wspace=0.1)
fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_z500hPa_{track_name}_{list_xp[0]}vs{list_xp[1]}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 101)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -8
vmax = -vmin
levels = np.linspace(vmin, vmax, 100)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 850hPa wind [m/s].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1

im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      np.abs(err_wind850[list_xp[-1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_wind850[list_xp[-2]][0:-1:r, 0:-1:r]),
                      levels, cmap=cmap, extend='both')


# val2plot = np.abs(err_u750[list_xp[1]][0:-1:r, 0:-1:r]) - \
#     np.abs(err_u750[list_xp[0]][0:-1:r, 0:-1:r])
# im_diff = ax.scatter(lon2[0:-1:r, 0:-1:r].flatten(), lat2[0:-1:r, 0:-1:r].flatten(),
#                      c=val2plot.flatten(), s=20, cmap=cmap,vmin=vmin,vmax=vmax, transform=projection)


# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on \n850hPa wind [m/s]')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-40, 0, 40, 60])
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_wind850hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)

# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 51)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -2
vmax = -vmin
levels = np.linspace(vmin, vmax, 50)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa temperature [K].\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1

im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      np.abs(err_T500[list_xp[1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_T500[list_xp[0]][0:-1:r, 0:-1:r]),
                      levels, cmap=cmap, extend='both')


# val2plot = np.abs(err_u750[list_xp[1]][0:-1:r, 0:-1:r]) - \
#     np.abs(err_u750[list_xp[0]][0:-1:r, 0:-1:r])
# im_diff = ax.scatter(lon2[0:-1:r, 0:-1:r].flatten(), lat2[0:-1:r, 0:-1:r].flatten(),
#                      c=val2plot.flatten(), s=20, cmap=cmap,vmin=vmin,vmax=vmax, transform=projection)


# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on \n500hPa temperature [K]')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-40, 0, 40, 60])
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_T500hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)


# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 101)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -40
vmax = -vmin
levels = np.linspace(vmin, vmax, 100)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 850hPa wind angle ($^\circ$).\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1
im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      (np.abs(err_ang850[list_xp[1]][0:-1:r, 0:-1:r]*180/np.pi) -
                      np.abs(err_ang850[list_xp[0]][0:-1:r, 0:-1:r]*180/np.pi)),
                      levels, cmap=cmap, extend='both')


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on \n850hPa wind angle($^\circ$)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-40, 0, 40, 60])
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_windangle850hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)

# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 101)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -40
vmax = -vmin
levels = np.linspace(vmin, vmax, 100)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h.Track {track_name}.\nRelative error on 250hPa wind angle ($^\circ$).')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1
im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      (np.abs(err_ang850[list_xp[1]][0:-1:r, 0:-1:r]*180/np.pi) -
                      np.abs(err_ang850[list_xp[0]][0:-1:r, 0:-1:r]*180/np.pi)),
                      levels, cmap=cmap, extend='both')


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on \n850hPa wind angle($^\circ$)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-40, 0, 40, 60])
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_windangle850hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)


# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 101)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']


# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 101)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -15
vmax = -vmin
levels = np.linspace(vmin, vmax, 100)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 850hPa wind error vector module (m/s).\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1

val2plot = np.sqrt(err_u850[list_xp[1]][0:-1:r, 0:-1:r]**2 + err_v850[list_xp[1]][0:-1:r, 0:-1:r]**2) - \
    np.sqrt(err_u850[list_xp[0]][0:-1:r, 0:-1:r]**2 +
            err_v850[list_xp[0]][0:-1:r, 0:-1:r]**2)


im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      val2plot,
                      levels, cmap=cmap, extend='both')


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on 850hPa\n wind error vector module(m/s)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-40, 0, 40, 60])
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_errwindvector850hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)


# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 101)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -15
vmax = -vmin
levels = np.linspace(vmin, vmax, 100)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h.Track {track_name}.\nRelative error on 250hPa wind error vector module (m/s).')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1

val2plot = np.sqrt(err_u250[list_xp[1]][0:-1:r, 0:-1:r]**2 + err_v250[list_xp[1]][0:-1:r, 0:-1:r]**2) - \
    np.sqrt(err_u250[list_xp[0]][0:-1:r, 0:-1:r]**2 +
            err_v250[list_xp[0]][0:-1:r, 0:-1:r]**2)


im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      val2plot,
                      levels, cmap=cmap, extend='both')


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on 250hPa\n wind error vector module(m/s)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-40, 0, 40, 60])
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_errwindvector250hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)

# %%
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 101)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -0.0005
vmax = -vmin
levels = np.linspace(vmin, vmax, 100)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 850hPa abs. vorticity ($^\circ$).\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1
im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      np.abs(err_av850[list_xp[1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_av850[list_xp[0]][0:-1:r, 0:-1:r]),
                      levels, cmap=cmap, extend='both')


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_diff, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on \n850hPa abs. vorticity ($^\circ$)')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
# for i_time, time in enumerate(list_traj['time'][:index_stop]):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
# print(f'# OK {time}')
# ax.text(list_traj['lonc'][i_time], list_traj['latc']
#         [i_time], time, fontsize=7)
# # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
# ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
#         markersize=10, marker='x', color='r')
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

# ncol_lgd = 3
# handles, plot_labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
#           fancybox=True, shadow=True, ncol=ncol_lgd)

# set graphical parameters
ax.coastlines(resolution='50m')
gl = ax.gridlines(draw_labels=True, linestyle='--',
                  linewidth=0.5, color='grey')

ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], dom['latmax']])
ax.set_extent([-40, 0, 40, 60])
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_av850hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
fig.savefig(save_path, format='png', dpi=300)
