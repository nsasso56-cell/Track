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
import numpy as np
from scipy.ndimage import uniform_filter
from datetime import datetime, timedelta
import json

import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable


epygram.init_env()
matplotlib.rcParams["figure.dpi"] = 200
# matplotlib.rc('font', size=18)
# matplotlib.use('Agg')
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "pgf.rcfonts": False,
    "savefig.transparent": False,
    "patch.force_edgecolor": True
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
origin = 'an'

traj_path = f'/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/outputs/{xp}/track_{tag}_{algo}_{xp}_{origin}_{begin_date}_to_{end_date}_v0.92.json'


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

# Réucpération du GRIB

list_xp = ['GV8T', 'GWNM', xp]
grib_paths = []
if origin == 'an':
    for exp in list_xp:
        vortex_target_time = f'{target_time[:8]}T{target_time[-2:]}00A'
        vortex_dir = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/{exp}/{vortex_target_time}/forecast'
        grib_path = vortex_dir + '/grid.arpege-forecast.glob025+0006:00.grib'
        grib_paths.append(grib_path)
        dt_hours = 6
elif origin == 'fc':
    vortex_target_time = f'{basetime[:8]}T{basetime[-2:]}00P'
    datetime_object = datetime.strptime(target_time, time_fmt)
    dt = datetime.strptime(target_time, time_fmt) - \
        datetime.strptime(basetime, time_fmt)
    dt_hours = dt.days * 24 + dt.seconds/3600
    vortex_dir = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/{xp}/{vortex_target_time}/forecast'
    grib_path = vortex_dir + \
        '/grid.arpege-forecast.glob025+00%.2i:00.grib' % (dt_hours)


# %% Get epygram files !

# Epygram setup

# Paramètres de la commande requète Vortex
block = 'forecast'
remote = 'hendrix'
cutoff = 'assim'
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
    else:
        geom = 'GLOB025'
    grib = vtx.get_resources(experiment=exp,
                             namespace='vortex.multi.fr',
                             origin='hst',
                             kind=kind,
                             nativefmt=nativefmt,
                             date=vortex_target_time,
                             term=6,
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
u250 = {}
v250 = {}
hu500 = {}
hu850 = {}
T500 = {}
av850 = {}
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

    print('# U wind 250hPa...')
    dict = {'shortName': 'u', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 2, 'level': 250, 'typeOfFirstFixedSurface': 103}
    TMP = myfile.readfield(dict)
    u250[list_xp[i]] = TMP.data

    print('# V wind 250hPa...')
    dict = {'shortName': 'v', 'discipline': 0, 'parameterCategory': 2,
            'parameterNumber': 3, 'level': 250, 'typeOfFirstFixedSurface': 103}
    TMP = myfile.readfield(dict)
    v250[list_xp[i]] = TMP.data

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
        u250[exp] = u250[exp][i_min:i_max, j_min:j_max]
        v250[exp] = v250[exp][i_min:i_max, j_min:j_max]

        hu500[exp] = hu500[exp][i_min:i_max, j_min:j_max]
        hu850[exp] = hu850[exp][i_min:i_max, j_min:j_max]

        av850[exp] = av850[exp][i_min:i_max, j_min:j_max]

# Rain rate errors :
window_size = 8
total_rainrate = {}
total_rr_moy = {}
windmodule = {}
err_u = {}
err_v = {}
err_u250 = {}
err_v250 = {}
err_u850 = {}
err_v850 = {}
err_T500 = {}
err_hu500 = {}
err_hu850 = {}
err_wind250 = {}
err_wind500 = {}
err_wind850 = {}
err_ang850 = {}
err_ang250 = {}
err_av850 = {}
for exp in list_xp:
    total_rainrate[exp] = crr6h[exp] + rr6h[exp]
    total_rr_moy[exp] = uniform_filter(
        total_rainrate[exp], size=window_size, mode='reflect')
    windmodule[exp] = np.sqrt(u500[exp]**2 + v500[exp]**2)


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

for exp in list_xp[:-1]:
    err_u[exp] = u500[exp] - u500[xp]
    err_v[exp] = v500[exp] - v500[xp]
    err_u250[exp] = u250[exp] - u250[xp]
    err_v250[exp] = v250[exp] - v250[xp]
    err_u850[exp] = u850[exp] - u850[xp]
    err_v850[exp] = v850[exp] - v850[xp]
    err_hu500[exp] = hu500[exp] - hu500[xp]
    err_hu850[exp] = hu850[exp] - hu850[xp]
    err_T500[exp] = T500[exp] - T500[xp]
    err_wind250[exp] = np.sqrt(
        u250[exp]**2 + v250[exp]**2) - np.sqrt(u250[xp]**2 + v250[xp]**2)
    err_wind850[exp] = np.sqrt(
        u850[exp]**2 + v850[exp]**2) - np.sqrt(u850[xp]**2 + v850[xp]**2)

    err_wind500[exp] = np.sqrt(
        u500[exp]**2 + v500[exp]**2) - np.sqrt(u500[xp]**2 + v500[xp]**2)

    err_ang850[exp] = np.arccos((u850[exp]*u850[xp] + v850[exp]*v850[xp]) /
                                (np.sqrt(u850[exp]**2 + v850[exp]**2)
                                 * np.sqrt(u850[xp]**2 + v850[xp]**2)))

    err_ang250[exp] = np.arccos((u250[exp]*u250[xp] + v250[exp]*v250[xp]) /
                                (np.sqrt(u250[exp]**2 + v250[exp]**2)
                                 * np.sqrt(u250[xp]**2 + v250[xp]**2)))
    err_av850[exp] = av850[exp] - av850[xp]

    # %%

for exp in list_xp:

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.get_cmap('jet', 50)
    cmap.set_over('k')
    cmap.set_under('w')

    colors = ['r', 'g', 'r']
    labels = ['NR', 'Ctrl', 'WIVERN err2']

    vmin = 1
    vmax = 10
    levels = np.linspace(vmin, vmax, 50)

    fig.suptitle(
        f'{exp}. {target_time} +{dt_hours}h. Total rain rate [mm].')

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
            marker='o', markersize=2, linewidth=1, color='r', label=labels[0])
    ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
            marker='d', markersize=5, linewidth=1, color='r', label="_no_label")

    r = 1
    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
    if exp == xp:
        im_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                              rr6h[exp][0:-1:r, 0:-1:r] + crr6h[exp][0:-1:r, 0:-1:r], levels, cmap=cmap, extend='max')
    else:

        im_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                              rr6h[exp][0:-1:r, 0:-1:r] + crr6h[exp][0:-1:r, 0:-1:r], levels, cmap=cmap, extend='max')

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

    # Colorbar :
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="2%", pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(im_rain, cax=cax)
    cbar.ax.tick_params()
    cbar.set_ticks(np.linspace(vmin, vmax, 10))
    cbar.set_label('Total rain rate [mm]')
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
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
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    fig.tight_layout()
    save_path = os.getcwd() + \
        f'/outputs/rainrate_and_track_{track_name}_{exp}_{target_time}.png'
    fig.savefig(save_path, format='png', dpi=300)
    # plt.show()

# %%

for exp in list_xp:

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.get_cmap('jet', 50)
    cmap.set_over('k')
    cmap.set_under('w')

    colors = ['r', 'g', 'r']
    labels = ['NR', 'Ctrl', 'WIVERN err2']

    vmin = 1
    vmax = 10
    levels = np.linspace(vmin, vmax, 50)

    fig.suptitle(
        f'{exp}. {target_time} +{dt_hours}h. Filtered total rain rate [mm].')

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
            marker='o', markersize=2, linewidth=1, color='r', label=labels[0])
    ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
            marker='d', markersize=5, linewidth=1, color='r', label="_no_label")

    r = 4
    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
    if exp == xp:
        im_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                              total_rr_moy[exp][0:-1:r, 0:-1:r], levels, cmap=cmap, extend='max')
    else:

        im_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                              total_rr_moy[exp][0:-1:r, 0:-1:r], levels, cmap=cmap, extend='max')

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

    # Colorbar :
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="2%", pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(im_rain, cax=cax)
    cbar.ax.tick_params()
    cbar.set_ticks(np.linspace(vmin, vmax, 10))
    cbar.set_label('Filt. total rain rate [mm]')
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
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
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # fig.tight_layout()
    # save_path = os.getcwd() + \
    #     f'/outputs/filtrainrate_and_track_{track_name}_{exp}_{target_time}.png'
    # fig.savefig(save_path, format='png', dpi=300)
    # # plt.show()

# %%
for exp in list_xp:

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.get_cmap('jet', 50)
    cmap.set_over('k')
    cmap.set_under('w')

    colors = ['r', 'g', 'r']
    labels = ['NR', 'Ctrl', 'WIVERN err2']

    vmin = 0.1
    vmax = 10
    levels = np.linspace(vmin, vmax, 50)

    fig.suptitle(
        f'{exp}. {target_time} +{dt_hours}h. Convective rain rate.')

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
            marker='o', markersize=2, linewidth=1, color='r', label=labels[0])
    ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
            marker='d', markersize=5, linewidth=1, color='r', label="_no_label")

    r = 4
    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
    if exp == xp:
        im_rain = ax.contourf(lon2[0:-1:r], lat2[0:-1:r],
                              np.abs(crr6h[exp][0:-1:r]), levels, cmap=cmap, extend='max')
    else:

        im_rain = ax.contourf(lon2[0:-1:r], lat2[0:-1:r],
                              np.abs(crr6h[exp][0:-1:r]), levels, cmap=cmap, extend='max')

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

    # Colorbar :
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="2%", pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(im_rain, cax=cax)
    cbar.ax.tick_params()
    cbar.set_ticks(np.linspace(vmin, vmax, 11))
    cbar.set_label('Convective rain rate [mm]')
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
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
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # fig.tight_layout()
    # save_path = os.getcwd() + \
    #     f'/outputs/convectiverainrate_and_track_plot_{exp}_{target_time}+{dt_hours}h.png'
    # fig.savefig(save_path, format='png', dpi=300)
    # plt.show()

    # %%
for exp in list_xp:

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.get_cmap('jet', 50)
    cmap.set_over('k')
    cmap.set_under('w')

    colors = ['r', 'g', 'r']
    labels = ['NR', 'Ctrl', 'WIVERN err2']

    vmin = -40
    vmax = 40
    levels = np.linspace(vmin, vmax, 50)

    ax.set_title(
        f'{exp}. {target_time} +{dt_hours}h. U wind component 500hPa.\n{track_name}')

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
            marker='o', markersize=2, linewidth=1, color='r', label=labels[0])
    ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
            marker='d', markersize=5, linewidth=1, color='r', label="_no_label")

    r = 1
    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
    if exp == xp:
        im_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                              u500[exp][0:-1:r, 0:-1:r], levels, cmap=cmap, extend='max')
    else:
        im_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                              u500[exp][0:-1:r, 0:-1:r], levels, cmap=cmap, extend='max')

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

    # Colorbar :
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="2%", pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(im_rain, cax=cax)
    cbar.ax.tick_params()
    cbar.set_ticks(np.linspace(vmin, vmax, 11))
    cbar.set_label('U wind 500hPa [m/s]')
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
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
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # fig.tight_layout()
    # save_path = os.getcwd() + \
    #     f'/outputs/uwind500hPa_and_track_plot_{exp}_{target_time}+{dt_hours}h.png'
    # fig.savefig(save_path, format='png', dpi=300)
    # plt.show()

    # %%
for exp in list_xp:

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.get_cmap('jet', 51)
    cmap.set_over('k')
    cmap.set_under('w')

    colors = ['r', 'g', 'r']
    labels = ['NR', 'Ctrl', 'WIVERN err2']

    vmin = 5
    vmax = 40
    levels = np.linspace(vmin, vmax, 50)

    # ax.set_title(
    #     f'{exp}. {target_time} +{dt_hours}h. Winds and wind module at 500hPa.\n{track_name}')

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
            marker='o', markersize=3, linewidth=2, color='k', label='System track')
    ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
            marker='d', markersize=6, linewidth=2, color='k', label="_no_label")

    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)

    if exp == xp:
        r = 2
        im = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                         windmodule[exp][0:-1:r, 0:-1:r],
                         levels, cmap=cmap, extend='both')
        r = 8
        im2 = ax.quiver(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                        u500[exp][0:-1:r, 0:-1:r], v500[exp][0:-1:r, 0:-1:r], pivot='middle',)

    else:
        r = 2
        im = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                         windmodule[exp][0:-1:r, 0:-1:r],
                         levels, cmap=cmap, extend='both')
        r = 8
        im2 = ax.quiver(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                        u500[exp][0:-1:r, 0:-1:r], v500[exp][0:-1:r, 0:-1:r], pivot='middle',)

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

    # Colorbar :
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="2%", pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params()
    cbar.set_ticks(np.linspace(vmin, vmax, 8))
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    # ncol_lgd = 3
    # handles, plot_labels = ax.get_legend_handles_labels()
    # lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
    #           fancybox=True, shadow=True, ncol=ncol_lgd)

    # set graphical parameters
    ax.coastlines(resolution='50m')
    gl = ax.gridlines(draw_labels=True, linestyle='--',
                      linewidth=0.5, color='grey')

    ax.set_extent([dom['lonmin'], dom['lonmax'], dom['latmin'], 65])
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    fig.tight_layout()
    save_path = os.getcwd() + \
        f'/outputs/winds500hPa_and_track{track_name}_{exp}_{target_time}+{dt_hours}h.pgf'
    fig.savefig(save_path, bbox_inches='tight', format='pgf', dpi=50)
    # plt.show()


# %%
for exp in list_xp[:-1]:

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.get_cmap('seismic', 50)
    cmap.set_over('k')
    cmap.set_under('w')

    colors = ['r', 'g', 'r']
    labels = ['NR', 'Ctrl', 'WIVERN err2']

    vmin = -4
    vmax = 4
    levels = np.linspace(vmin, vmax, 50)

    # ax.set_title(
    #     f'{exp} - {xp}. {target_time} +{dt_hours}h. Error on 6h precipitation rate.\n{track_name}')

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
            marker='o', markersize=2, linewidth=1, color='r', label='System track')
    ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
            marker='d', markersize=5, linewidth=1, color='r', label="_no_label")

    r = 1
    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)

    diff_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                            (crr6h[exp][0:-1:r, 0:-1:r] + rr6h[exp][0:-1:r, 0:-1:r]) -
                            (crr6h[xp][0:-1:r, 0:-1:r] +
                             rr6h[xp][0:-1:r, 0:-1:r]),
                            levels, cmap=cmap, extend='both')

    # ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

    # ax.clabel(ctr_mslp)

    # Colorbar :
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="2%", pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(diff_rain, cax=cax)
    cbar.ax.tick_params()
    cbar.set_ticks(np.linspace(vmin, vmax, 11))
    cbar.set_label('Error on precipitation rate [mm]')
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
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    fig.tight_layout()
    save_path = os.getcwd() + \
        f'/outputs/error_rainrate_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
    fig.savefig(save_path, format='png', dpi=300)
    # plt.show()


# %%


# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 50)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -4
vmax = 4
levels = np.linspace(vmin, vmax, 50)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 6h precipitation rate [mm].\n{track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=2, linewidth=1, color='r', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=5, linewidth=1, color='r', label="_no_label")

r = 1

diff_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                        c[0:-1:r, 0:-1:r],
                        levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(diff_rain, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Relative error on \nprecipitation rate [mm]')
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
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_rainrate_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.show()


# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

cmap = plt.get_cmap('seismic', 50)
cmap.set_over('k')
cmap.set_under('w')

colors = ['r', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = -1
vmax = 1
levels = np.linspace(vmin, vmax, 50)

ax.set_title(
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Filt. relative error on 6h precipitation rate [mm].\nWindow size = {0.25*16}$^\circ$. {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=2, linewidth=1, color='m', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=5, linewidth=1, color='m', label="_no_label")

r = 1

diff_rain = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                        c_filt[0:-1:r, 0:-1:r],
                        levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)

# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(diff_rain, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Filt. relative error on \nprecipitation rate [mm]')
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
# add these before plotting
gl.top_labels = False   # suppress top labels
gl.right_labels = False  # suppress right labels
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/filt_relativeerror_rainrate_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)


# %%
for exp in list_xp[:-1]:

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.get_cmap('seismic', 50)
    cmap.set_over('k')
    cmap.set_under('w')

    colors = ['r', 'g', 'r']
    labels = ['NR', 'Ctrl', 'WIVERN err2']

    vmin = -8
    vmax = 8
    levels = np.linspace(vmin, vmax, 50)

    # ax.set_title(
    #     f'{exp} - {xp}. {target_time} +{dt_hours}h. Error on 500hPa wind [m/s].\n{track_name}')

    # Plot NR :
    index_stop = list_traj['time'].index(time_traj)
    ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
            marker='o', markersize=3, linewidth=2, color='k', label='System track')
    ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
            marker='d', markersize=5, linewidth=2, color='k', label="_no_label")

    r = 1
    # im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)

    im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                          err_wind500[exp][0:-1:r, 0:-1:r],
                          levels, cmap=cmap, extend='both')

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
    cbar.set_label('Error on 500hPa\n wind speed [m/s]')
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
    # add these before plotting
    gl.top_labels = False   # suppress top labels
    gl.right_labels = False  # suppress right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    fig.tight_layout()
    save_path = os.getcwd() + \
        f'/outputs/error_wind500hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
    fig.savefig(save_path, format='png', dpi=300)
    # plt.show()


# %%
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
                      (np.abs(err_wind500[list_xp[1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_wind500[list_xp[0]][0:-1:r, 0:-1:r])),
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
cbar.set_label('Relative error on 500hPa\n wind module [m/s]')
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

fig.tight_layout()
save_path = os.getcwd() + \
    f'/outputs/relativeerror_wind500hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.close()

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
                      np.abs(err_wind850[list_xp[1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_wind850[list_xp[0]][0:-1:r, 0:-1:r]),
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
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 500hPa relative humidity (\%).\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1
im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      np.abs(err_hu500[list_xp[1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_hu500[list_xp[0]][0:-1:r, 0:-1:r]),
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
cbar.set_label('Relative error on \nrelative humidity (\%)')
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
    f'/outputs/relativeerror_hu500hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
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
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 850hPa relative humidity (\%).\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1
im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      np.abs(err_hu850[list_xp[1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_hu850[list_xp[0]][0:-1:r, 0:-1:r]),
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
cbar.set_label('Relative error on \n850hPa relative humidity (\%)')
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
    f'/outputs/relativeerror_hu850hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
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
    f'{list_xp[1]} - {list_xp[0]}. Ref {xp}. {target_time} +{dt_hours}h. Relative error on 250hPa wind (m/s).\n {track_name}')

# Plot NR :
index_stop = list_traj['time'].index(time_traj)
ax.plot(list_traj['lonc'][:index_stop+1], list_traj['latc'][:index_stop+1],
        marker='o', markersize=4, linewidth=2, color='k', label=labels[0])
ax.plot(list_traj['lonc'][index_stop], list_traj['latc'][index_stop],
        marker='d', markersize=10, linewidth=2, color='k', label="_no_label")

r = 1
im_diff = ax.contourf(lon2[0:-1:r, 0:-1:r], lat2[0:-1:r, 0:-1:r],
                      np.abs(err_wind250[list_xp[1]][0:-1:r, 0:-1:r]) -
                      np.abs(err_wind250[list_xp[0]][0:-1:r, 0:-1:r]),
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
cbar.set_label('Relative error on \n250hPa wind (m/s)')
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
    f'/outputs/relativeerror_windangle250hPa_{track_name}_{exp}_ref{xp}_{target_time}+{dt_hours}h.png'
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
