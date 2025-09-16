#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:22:09 2024

@author: sasson
"""

import os
from traject import *
from visu_traject import *
import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rc('font', size=10)
# matplotlib.use('Agg')


# %%

# Directories
repin = './inputs/'
# repout=f'./outputs/{exp}/'

cyclone_name = ''
algo = 'VDGfree'
xp = 'GTDJ'
label = 'WIVERN err2'
begin_date = '2021121000'
end_date = '2022031000'

target_time = '2021121500'

dom = {"lonmin": -135, "lonmax": -110, "latmin": 5, "latmax": 20}


list_traj = {}
# read indef of experiment !
indef_file = open(repin + f'indef_{xp}.json')
indef = json.load(indef_file)
indef_file.close()

origin = indef['origin']
origin = 'an'

traj_path = f'/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/outputs/{xp}/track_{tag}_{algo}_{xp}_{origin}_{begin_date}_to_{end_date}_v0.92.json'


ltraj = Read(traj_path)
if cyclone_name != '':
    ltraj = Select(ltraj, {"name": cyclone_name})[0]
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


# Réucpération du GRIB
if origin == 'an':
    vortex_target_time = f'{target_time[:8]}T{target_time[-2:]}00A'
    vortex_dir = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/{xp}/{vortex_target_time}/forecast'
    grib_path = vortex_dir + '/grid.arpege-forecast.glob025+0006:00.grib'
    dt_hours = 0
elif origin == 'fc':
    vortex_target_time = f'{basetime[:8]}T{basetime[-2:]}00P'
    datetime_object = datetime.strptime(target_time, time_fmt)
    dt = datetime.strptime(target_time, time_fmt) - \
        datetime.strptime(basetime, time_fmt)
    dt_hours = dt.days * 24 + dt.seconds/3600
    vortex_dir = f'/cnrm/obs/data1/sasson/NO_SAVE/vortex/arpege/4dvarfr/{xp}/{vortex_target_time}/forecast'
    grib_path = vortex_dir + \
        '/grid.arpege-forecast.glob025+00%.2i:00.grib' % (dt_hours)


myfile = epygram.formats.resource(grib_path, 'r')

print('# Get .grib geometry...')
toread = 'name:Temperature, typeOfFirstFixedSurface: 100, level:1000'
champref = myfile.readfield(toread)
# refgeom=champref.geometry
# nbptsx=int(resol_calc/champref.geometry.grid['X_resolution'].get('degrees'))
# nbptsy=int(resol_calc/champref.geometry.grid['Y_resolution'].get('degrees'))
# newgeom=champref.geometry.make_subsample_geometry(nbptsx,nbptsy,1)
# (lons,lats)=newgeom.get_lonlat_grid()
lon, lat = champref.geometry.get_lonlat_grid()

print('# OK\n\n')

print('# Potential Vorticity...')
toread = f'shortName:absv, level:850, typeOfFirstFixedSurface:100'
TMP = myfile.readfield(toread)
pv850 = TMP.data

print('# MSLP...')
toread = 'shortName:prmsl'
TMP = myfile.readfield(toread)
mslp = TMP.data

print('# Medium Cloud Cover...')
toread = 'shortName:mcc'
TMP = myfile.readfield(toread)
mcc = TMP.data

print('# High Cloud Cover...')
toread = 'shortName:hcc'
TMP = myfile.readfield(toread)
hcc = TMP.data


# %%


fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})


cmap = plt.get_cmap('hot_r', 50)
cmap.set_over('k')
cmap.set_under('w')

colors = ['b', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = 0
vmax = 0.5*1e-3
levels = np.linspace(vmin, vmax, 50)

fig.suptitle(f'{xp}. {basetime} +{dt_hours}h. Absolute vorticity at 850hPa.')

# Plot NR :
index_stop = list_traj['time'].index(target_time) + 1
ax.plot(list_traj['lonc'][:index_stop], list_traj['latc'][:index_stop],
        marker='o', markersize=1, linewidth=0.5, color='blue', label=labels[0])

r = 4
# im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
im_vort = ax.contourf(lon[0:-1:r], lat[0:-1:r],
                      np.abs(pv850[0:-1:r]), levels, cmap=cmap, extend='max')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_vort, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Absolute vorticity at 850hPa [s$^{-1}$]')
cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
for i_time, time in enumerate(list_traj['time'][:index_stop]):
    # # print(f'# OK {time}')
    # basetime_NR=list_traj[xp[0]][basetime]['time']
    if time[-2:] == '00':
        print(f'# OK {time}')
        ax.text(list_traj['lonc'][i_time], list_traj['latc']
                [i_time], time, fontsize=7)
        # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
        ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
                markersize=1, marker='d', color='k')
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
    f'/outputs/fieldrv850_and_track_plot_{xp}_{target_time}.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.show()

# %%


fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# dom={"lonmin":85,"lonmax":105,"latmin":0,"latmax":10}

cmap = plt.get_cmap('hot_r', 50)
cmap.set_over('k')
cmap.set_under('w')

colors = ['b', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = 100500
vmax = 101500
levels = np.linspace(vmin, vmax, 50)

fig.suptitle(f'{xp}. {basetime} +{dt_hours}h. Mean sea level pressure.')


# Plot NR :
index_stop = list_traj['time'].index(target_time) + 1
ax.plot(list_traj['lonc'][:index_stop], list_traj['latc'][:index_stop],
        marker='o', markersize=1, linewidth=0.5, color='blue', label=labels[0])

r = 4
# im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
im_vort = ax.contourf(lon[0:-1:r], lat[0:-1:r],
                      mslp[0:-1:r], levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_vort, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Mean Sea Level Pressure [Pa]')
cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
for i_time, time in enumerate(list_traj['time'][:index_stop]):
    # # print(f'# OK {time}')
    # basetime_NR=list_traj[xp[0]][basetime]['time']
    if time[-2:] == '00':
        print(f'# OK {time}')
        ax.text(list_traj['lonc'][i_time], list_traj['latc']
                [i_time], time, fontsize=7)
        # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
        ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
                markersize=1, marker='d', color='k')
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
save_path = os.getcwd() + \
    f'/outputs/fieldmslp_and_track_plot_{xp}_{target_time}.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.show()


# %%


fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# dom={"lonmin":85,"lonmax":105,"latmin":0,"latmax":10}

cmap = plt.get_cmap('Grays', 50)
cmap.set_over('k')
cmap.set_under('w')

colors = ['b', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = 0
vmax = 100
levels = np.linspace(vmin, vmax, 50)

# Plot NR :
index_stop = list_traj['time'].index(target_time) + 1
ax.plot(list_traj['lonc'][:index_stop], list_traj['latc'][:index_stop],
        marker='o', markersize=1, linewidth=0.5, color='blue', label=labels[0])

r = 4
# im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
im_mcc = ax.contourf(lon[0:-1:r], lat[0:-1:r],
                     mcc[0:-1:r], levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_mcc, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('Medium Cloud Cover [%]')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
for i_time, time in enumerate(list_traj['time'][:index_stop]):
    # # print(f'# OK {time}')
    # basetime_NR=list_traj[xp[0]][basetime]['time']
    if time[-2:] == '00':
        print(f'# OK {time}')
        ax.text(list_traj['lonc'][i_time], list_traj['latc']
                [i_time], time, fontsize=7)
        # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
        ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
                markersize=1, marker='d', color='k')
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
    f'/outputs/field_mcc_and_track_plot_{xp}_{target_time}.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.show()


# %%


fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# dom={"lonmin":85,"lonmax":105,"latmin":0,"latmax":10}

cmap = plt.get_cmap('Grays', 50)
cmap.set_over('k')
cmap.set_under('w')

colors = ['b', 'g', 'r']
labels = ['NR', 'Ctrl', 'WIVERN err2']

vmin = 0
vmax = 100
levels = np.linspace(vmin, vmax, 50)

# Plot NR :
index_stop = list_traj['time'].index(target_time) + 1
ax.plot(list_traj['lonc'][:index_stop], list_traj['latc'][:index_stop],
        marker='o', markersize=1, linewidth=0.5, color='blue', label=labels[0])

r = 4
# im_vort = ax.scatter(lon[0:-1:r],lat[0:-1:r],c=np.abs(pv850[0:-1:r]), cmap = cmap, s=3, vmin=0, vmax=0.5*1e-3)
im_hcc = ax.contourf(lon[0:-1:r], lat[0:-1:r],
                     hcc[0:-1:r], levels, cmap=cmap, extend='both')

# ctr_mslp = ax.contour(lon, lat, mslp/100, levels=np.arange(950,1050,10) , cmap='Greys_r',linewidths=0.75)

# ax.clabel(ctr_mslp)


# Colorbar :
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im_mcc, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin, vmax, 11))
cbar.set_label('High Cloud Cover [%]')
# cbar.formatter.set_powerlimits((0, 0))

# # Colorbar :
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.60, axes_class=plt.Axes)
# fig.add_axes(cax)
# cbar = fig.colorbar(ctr_mslp, cax=cax, location='right', shrink = 30)
# cbar.set_label('mslp [hPa]')

# Time plots
for i_time, time in enumerate(list_traj['time'][:index_stop]):
    # # print(f'# OK {time}')
    # basetime_NR=list_traj[xp[0]][basetime]['time']
    if time[-2:] == '00':
        print(f'# OK {time}')
        ax.text(list_traj['lonc'][i_time], list_traj['latc']
                [i_time], time, fontsize=7)
        # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)
        ax.plot(list_traj['lonc'][i_time], list_traj['latc'][i_time],
                markersize=1, marker='d', color='k')
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
    f'/outputs/field_hcc_and_track_plot_{xp}_{target_time}.png'
fig.savefig(save_path, format='png', dpi=300)
# plt.show()
