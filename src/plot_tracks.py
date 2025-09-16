#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:48:24 2024

Script to generate several plots for a Cyclone trajectory found through
TRAJECT library.

@author: sasson
"""
import os
from traject import *
from visu_traject import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 200
matplotlib.use('Agg')

# %%

# Directories
repin = './inputs/'
# repout=f'./outputs/{exp}/'


# xp = ['GRX6','GTUW','GX9Y']
# labels = ['NR','Ctrl','WIVERN err2']

# xp = ['GTDJ','GV8T','GXCE']
# labels = ['NR','Ctrl','WIVERN err2']


cyclone_name = ''
tag = 'dep_atln_v2'
algo = 'VDGfree'
xp = ['GTDJ']
labels = ['NR']

# domtraj={"lonmin":20,"latmin":-40,"lonmax":180,"latmax":-2} #Indian ocean
domtraj = {"lonmin": -60.0, "latmin": 30.0,
           "lonmax": 30.0, "latmax": 70.0}  # ATLN


basetime = '2021121000'
endtime = '2022031000'

traj_path = {}
ltraj = []
indef = []
for i_exp, exp in enumerate(xp):

    indef_file = open(repin + f'indef_{exp}.json')
    indef.append(json.load(indef_file))
    indef_file.close()

    origin = indef[i_exp]['origin']

    # .json file for traj
    traj_path[exp] = f'/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/outputs/{exp}/track_{tag}_{algo}_{exp}_{origin}_{basetime}_to_{endtime}_v0.92.json'

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

diags = ["mslp_min_p", "rv850_max_o", "ff10m_max_s4"]


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

# %%

# Plot tracks and diag value diagnostics :
# title = "Tracks of cyclone {cyclone_name.upper()}."
# fig, ax1=plot_diag(ltraj,"mslp",'test_diag.png', dom={"lonmin":-70,"lonmax":-40,"latmin":10,"latmax":40},\
#            leg=labels)
# fig, ax = plt.subplots(figsize=(10, 10))
title = f"Tracks of cyclone {cyclone_name.upper()}."
# for i_exp,exp in enumerate(xp):
#    figout = figout + "{}"

if cyclone_name == '':
    fig, ax = plot_track(
        ltraj, f'trajs_{tag}_{xp[0]}_{basetime}_to_{endtime}.png', dom=domtraj, text=False)
else:
    fig, ax = plot_track(
        ltraj, f'trajs_{tag}_{cyclone_name}_{xp[0]}_{basetime}_to_{endtime}.png', dom=domtraj, text=True)
# fig.suptitle(title)


for diag in diags:
    filename = f"diag_{diag}_{cyclone_name.upper()}_{tag}_{xp[0]}_{basetime}_to_{endtime}.png"
    if cyclone_name == '':
        fig, ax = plot_diag(ltraj, diag, filename)
    else:
        fig, ax = plot_diag(ltraj, diag, filename)
    print(f'# {filename}')


# %%
# # traj_NR = Read(traj_path[xp[0]])


# list_traj = {}
# for i_exp,exp in enumerate(xp):
#     list_traj[exp] = {}

#     #read indef of experiment !
#     indef_file = open(repin +f'indef_{exp}.json')
#     indef.append(json.load(indef_file))
#     indef_file.close()

#     if indef[i_exp]['origin'] == 'an':

#         basetime = ltraj.traj[0].time
#         basetime_NR=basetime
#         list_traj[exp][basetime]={}
#             # Mise en forme des vecteurs lon, lat, etc
#         lonc=[];latc=[];timec=[]
#         for i in range(len(ltraj.traj)):
#             lonc.append(ltraj.traj[i].lonc)
#             latc.append(ltraj.traj[i].latc)
#             timec.append(ltraj.traj[i].time)

#         list_traj[exp][basetime]={'time':timec,'lonc':lonc,'latc':latc}
#     else:
#         for i_basetime, basetime in enumerate(list_basetime):

#             list_traj[exp][basetime]={}
#                 # Mise en forme des vecteurs lon, lat, etc
#             lonc=[];latc=[];timec=[]
#             for i in range(len(ltraj.traj)):
#                 lonc.append(ltraj.traj[i].lonc)
#                 latc.append(ltraj.traj[i].latc)
#                 timec.append(ltraj.traj[i].time)

#             list_traj[exp][basetime]={'time':timec,'lonc':lonc,'latc':latc}


# #%
# # Spaghetti plot

# fig,ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# lonmin=-65; lonmax=-40
# latmin=20; latmax=45

# colors = ['b','g','r']
# labels=['NR','Ctrl','WIVERN err2']

# #Plot NR :
# ax.plot(list_traj[xp[0]][basetime_NR]['lonc'], list_traj[xp[0]][basetime_NR]['latc'],\
#         marker='o', markersize= 2, linewidth=1, color ='blue', label=labels[0] )

# #Plot forecasts
# for i_exp, exp in enumerate(xp[1:]):
#     for basetime in list_basetime:
#         if basetime == list_basetime[0]:
#             ax.plot(list_traj[exp][basetime]['lonc'], list_traj[exp][basetime]['latc'],\
#                     linewidth=1, marker='d', markersize = 2, linestyle = ':',\
#                         color=colors[i_exp+1], alpha=0.8, label=labels[i_exp+1] )

# # Time plots
# for i_time, time in enumerate(list_traj[xp[0]][basetime_NR]['time']):
#     # # print(f'# OK {time}')
#     # basetime_NR=list_traj[xp[0]][basetime]['time']
#     if time[-2:] == '00':
#         print(f'# OK {time}')
#         ax.text(list_traj[xp[0]][basetime_NR]['lonc'][i_time], list_traj[xp[0]][basetime_NR]['latc'][i_time], time, fontsize=7 )
#         # ax.text(ltraj[0].traj[-1].lonc, ltraj[0].traj[-1].latc, ltraj[0].traj[-1].time)

# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)

# # ncol_lgd = 3
# # handles, plot_labels = ax.get_legend_handles_labels()
# # lgd=fig.legend(handles=handles, labels=plot_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
# #           fancybox=True, shadow=True, ncol=ncol_lgd)

# # set graphical parameters
# ax.coastlines(resolution = '50m')
# gl = ax.gridlines(draw_labels=True, linestyle = '--', linewidth = 0.5)

# ax.set_extent([dom['lonmin'],dom['lonmax'],dom['latmin'],dom['latmax']])
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER

# # fig.tight_layout()
# save_path = os.getcwd() + '/outputs/spaghetti_plot.png'
# fig.savefig(save_path, format='png', dpi=200)
# plt.show()
