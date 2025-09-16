#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test case 3

Apply Ayrault algorithm on ARPEGE analyses during the period of Alex (Sept. & Oct. 2020)
Required inputs :
    - inputdef file ./inputs/indef_testcase3.json
    - algodef file : ./inputs/algo_testcase3.json
    - data files /cnrm/recyf/NO_SAVE/Data/users/plu/TRAJECT/ARPana/grid*.grib (as specified in inputdef file)

Expected outputs (in ./inputs/):
    - track file track_test_case3.json
    - figure file track_test_case3.png

"""

from traject import *

# Epygram environment
import os
import epygram
import json
epygram.init_env()
os.environ["ECCODES_SAMPLES_PATH"] = (
    "/home/common/epygram/ext/eccodes/share/eccodes/samples")


# Users inputs
cyclone_name = ''
track_name = ''
tag = 'dep_atln'
algo = 'VDG'
xp = 'GXCE'
date_dict = {'start': "2021121000", 'final': "2022031000", 'step': "24"}
termtraj = {'final': 96, 'step': 6}
prepare_files = False

if (xp == 'GTDJ') | (xp == 'GRX6'):
    algo = 'VDGfree'
    date_dict['step'] = '6'

algodef_name = f'namtraject_{tag}_{algo}.json'

basetime = date_dict['start']
endtime = date_dict['final']

# Reference track if needed
# usually the NatureRun, xp from where the first trajectory has been computed.
xp_ref = 'GTDJ'
# trackout = f'track_{tag}_{algo}_{xp}_{origin}_{basetime}_v'+str(traject_version)+'.json'
reftraj_path = f'./outputs/{xp_ref}/track_{tag}_VDGfree_{xp_ref}_an_{basetime}_to_{endtime}_v' + \
    str(traject_version)+'.json'


# %%
# Creation of different paths :

# Directory
repin = './inputs/'
repout = f'./outputs/{xp}/'
os.makedirs(repout, exist_ok=True)

indef_file = open(repin + f'indef_{xp}.json')
indef = json.load(indef_file)
indef_file.close()

# algodef_file = open(repin +f'algodef_{algo}.json')
algodef_file = open(repin + algodef_name)
algodef = json.load(algodef_file)
algodef_file.close()

os.makedirs(repout, exist_ok=True)
os.makedirs(indef['directory']+indef['experiment'], exist_ok=True)
os.makedirs(f"./tmp/{indef['experiment']}", exist_ok=True)

# Parameters defintion
origin = indef['origin']
cutoff = indef['cutoff']

# Fichiers de sortie :
trackout = f'track_{tag}_{algo}_{xp}_{origin}_{basetime}_to_{endtime}_v' + \
    str(traject_version)+'.json'
plotout = f'plottracks_{tag}_{algo}_{xp}_{origin}_{basetime}_to_{endtime}_v' + \
    str(traject_version)+'.png'


# %%
# ============================================================================== La ca commence
# Prepare input files
# Extraction from hendrix using vortex
# Extract data fields in single grib files in /cnrm/recyf/NO_SAVE/Data/users/plu/TRAJECT/ARPana/
# PrepareFiles(repin+f"algodef_{cyclone_name}.json",repin+f"indef_{xp}.json",\
#               {"vortex":"","extract":f"/cnrm/obs/data1/sasson/NO_SAVE/tmp/{indef['experiment']}/","filter":f"./tmp/{indef['experiment']}/","dirout":repout},\
#               {'start':"2021090900",'final':"2021090900",'step':"00"},\
#               termtraj={'final':24,'step':6}  )
if prepare_files:
    PrepareFiles(repin + algodef_name, repin+f"indef_{xp}.json",
                 {"vortex": "", 'dirout': repout},
                 termtraj=termtraj,
                 timetraj=date_dict)

if algo == 'VDG':
    ltraj_tmp = Read(reftraj_path)
    if cyclone_name != '':
        reftraj = Select(ltraj_tmp, {"name": cyclone_name})[0]
    else:
        reftraj = ltraj_tmp

    ltraj = track(repin + algodef_name, repout+f"indef_vortex.json",
                  date_dict,
                  termtraj=termtraj,
                  reftraj=reftraj,
                  outfile=repout+trackout,
                  plotfile=repout+plotout)

# only the filter for testing
# PrepareFiles(repin+"algo_testcase3.json",repout+"indef_xtract.json",{"filter":"./tmp/case3/","dirout":repout},{'start':"2020092500",'final':"2020100500",'step':"06"})

# a/ Compute track using the extracted grib files
else:
    ltraj = track(repin + algodef_name, repout+f"indef_vortex.json",
                  date_dict,
                  termtraj=termtraj,\
                  # ref_traj = repout + "/GRX6/track_julie_VDGfree_GRX6_an_v0.92",\
                  outfile=repout+trackout,\
                  plotfile=repout+plotout)

# b/ Compute track using the domain-filtered data
# ltraj=track(repin+"algo_testcase3.json",repout+"indef_filter.json",{'start':"2020092500",'final':"2020100500",'step':"06"},outfile=repout+'track_test_case3b_v'+str(traject_version)+'.json',plotfile=repout+'track_test_case3b_v'+str(traject_version)+'.png')

if False:
    # Read output track
    # ltraj1=Read(repout+'track_test_case3a.json')
    # for ivi in range(len(ltraj1)):
    #    print("Reading the track afterwards :")
    #    print(ltraj1[ivi].__dict__)
    ltraj2 = Read(
        repout+f'track_{cyclone_name}_{xp}_v'+str(traject_version)+'.json')
    for ivi in range(len(ltraj2)):
        print("Reading the track afterwards :")
        print(ltraj2[ivi].__dict__)
