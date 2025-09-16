#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:40:54 2025

Tool to select tracks in a track file, based on specific threshold on diagnostics variables.

@author: sasson
"""


import matplotlib
from traject import *

import os
import epygram
import json
epygram.init_env()
os.environ["ECCODES_SAMPLES_PATH"] = (
    "/home/common/epygram/ext/eccodes/share/eccodes/samples")

matplotlib.rcParams["figure.dpi"] = 200
matplotlib.use('Agg')


output_dir = '/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/outputs'
input_dir = '/cnrm/obs/data1/sasson/WIVERN/OSSE/traj_cyclone/track/inputs'


# Input variables from user
# This tag will be added to the previous one to create a new filename.
add_tag = 'thr980hPa'
mslp_thr = 980

xp = 'GTDJ'
tag = 'dep_atln'
begin_date = '2021121000'
end_date = '2022031000'


# %%
algo = 'VDG'
cutoff = 'fc'
if xp == 'GTDJ':
    algo = 'VDGfree'
    cutoff = 'an'

algodef = f'{input_dir}/namtraject_{tag}_{algo}.json'
indef = f'{input_dir}/indef_{xp}.json'

input_path = f'{output_dir}/{xp}/track_{tag}_{algo}_{xp}_{cutoff}_{begin_date}_to_{end_date}_v0.92.json'

assert os.path.isfile(input_path), f"{input_path} doesn't exist."
assert os.path.isfile(algodef), f"{algodef} doesn't exist."
assert os.path.isfile(indef), f"{indef} doesn't exist."

# Opening of the .json :
ltraj = Read(input_path)

ltraj2 = []
for traj in ltraj:
    mini = traj.tmin("mslp_min_p")/100
    if mini < mslp_thr:  # If reference track,
        print(f'{traj.name}. Minimum mslp: {mini} hPa  --> Kept.')
        ltraj2.append(traj)

# If not the Nature-Run, ie the reference track, we will match the names of the reference cyclones
# to create the new track file.
if xp != 'GTDJ':
    ltraj2 = []
    ref_track = f'{output_dir}/{xp}/track_{tag}_{add_tag}_VDGfree_GTDJ_an_{begin_date}_to_{end_date}_v0.92.json'


# Ecriture de ces nouvelles trajectoires :
output_file = f'{output_dir}/{xp}/track_{tag}_{add_tag}_{algo}_{xp}_{cutoff}_{begin_date}_to_{end_date}_v0.92.json'
# write_fc(ltraj2, output_file, algodef, indef)
for traj in ltraj2:
    Write(traj, output_file)
