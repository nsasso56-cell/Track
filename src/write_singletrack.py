#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:40:54 2025

Tool to select a single track in a track file, and write it in a separate .json file. 

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
# Name of the track to put in the new .json.
track_name = 'VDG-2021121000-10'

xp = 'GTDJ'
tag = 'dep_atln_v2'
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
print(f'\n# Read file {input_path}.\n')

ltraj2 = []

print(f'\n# Select track named {track_name}.\n')
traj = Select(ltraj, {"name": track_name})

# If not the Nature-Run, ie the reference track, we will match the names of the reference cyclones
# to create the new track file.
if xp != 'GTDJ':
    ltraj2 = []
    ref_track = f'{output_dir}/{xp}/track_{tag}_{add_tag}_VDGfree_GTDJ_an_{begin_date}_to_{end_date}_v0.92.json'

# Ecriture de ces nouvelles trajectoires :

output_file = f'{output_dir}/{xp}/track_{track_name}_{tag}_{algo}_{xp}_{cutoff}_{begin_date}_to_{end_date}_v0.92.json'
print(f'# Writing track in {output_file}')
# write_fc(ltraj2, output_file, algodef, indef)
Write(traj[0], output_file, mode='w')
