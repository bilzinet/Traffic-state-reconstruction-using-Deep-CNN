# -*- coding: utf-8 -*-
"""
Analysing Vissim link segment results

Created on Mon Jun 18 12:16:34 2018

@author: btt1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()
corridor = "./10 hour/Abudhabi_Alain_Road"
data_folder = "Raw data"
study_link = 4
lane_id = 1
section_len = 810
time_start = 600
time_end = 36000
timeSpan = time_end-time_start
t_interval = 1
x_interval = 10
interval = 60

vr_filename = corridor + "_Vehicle Record (0-10).fzp"
vr_file_location = data_folder + "/" + vr_filename
vr_column_names = ['SimSec','TimeInNet','VehNo','LinkNo','LaneInd','Pos','LatPos','CoordFront','CoordRear','Length','Speed','Acc','TotDistTrav','FollowDist','Hdwy']
probe_data = pd.read_csv(os.path.join(cwd, vr_file_location), skiprows=30, sep=';', names=vr_column_names)
probe_data.SimSec = probe_data.SimSec.apply(lambda x: np.around(x))

time_probe = probe_data[probe_data['SimSec'] >= time_start]
time_probe = time_probe[time_probe['SimSec'] < time_end]
spacetime_probe = time_probe[time_probe['LinkNo'] == study_link]

def veh_trajectories(spacetime_probe, probe_percentage):

    vehs = spacetime_probe.VehNo.unique()
    num_probes = int(len(vehs)*probe_percentage/100)
    print('Total number of vehs: ', len(vehs))
    print('Number of sampled vehs: ', num_probes)
    sampled_vehs = np.random.choice(vehs, num_probes)
    return sampled_vehs

def extract_probe_veh_info(spacetime_probe, sampled_vehs, t_interval, probe_percentage):
    
    known_values = []
    for sim_sec in spacetime_probe.SimSec.unique():
        if sim_sec % t_interval == 0:
            sim_spacetime_probe = spacetime_probe[spacetime_probe['SimSec'] == sim_sec]
            if np.sum(np.isin(sampled_vehs, sim_spacetime_probe.VehNo.unique())) > 0:
                for probe_veh in sampled_vehs[np.isin(sampled_vehs, sim_spacetime_probe.VehNo.unique())]:
                    veh_pos = sim_spacetime_probe.loc[sim_spacetime_probe['VehNo'] == probe_veh, 'Pos'].iloc[0]
                    veh_speed = sim_spacetime_probe.loc[sim_spacetime_probe['VehNo'] == probe_veh, 'Speed'].iloc[0]
                    known_values.append((sim_sec-time_start, veh_pos, np.around(veh_speed,2)))
    probe_coords = np.array(known_values)
    return probe_coords


def plot_time_steps(interval, percentage, probe_coords, Img_Num_Lst):
    if percentage == 100:
        subfolder = 'Full/'
    else:
        subfolder = 'Sample/'
    xcount=time_start
    for i in Img_Num_Lst:
#        clean = np.logical_and(probe_coords[:,0]>xcount,probe_coords[:,0]<=xcount+60)    
        fig3 = plt.figure(figsize=(6,8))
        plt.scatter(probe_coords.SimSec,probe_coords.Pos,c=probe_coords.Speed, cmap='jet_r', vmin=0, vmax=60)
        plt.axis('off')
        plt.yticks([])
        plt.xticks([])
        plt.ylim(0,800)
        plt.xlim(xcount,xcount+60)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig3.savefig('./Data_14072019/{}/{}.jpg'.format(subfolder,i))
        plt.close(fig3)
        xcount+=interval
    return fig3

for i in range(3):
    spacetime_probe = spacetime_probe[spacetime_probe['LaneInd'] == i+1]
    spacetime_probe = spacetime_probe[spacetime_probe['Pos'] < section_len]
    Img_Num_Lst = np.linspace((i*timeSpan/interval)+1, (i+1)*(timeSpan/interval), timeSpan/interval).astype('int')
    
    # Part A: Complete information about vehicles
    full_vehs = veh_trajectories(spacetime_probe, probe_percentage=100)
    full_probe_coords = spacetime_probe[spacetime_probe['VehNo'].isin(full_vehs)]
#    full_probe_coords = extract_probe_veh_info(spacetime_probe, full_vehs, t_interval=1, probe_percentage=100)
    plot_time_steps(interval, 100, full_probe_coords, Img_Num_Lst)
    #np.save('./{}/full_vehs.npy'.format(data_folder), full_vehs)
    
    # Part B: Partial information about vehicles
    percent = 5
    sample_vehs = veh_trajectories(spacetime_probe, probe_percentage=percent)
    sample_probe_coords = spacetime_probe[spacetime_probe['VehNo'].isin(sample_vehs)]
#    sample_probe_coords = extract_probe_veh_info(spacetime_probe, sample_vehs, t_interval=1, probe_percentage=percent)
    plot_time_steps(interval, percent, sample_probe_coords, Img_Num_Lst)
    #np.save('./{}/sample_vehs.npy'.format(data_folder), sample_vehs)