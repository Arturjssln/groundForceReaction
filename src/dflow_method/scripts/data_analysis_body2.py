#!/usr/local/bin/python3
import os, sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import opensim
from utils import *

#Initialize path
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(fileDir)

#initilization
model_file, ik_data, id_data, u, a, _ = import_from_storage(parentDir)

model = opensim.Model(model_file)
state = model.initSystem()
coordinate_set = model.updCoordinateSet()
pelvis = model.updBodySet().get('pelvis')
calcn_r = model.updBodySet().get('calcn_r')
calcn_l = model.updBodySet().get('calcn_l')
toes_r = model.updBodySet().get('toes_r')
toes_l = model.updBodySet().get('toes_l')
bodies_left = [calcn_l] * 9 + [toes_l]
bodies_right = [calcn_r] * 9 + [toes_r]

points_r = [opensim.Vec3(0, -0.005, 0), opensim.Vec3(0.1, -0.005, 0), 
            opensim.Vec3(0.2, -0.005, 0), opensim.Vec3(0.1, -0.005, 0.05),
            opensim.Vec3(0.2, -0.005, 0.05), opensim.Vec3(0.1, -0.005, -0.05),
            opensim.Vec3(0.05, -0.005, -0.025), opensim.Vec3(0.05, -0.005, 0.025),
            opensim.Vec3(0.05, -0.005, -0.025), opensim.Vec3(0.057, 0.0, -0.015)]
points_l = [opensim.Vec3(0, -0.005, 0), opensim.Vec3(0.1, -0.005, 0), 
            opensim.Vec3(0.2, -0.005, 0), opensim.Vec3(0.1, -0.005, 0.05),
            opensim.Vec3(0.2, -0.005, 0.05), opensim.Vec3(0.1, -0.005, -0.05),
            opensim.Vec3(0.05, -0.005, -0.025), opensim.Vec3(0.05, -0.005, 0.025),
            opensim.Vec3(0.05, -0.005, -0.025), opensim.Vec3(0.057, 0.0, 0.015)]

assert(ik_data.shape == id_data.shape)
assert(ik_data.shape[0] == u.shape[0])

# Declare moment names
moments = ['pelvis_list_moment', 'pelvis_rotation_moment', 'pelvis_tilt_moment']
# Declare force names
force = ['pelvis_tx_force', 'pelvis_ty_force', 'pelvis_tz_force']

with open('../data/data_body.csv', 'w') as csvfile:
    csvfile.truncate()
    writer = csv.writer(csvfile)
    for i in range(ik_data.shape[0]):

        time = id_data.iloc[i]['time']

        # update model pose
        for coordinate in coordinate_set:
            coordinate.setValue(state, ik_data.iloc[i][coordinate.getName()])
            coordinate.setSpeedValue(state, u.iloc[i][coordinate.getName()])

        model.realizePosition(state)
        model.realizeVelocity(state)
        for body_part, position in zip(bodies_right, points_r):
            writer.writerow([body_part.findStationLocationInGround(state, position)[i] for i in range(3)])
            writer.writerow([body_part.findStationVelocityInGround(state, position)[i] for i in range(3)])


with open('../data/data_body.csv') as file:
    data = csv.reader(file)
    data_array = []
    i = 0
    for row in data:
        data_array.append(np.asarray(row, dtype = np.float32))


    data_array = np.asarray(data_array)

    # Display data :
    fig1, axes1 = plt.subplots(1, 2)
    axes1[0].set_title("Position")
    axes1[1].set_title("Speed")
    for j in range(len(bodies_left)):
        axes1[0].plot(np.linspace(0., 2.5, int(len(data_array)/(2*len(bodies_left)))), data_array[(0+2*j)::(2*len(bodies_left)), 1], label="Point"+str(j+1))
        axes1[1].plot(np.linspace(0., 2.5, int(len(data_array)/(2*len(bodies_left)))), abs(data_array[(1+2*j)::(2*len(bodies_left)), 1]), label="Point"+str(j+1))
    axes1[0].axhline(y=0.05, xmin=0.0, xmax=1.0, color='b', linestyle='--')
    axes1[1].axhline(y=0.1, xmin=0.0, xmax=1.0, color='b', linestyle='--')
    axes1[0].legend()
    axes1[1].legend()
    plt.show()
