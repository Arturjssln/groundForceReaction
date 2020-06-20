# Implementation of DFLOW ground reaction force prediction.
#
# author: Artur Jesslen <artur.jesslen@epfl.ch>
##
#!/usr/local/bin/python3
import os, sys
import argparse
import opensim
from utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Project 1 - Classification.')
parser.add_argument('-D', '--debug',
                    action='store_true', default=False,
                    help = 'Display debug messages (default: False)')
args = parser.parse_args()

#Initialize path
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(fileDir)

#initilization
model_file, ik_data, id_data, u, a, exp_data = import_from_storage(parentDir)

model = opensim.Model(model_file)
state = model.initSystem()
coordinate_set = model.updCoordinateSet()
pelvis = model.updBodySet().get('pelvis')
calcn_r = model.updBodySet().get('calcn_r')
calcn_l = model.updBodySet().get('calcn_l')
toes_r = model.updBodySet().get('toes_r')
toes_l = model.updBodySet().get('toes_l')
bodies_left = [calcn_l] * 9 + [toes_l] * 2
bodies_right = [calcn_r] * 9 + [toes_r] * 2
points_r = [opensim.Vec3(0, -0.005, 0),  # Heel
            opensim.Vec3(0.05, -0.005, -0.025),  # Heel
            opensim.Vec3(0.05, -0.005, 0.025),  # Heel
            opensim.Vec3(0.1, -0.005, 0.04),  # Midfoot
            opensim.Vec3(0.1, -0.005, -0.04),  # Midfoot
            opensim.Vec3(0.15, -0.005, 0.05),  # Midfoot
            opensim.Vec3(0.15, -0.005, -0.05),  # Midfoot
            opensim.Vec3(0.2, -0.005, -0.05),  # Forefoot
            opensim.Vec3(0.2, -0.005, 0.05),  # Forefoot
            opensim.Vec3(0.07, 0, -0.02),  # Toes
            opensim.Vec3(0.04, 0, 0.025)]  # Toes
points_l = [opensim.Vec3(0, -0.005, 0),  # Heel
            opensim.Vec3(0.05, -0.005, -0.025),  # Heel
            opensim.Vec3(0.05, -0.005, 0.025),  # Heel
            opensim.Vec3(0.1, -0.005, 0.04),  # Midfoot
            opensim.Vec3(0.1, -0.005, -0.04),  # Midfoot
            opensim.Vec3(0.15, -0.005, 0.05),  # Midfoot
            opensim.Vec3(0.15, -0.005, -0.05),  # Midfoot
            opensim.Vec3(0.2, -0.005, -0.05),  # Forefoot
            opensim.Vec3(0.2, -0.005, 0.05),  # Forefoot
            opensim.Vec3(0.07, 0, 0.02),  # Toes
            opensim.Vec3(0.04, 0, -0.025)]  # Toes
# Declare thresholds
thresholds = (0.05, 0.1)

assert(ik_data.shape == id_data.shape)
assert(ik_data.shape[0] == u.shape[0])

# Declare moment names
moments = ['pelvis_list_moment', 'pelvis_rotation_moment', 'pelvis_tilt_moment']
# Declare force names
force = ['pelvis_tx_force', 'pelvis_ty_force', 'pelvis_tz_force']

forces = []
left_forces = []
right_forces = []
times = []
cops = []
time_left_on_ground = []
time_right_on_ground = []
left_foot_position = []
right_foot_position = []
right_foot_usage = []
for i in range(ik_data.shape[0]):

    time = id_data.iloc[i]['time']
    times.append(time)

    # get residual moment and forces from inverse dynamics (expressed
    # in local frame of pelvis)
    M_p = [id_data.iloc[i][name] for name in moments]
    F_p = [id_data.iloc[i][name] for name in force]

    # update model pose
    for coordinate in coordinate_set:
        coordinate.setValue(state, ik_data.iloc[i][coordinate.getName()])
        coordinate.setSpeedValue(state, u.iloc[i][coordinate.getName()])

    model.realizePosition(state)
    model.realizeVelocity(state)

    # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Frame.html
    # get transformation of pelvis in ground frame
    X_PG = pelvis.getTransformInGround(state)
    R_PG = X_PG.R()
    r_P = X_PG.p()

    # do the calculations
    R_GP = simtk_matrix_to_np_array(R_PG).transpose()
    F_e = R_GP.dot(F_p)
    M_e = R_GP.dot(M_p)

    friction_coeff = 0.8
    assert(F_e[1] > friction_coeff*F_e[0] and F_e[1] > friction_coeff*F_e[2])

    # Determine which foot is on ground
    right_state = [ np.asarray([ np.asarray([body_part.findStationLocationInGround(state, position)[i] for i in range(3)]),
                    np.asarray([abs(body_part.findStationVelocityInGround(state, position)[i]) for i in range(3)])]) for body_part, position in zip(bodies_right, points_r)]
    left_state = [  np.asarray([  np.asarray([body_part.findStationLocationInGround(state, position)[i] for i in range(3)]),
                    np.asarray([abs(body_part.findStationVelocityInGround(state, position)[i]) for i in range(3)])]) for body_part, position in zip(bodies_left, points_l)]

    forces.append(F_e)
    left_on_ground = compute_force_2(left_state, thresholds, F_e, left_forces)
    right_on_ground = compute_force_2(right_state, thresholds, F_e, right_forces)

    if left_on_ground:
        time_left_on_ground.append(i)
    if right_on_ground:
        time_right_on_ground.append(i)

# Apply moving average on forces  
left_forces = moving_average(left_forces, width=10)
right_forces = moving_average(right_forces, width=10)

# Declare groundtruth force names
grdtruth_force = ['ground_force_vx', 'ground_force_vy', 'ground_force_vz', '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz']
grdtruth_moments = ['ground_torque_x', 'ground_torque_y', 'ground_torque_z', '1_ground_torque_x', '1_ground_torque_y', '1_ground_torque_z']
time_grdtruth = []
groundtruth = []
groundtruth_m = []
for i in range(exp_data.shape[0]):
    time = exp_data.iloc[i]['time']
    time_grdtruth.append(time)
    grdtruth = [exp_data.iloc[i][name] for name in grdtruth_force]
    grdtruth_m = [exp_data.iloc[i][name] for name in grdtruth_moments]
    groundtruth.append(grdtruth)
    groundtruth_m.append(grdtruth_m)

plot_results(time_grdtruth, groundtruth, groundtruth_m, times, time_left_on_ground, time_right_on_ground, forces, left_forces, right_forces)
