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
write_to_storage('../results/prediction.mot', 
                labels=['time', 'ground_force_vx','ground_force_vy','ground_force_vz', 
                    'ground_force_px', 'ground_force_py', 'ground_force_pz', 
                    '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz',
                    '1_ground_force_px', '1_ground_force_py', '1_ground_force_pz',
                    'ground_torque_x', 'ground_torque_y', 'ground_torque_z', 
                    '1_ground_torque_x', '1_ground_torque_y', '1_ground_torque_z'],
                prepare=True)

model = opensim.Model(model_file)
weight = 72.6 #kg
height = 1.8 #m
state = model.initSystem()
coordinate_set = model.updCoordinateSet()
pelvis = model.updBodySet().get('pelvis')
calcn_r = model.updBodySet().get('calcn_r')
calcn_l = model.updBodySet().get('calcn_l')
toes_r = model.updBodySet().get('toes_r')
toes_l = model.updBodySet().get('toes_l')
bodies_left = [calcn_l] * 6 + [toes_l]
bodies_right = [calcn_r] * 6 + [toes_r]

points_r = [opensim.Vec3(0, -0.005, 0),  # Heel
            opensim.Vec3(0.05, -0.005, -0.025), # Heel
            opensim.Vec3(0.05, -0.005, -0.025), # Heel
            opensim.Vec3(0.2, -0.005, -0.05),  # Forefoot
            opensim.Vec3(0.2, -0.005, 0), # Forefoot
            opensim.Vec3(0.2, -0.005, 0.05), # Forefoot
            opensim.Vec3(0.057, 0.0, -0.015)] # Toes
points_l = [opensim.Vec3(0, -0.005, 0),  # Heel
            opensim.Vec3(0.05, -0.005, -0.025), # Heel
            opensim.Vec3(0.05, -0.005, 0.025), # Heel
            opensim.Vec3(0.2, -0.005, -0.05),  # Forefoot
            opensim.Vec3(0.2, -0.005, 0),  # Forefoot
            opensim.Vec3(0.2, -0.005, 0.05), # Forefoot
            opensim.Vec3(0.057, 0.0, 0.015)] # Toes

cop_bodies_left = [calcn_l] * 9 + [toes_l] * 2 
cop_bodies_right = [calcn_r] * 9 + [toes_r] * 2
cop_points_r = [opensim.Vec3(0, -0.005, 0),  # Heel
                opensim.Vec3(0.05, -0.005, -0.025),  # Heel
                opensim.Vec3(0.05, -0.005, 0.025),  # Heel
                opensim.Vec3(0.1, -0.005, 0),  # Midfoot
                opensim.Vec3(0.1, -0.005, 0.04),  # Midfoot
                opensim.Vec3(0.1, -0.005, -0.04),  # Midfoot
                opensim.Vec3(0.2, -0.005, -0.05),  # Forefoot
                opensim.Vec3(0.2, -0.005, 0),  # Forefoot
                opensim.Vec3(0.2, -0.005, 0.05),  # Forefoot
                opensim.Vec3(0.07, 0, -0.02), # Toes
                opensim.Vec3(0.04, 0, 0.025)]  # Toes
cop_points_l = [opensim.Vec3(0, -0.005, 0),  # Heel
                opensim.Vec3(0.05, -0.005, -0.025),  # Heel
                opensim.Vec3(0.05, -0.005, 0.025),  # Heel
                opensim.Vec3(0.1, -0.005, 0),  # Midfoot
                opensim.Vec3(0.1, -0.005, 0.04),  # Midfoot
                opensim.Vec3(0.1, -0.005, -0.04),  # Midfoot
                opensim.Vec3(0.2, -0.005, -0.05),  # Forefoot
                opensim.Vec3(0.2, -0.005, 0),  # Forefoot
                opensim.Vec3(0.2, -0.005, 0.05),  # Forefoot
                opensim.Vec3(0.07, 0, 0.02),  # Toes
                opensim.Vec3(0.04, 0, -0.025)]  # Toes


assert(ik_data.shape == id_data.shape)
assert(ik_data.shape[0] == u.shape[0])

# Declare moment names
moment = ['pelvis_list_moment', 'pelvis_rotation_moment', 'pelvis_tilt_moment']
# Declare force names
force = ['pelvis_tx_force', 'pelvis_ty_force', 'pelvis_tz_force']

forces = []
left_forces = []
right_forces = []
moments = []
left_moments = []
right_moments = []
times = []
heel_r = []
toes_r = []
heel_l = []
toes_l = []
time_left_on_ground = []
time_right_on_ground = []
cops_l = []
cops_r = []
right_foot_usage = []
left_on_ground = True
right_on_ground = True
for i in range(ik_data.shape[0]):

    time = id_data.iloc[i]['time']
    times.append(time)

    # get residual moment and forces from inverse dynamics (expressed
    # in local frame of pelvis)
    M_p = [id_data.iloc[i][name] for name in moment]
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
    right_state = np.array([np.asarray([np.asarray([body_part.findStationLocationInGround(state, position)[i] for i in range(3)]),
                    np.asarray([body_part.findStationVelocityInGround(state, position)[i] for i in range(3)])]) for body_part, position in zip(bodies_right, points_r)])
    left_state = np.array([  np.asarray([np.asarray([body_part.findStationLocationInGround(state, position)[i] for i in range(3)]),
                    np.asarray([body_part.findStationVelocityInGround(state, position)[i] for i in range(3)])]) for body_part, position in zip(bodies_left, points_l)])

    left_on_ground = compute_force_3(left_state, left_forces, heel_l, toes_l, left_on_ground, pelvis_speed=1)
    right_on_ground = compute_force_3(right_state, right_forces, heel_r, toes_r, right_on_ground, pelvis_speed=1)

    if left_on_ground:
        time_left_on_ground.append(i)
    if right_on_ground:
        time_right_on_ground.append(i)

    if left_on_ground and right_on_ground:
        right_foot_usage.append(0.5)
    elif left_on_ground and not right_on_ground:
        right_foot_usage.append(0)
    elif not left_on_ground and right_on_ground:
        right_foot_usage.append(1)
    else:
        right_foot_usage.append(0.5)

    forces.append(np.asarray(F_e) / weight)
    moments.append(np.asarray(M_e) / (weight*height))

    cop_l = find_cop(cop_bodies_left, cop_points_l, state)
    cop_r = find_cop(cop_bodies_right, cop_points_r, state)
    cops_l.append(cop_l)
    cops_r.append(cop_r)
    
right_foot_usage = np.array(right_foot_usage)
spline_interpolation_(right_foot_usage)

right_forces = np.asarray(forces) * np.asarray(right_foot_usage)[:, None] 
left_forces = np.asarray(forces) * (1 - np.asarray(right_foot_usage))[:, None]
#right_moments = np.asarray(moments) * np.asarray(right_foot_usage)[:, None] 
#left_moments = np.asarray(moments) * (1 - np.asarray(right_foot_usage))[:, None] 
right_moments = np.zeros_like(right_forces)
left_moments = np.zeros_like(left_forces)

#right_moments[:, 2] = - right_forces[:, 1] * np.array(cops_r)[:, 0] / height
#left_moments[:, 2] = - left_forces[:, 1] * np.array(cops_l)[:, 0] / height
#right_moments[:, 0] = right_forces[:, 1] * np.array(cops_r)[:, 2] / height
#left_moments[:, 0] = left_forces[:, 1] * np.array(cops_l)[:, 2] / height

# Declare groundtruth force names
grdtruth_force = ['ground_force_vx', 'ground_force_vy', 'ground_force_vz', '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz']
grdtruth_moments = ['ground_torque_x', 'ground_torque_y', 'ground_torque_z', '1_ground_torque_x', '1_ground_torque_y', '1_ground_torque_z']
cops_name = ['ground_force_px', 'ground_force_py', 'ground_force_pz', '1_ground_force_px', '1_ground_force_py', '1_ground_force_pz']

time_grdtruth = []
groundtruth = []
groundtruth_m = []
cops = []
for i in range(exp_data.shape[0]):
    time = exp_data.iloc[i]['time']
    time_grdtruth.append(time)
    grdtruth = np.asarray([exp_data.iloc[i][name] for name in grdtruth_force]) / weight
    grdtruth_m = np.asarray([exp_data.iloc[i][name] for name in grdtruth_moments]) / (weight*height)
    cop = np.asarray([exp_data.iloc[i][name] for name in cops_name])
    groundtruth.append(grdtruth)
    groundtruth_m.append(grdtruth_m)
    cops.append(cop)

# Save results and plot them
write_results('../results/prediction.mot', times, np.asarray(right_forces)*weight, cops_r,
              np.asarray(left_forces)*weight, cops_l, np.asarray(right_moments)*weight*height, np.asarray(left_moments)*weight*height)

plot_results(time_grdtruth, groundtruth, groundtruth_m, times, time_left_on_ground, time_right_on_ground, forces, left_forces,
             right_forces, right_foot_usage=right_foot_usage, moments=moments, left_moments=left_moments, right_moments=right_moments,
             cops=cops, cops_l=cops_l, cops_r=cops_r)

