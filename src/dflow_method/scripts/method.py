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


weight = 72.6  # kg
height = 1.8  # m
model = opensim.Model(model_file)
state = model.initSystem()
coordinate_set = model.updCoordinateSet()
pelvis = model.updBodySet().get('pelvis')
calcn_r = model.updBodySet().get('calcn_r')
point_r = [opensim.Vec3(0.,-0.04,0.), opensim.Vec3(0.25,-0.028,-0.015)]
calcn_l = model.updBodySet().get('calcn_l')
point_l = [opensim.Vec3(0.,-0.04,0.), opensim.Vec3(0.25,-0.028,0.015)]
# Declare threshold to considere a foot on the ground
thresholds = [(0.02,0.25), (0.02, 0.2)]

#left_foot = [model.updBodySet().get(body + '_l') for body in fool_elt]
#right_foot = [model.updBodySet().get(body + '_r') for body in fool_elt]

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
    M_p = [ id_data.iloc[i][name] for name in moment]
    F_p = [ id_data.iloc[i][name] for name in force]

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
    right_state = np.asarray([(calcn_r.findStationLocationInGround(state, pt)[1],
                    abs(calcn_r.findStationVelocityInGround(state, pt)[1]),
                    [calcn_r.findStationLocationInGround(state, pt)[i] for i in range(3)]) for pt in point_r])
    left_state = np.asarray([(calcn_l.findStationLocationInGround(state, pt)[1],
                    abs(calcn_l.findStationVelocityInGround(state, pt)[1]),
                    [calcn_l.findStationLocationInGround(state, pt)[i] for i in range(3)]) for pt in point_l])

    pelvis_pos = np.asarray([pelvis.getPositionInGround(state)[i] for i in range(3)])

    left_foot_position.append([elt[2] - pelvis_pos for elt in left_state])
    right_foot_position.append([elt[2] - pelvis_pos for elt in right_state])
    left_on_ground, right_on_ground = foot_on_ground(left_state, right_state, thresholds)

    left_ground = any([on_ground for on_ground, _ in left_on_ground])
    right_ground = any([on_ground for on_ground, _ in right_on_ground])

    if left_ground:
        time_left_on_ground.append(i)
    if right_ground:
        time_right_on_ground.append(i)


    assert(left_ground or right_ground)

    # Definition of the center of pressure:
    CoP = [M_e[2] / F_e[1], 0, - M_e[0] / F_e[1]]
    cops.append(CoP)
    # Calculating weighting factors
    d_l = minDistance1d(CoP, left_state[:, 2], pelvis_pos)
    d_r = minDistance1d(CoP, right_state[:, 2], pelvis_pos)
    w_r = d_l / (d_l + d_r)
    w_l = d_r / (d_l + d_r)
    F_e /= weight
    M_e /= weight * height
    forces.append(F_e)
    moments.append(M_e)
    if left_ground and right_ground:
        left_forces.append(F_e * w_l)
        right_forces.append(F_e * w_r)
        left_moments.append(M_e * w_l)
        right_moments.append(M_e * w_r)
        right_foot_usage.append(w_r)

    elif left_ground and not right_ground :
        left_forces.append(F_e)
        right_forces.append([0, 0, 0])
        left_moments.append(M_e)
        right_moments.append([0, 0, 0])
        right_foot_usage.append(0)
    elif not left_ground and right_ground :
        right_forces.append(F_e)
        left_forces.append([0, 0, 0])
        left_moments.append([0, 0, 0])
        right_moments.append(M_e)
        right_foot_usage.append(1)
    else:
        right_forces.append([0, 0, 0])
        left_forces.append([0, 0, 0])
        left_moments.append([0, 0, 0])
        right_moments.append([0, 0, 0])
        right_foot_usage.append(0.5)


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


times = np.asarray(times)
time_grdtruth = np.asarray(time_grdtruth)
groundtruth = np.asarray(groundtruth) / weight
groundtruth_m = np.asarray(groundtruth_m) / (weight*height)
forces = np.asarray(forces)
left_forces = np.asarray(left_forces)
right_forces = np.asarray(right_forces)
moments = np.asarray(moments)
left_moments = np.asarray(left_moments) * np.array([0.0001, 1, 0.0001])
right_moments = np.asarray(right_moments) * np.array([0.0001, 1, 0.0001])

compare_results(time_grdtruth, groundtruth, groundtruth_m, None, times, left_forces, right_forces, left_moments, right_moments)

plot_results(time_grdtruth, groundtruth, groundtruth_m, times, \
    time_left_on_ground, time_right_on_ground, forces, left_forces, right_forces, \
    cops, right_foot_position, left_foot_position, right_foot_usage, \
    moments, left_moments, right_moments)
