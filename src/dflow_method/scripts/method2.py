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
bodies_left = [toes_l, calcn_l]
bodies_right = [toes_r, calcn_r]
# Declare thresholds
#thresholds = [(0.06, 0.2), (0.055, 0.1)]
thresholds = [(0.06, 0.2), (0.055, 0.25)]
# case = 0
# if case == 0:
#     bodies_left = [toes_l, calcn_l]
#     bodies_right = [toes_r, calcn_r]
#     thresholds = [(0.06, 0.2), (0.055, 0.25)]
# elif case == 1:
#     bodies_left = [toes_l]
#     bodies_right = [toes_r]
#     thresholds = [(0.06, 0.2)]
# else:
#     bodies_left = [calcn_l]
#     bodies_right = [calcn_r]
#     thresholds = [(0.1, 0.25)]

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
    M_p = [ id_data.iloc[i][name] for name in moments]
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
    right_state = [ np.asarray([ np.asarray([body_part.findStationLocationInGround(state, opensim.Vec3(0.,0.,0.))[i] for i in range(3)]),
                    np.asarray([abs(body_part.findStationVelocityInGround(state, opensim.Vec3(0.,0.,0.))[i]) for i in range(3)])]) for body_part in bodies_right]
    left_state = [  np.asarray([  np.asarray([body_part.findStationLocationInGround(state, opensim.Vec3(0.,0.,0.))[i] for i in range(3)]),
                    np.asarray([abs(body_part.findStationVelocityInGround(state, opensim.Vec3(0.,0.,0.))[i]) for i in range(3)])]) for body_part in bodies_left]

    forces.append(F_e)
    left_on_ground = compute_force(left_state, thresholds, F_e, left_forces)
    right_on_ground = compute_force(right_state, thresholds, F_e, right_forces)

    if left_on_ground:
        time_left_on_ground.append(i)
    if right_on_ground:
        time_right_on_ground.append(i)

# Declare groundtruth force names
grdtruth_force = ['ground_force_vx', 'ground_force_vy', 'ground_force_vz', '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz']
time_grdtruth = []
groundtruth = []
for i in range(exp_data.shape[0]):
    time = exp_data.iloc[i]['time']
    time_grdtruth.append(time)
    grdtruth = [ exp_data.iloc[i][name] for name in grdtruth_force]
    groundtruth.append(grdtruth)

plot_results(time_grdtruth, groundtruth, times, time_left_on_ground, time_right_on_ground, forces, left_forces, right_forces)
