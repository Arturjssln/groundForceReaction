# Implementation of DFLOW ground reaction force prediction.
#
##
import os, sys
import opensim
from utils import simtk_matrix_to_np_array, read_from_storage, foot_on_ground
import csv

DEBUG = False
if len(sys.argv) > 1:
    DEBUG = (sys.argv[1] == "--debug" or sys.argv[1] == "-d")

#Initialize path
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(fileDir)

# model file
model_file = os.path.abspath(
    os.path.join(parentDir, 'scale/model_scaled.osim'))
model = opensim.Model(model_file)
state = model.initSystem()
coordinate_set = model.updCoordinateSet()
pelvis = model.updBodySet().get('pelvis')

# model coordinates for this specific motion
inverse_kinematics_file = os.path.abspath(
    os.path.join(parentDir, 'inverse_kinematics/task_InverseKinematics.mot'))
ik_data = read_from_storage(model_file, inverse_kinematics_file)

# this file contains the results from inverse dynamics ignoring the
# ground reaction forces in the calculation.
inverse_dynamics_file = os.path.abspath(
    os.path.join(parentDir, 'inverse_dynamics/task_InverseDynamics.sto'))
id_data = read_from_storage(model_file, inverse_dynamics_file)

# this file contains the results from inverse dynamics ignoring the
# ground reaction forces in the calculation.
markers_file = os.path.abspath(
    os.path.join(parentDir, 'inverse_kinematics/ik_model_marker_locations.sto'))
ik_marker = read_from_storage(model_file, markers_file)

assert(ik_data.shape == id_data.shape)
assert(ik_data.shape[0] == ik_marker.shape[0])

# Declare moment names
moments = ['pelvis_list_moment', 'pelvis_rotation_moment', 'pelvis_tilt_moment']
# Declare force names
force = ['pelvis_tx_force', 'pelvis_ty_force', 'pelvis_tz_force']
# Declare markers names
markers = ['Heel', 'Midfoot.Lat', 'Toe.Lat', 'Toe.Tip', 'Toe.Med', 'Midfoot.Sup']
coordinates = ['_tx', '_ty', '_tz']

# Declare threshold for markers
thresholds = [0.075, 0.070, 0.035, 0.050, 0.055, 0.090]

for i in range(ik_data.shape[0]):

    time = id_data.iloc[i]['time']

    # get residual moment and forces from inverse dynamics (expressed
    # in local frame of pelvis)
    M_p = [ id_data.iloc[i][name] for name in moments]
    F_p = [ id_data.iloc[i][name] for name in force]

    # get markers position
    markers_r = { name : [ik_marker.iloc[i]["R." + name + coord] for coord in coordinates] for name in markers}
    markers_l = { name : [ik_marker.iloc[i]["L." + name + coord] for coord in coordinates] for name in markers}

    # update model pose
    for coordinate in coordinate_set:
        coordinate.setValue(state, ik_data.iloc[i][coordinate.getName()])


    model.realizePosition(state)
    # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Frame.html
    # get transformation of pelvis in ground frame
    X_PG = pelvis.getTransformInGround(state)
    R_PG = X_PG.R()
    r_P = X_PG.p()

    # do the calculations
    R_GP = simtk_matrix_to_np_array(R_PG).transpose()
    F_e = R_GP.dot(F_p)
    M_e = R_GP.dot(M_p)

    if DEBUG:
        print('-----------------------------------------------------------------------')
        print('Simulation time : {:.02f}'.format(time))
        print('Forces in ground frame : Fx = {:.03f} N, Fy = {:.03f} N, Fz = {:.03f} N'.format(F_e[0], F_e[1], F_e[2]))
        print('Moments in ground frame : Mx = {:.03f} Nm, My = {:.03f} Nm, Mz = {:.03f} Nm'.format(M_e[0], M_e[1], M_e[2]))

    # Need to determine which foot is in contact with the floor
    left_on_floor = foot_on_ground(markers_l, thresholds)
    right_on_floor = foot_on_ground(markers_r, thresholds)

    if DEBUG:
        if left_on_floor:
            print('RIGHT foot on floor')
        if right_on_floor:
            print('LEFT foot on floor')
