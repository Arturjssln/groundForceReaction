# Implementation of DFLOW ground reaction force prediction.
#
##
import os, sys
import opensim
from utils import simtk_matrix_to_np_array, read_from_storage
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


with open('../data/data_marker.csv', 'w') as csvfile:
    fieldnames = markers
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for i in range(ik_data.shape[0]):

        time = id_data.iloc[i]['time']
        writer.writerow({ "time" : time, name : ik_marker.iloc[i]["R." + name + "_ty"] for name in markers})
        writer.writerow({ "time" : time, name : ik_marker.iloc[i]["L." + name + "_ty"] for name in markers})
