# Implementation of DFLOW ground reaction force prediction.
#
##
import os
import opensim
from utils import simtk_matrix_to_np_array, read_from_storage

# model file
model_file = os.path.abspath(
    '../scale/model_scaled.osim')
model = opensim.Model(model_file)
state = model.initSystem()
coordinate_set = model.updCoordinateSet()
pelvis = model.updBodySet().get('pelvis')

# model coordinates for this specific motion
inverse_kinematics_file = os.path.abspath(
    '../inverse_kinematics/task_InverseKinematics.mot')
ik_data = read_from_storage(model_file, inverse_kinematics_file)

# this file contains the results from inverse dynamics ignoring the
# ground reaction forces in the calculation.
inverse_dynamics_file = os.path.abspath(
    '../inverse_dynamics/task_InverseDynamics.sto')
id_data = read_from_storage(model_file, inverse_dynamics_file)

assert(ik_data.shape == id_data.shape)

for i in range(ik_data.shape[0]):
    # get residual moment and forces from inverse dynamics (expressed
    # in local frame of pelvis)
    Mx = id_data.iloc[i]['pelvis_list_moment']
    My = id_data.iloc[i]['pelvis_rotation_moment']
    Mz = id_data.iloc[i]['pelvis_tilt_moment']
    Fx = id_data.iloc[i]['pelvis_tx_force']
    Fy = id_data.iloc[i]['pelvis_ty_force']
    Fz = id_data.iloc[i]['pelvis_tz_force']

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
    F_e = R_GP.dot([Fx, Fy, Fz])
    # ...
