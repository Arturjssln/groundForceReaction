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
fool_elt = ['calcn', 'toes']
# Declare threshold to considere a foot on the ground
thresholds = [(0.1,0.2), (0.06, 0.09)]

left_foot = [model.updBodySet().get(body + '_l') for body in fool_elt]
right_foot = [model.updBodySet().get(body + '_r') for body in fool_elt]

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

    if args.debug:
        print('-----------------------------------------------------------------------')
        print('Simulation time : {:.02f}'.format(time))
        print('Forces in ground frame : Fx = {:.03f} N, Fy = {:.03f} N, Fz = {:.03f} N'.format(F_e[0], F_e[1], F_e[2]))
        print('Moments in ground frame : Mx = {:.03f} Nm, My = {:.03f} Nm, Mz = {:.03f} Nm'.format(M_e[0], M_e[1], M_e[2]))

    # Determine which foot is on ground
    right_state = [(body.findStationLocationInGround(state, opensim.Vec3(0.,0.,0.))[1],
                    abs(body.findStationVelocityInGround(state, opensim.Vec3(0.,0.,0.))[1]),
                    body.getPositionInGround(state)) for body in right_foot]
    left_state = [(body.findStationLocationInGround(state, opensim.Vec3(0.,0.,0.))[1],
                    abs(body.findStationVelocityInGround(state, opensim.Vec3(0.,0.,0.))[1]),
                    body.getPositionInGround(state)) for body in left_foot]

    left_on_ground, right_on_ground = foot_on_ground(left_state, right_state, thresholds)

    left_ground = left_on_ground[0][0] or left_on_ground[1][0]
    right_ground = right_on_ground[0][0] or right_on_ground[1][0]

    forces.append(F_e)
    if left_ground and not right_ground :
        left_forces.append(F_e)
        right_forces.append([0, 0, 0])
    elif not left_ground and right_ground :
        right_forces.append(F_e)
        left_forces.append([0, 0, 0])
    else:
        right_forces.append([0, 0, 0])
        left_forces.append([0, 0, 0])

    assert(left_on_ground[0][0] or left_on_ground[1][0] or right_on_ground[0][0] or right_on_ground[1][0])

    if args.debug:
        print("(Left foot) Heel on ground : {}, Toes on ground : {}".format(left_on_ground[0][0], left_on_ground[1][0]))
        print("(Right foot) Heel on ground : {}, Toes on ground : {}".format(right_on_ground[0][0], right_on_ground[1][0]))

    # Definition of the center of pressure:
    CoP = [M_e[2] / F_e[1], 0, - M_e[0] / F_e[1]]

# Declare groundtruth force names
grdtruth_force = ['ground_force_vx', 'ground_force_vy', 'ground_force_vz', '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz']
time_grdtruth = []
groundtruth = []
for i in range(exp_data.shape[0]):
    time = exp_data.iloc[i]['time']
    time_grdtruth.append(time)
    grdtruth = [ exp_data.iloc[i][name] for name in grdtruth_force]
    groundtruth.append(grdtruth)

groundtruth = np.asarray(groundtruth)
forces = np.asarray(forces)
left_forces = np.asarray(left_forces)
right_forces = np.asarray(right_forces)

plt.figure()
plt.suptitle('Total force')
plt.subplot(311)
plt.plot(time_grdtruth, groundtruth[:, 0] + groundtruth[:, 3])
plt.plot(times, forces[:, 0])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (x-axis)')
plt.subplot(312)
plt.plot(time_grdtruth, groundtruth[:, 1] + groundtruth[:, 4])
plt.plot(times, forces[:, 1])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (y-axis)')
plt.subplot(313)
plt.plot(time_grdtruth, groundtruth[:, 2] + groundtruth[:, 5])
plt.plot(times, forces[:, 2])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (z-axis)')

plt.figure()
plt.suptitle('Left foot')
plt.subplot(311)
plt.plot(time_grdtruth, groundtruth[:, 3])
plt.plot(times, left_forces[:, 0])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (x-axis)')
plt.subplot(312)
plt.plot(time_grdtruth, groundtruth[:, 4])
plt.plot(times, left_forces[:, 1])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (y-axis)')
plt.subplot(313)
plt.plot(time_grdtruth, groundtruth[:, 5])
plt.plot(times, left_forces[:, 2])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (z-axis)')

plt.figure()
plt.suptitle('Right foot')
plt.subplot(311)
plt.plot(time_grdtruth, groundtruth[:, 0])
plt.plot(times, right_forces[:, 0])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (x-axis)')
plt.subplot(312)
plt.plot(time_grdtruth, groundtruth[:, 1])
plt.plot(times, right_forces[:, 1])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (y-axis)')
plt.subplot(313)
plt.plot(time_grdtruth, groundtruth[:, 2])
plt.plot(times, right_forces[:, 2])
plt.legend(['groundtruth', 'prediction'])
plt.title('Ground force (z-axis)')
plt.show()
