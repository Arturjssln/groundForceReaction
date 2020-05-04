# Utility functions.
#
# author: Dimitar Stanev <jimstanev@gmail.com>
##
import re
import os, sys
import opensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def osim_array_to_list(array):
    """Convert OpenSim::Array<T> to Python list.
    """
    temp = []
    for i in range(array.getSize()):
        temp.append(array.get(i))

    return temp


def list_to_osim_array_str(list_str):
    """Convert Python list of strings to OpenSim::Array<string>."""
    arr = opensim.ArrayStr()
    for element in list_str:
        arr.append(element)

    return arr


def np_array_to_simtk_matrix(array):
    """Convert numpy array to SimTK::Matrix"""
    n, m = array.shape
    M = opensim.Matrix(n, m)
    for i in range(n):
        for j in range(m):
            M.set(i, j, array[i, j])

    return M


def simtk_matrix_to_np_array(matrix):
    n = matrix.nrow()
    m = matrix.ncol()
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = matrix.get(i, j)

    return M


def read_from_storage(model_file, file_name):
    """Read OpenSim.Storage files.
    Parameters
    ----------
    file_name: (string) path to file
    Returns
    -------
    tuple: (labels, time, data)
    """
    sto = opensim.Storage(file_name)
    if sto.isInDegrees():
        model = opensim.Model(model_file)
        model.initSystem()
        model.getSimbodyEngine().convertDegreesToRadians(sto)

    # sto.resampleLinear(sampling_interval)

    labels = osim_array_to_list(sto.getColumnLabels())
    time = opensim.ArrayDouble()
    sto.getTimeColumn(time)
    time = osim_array_to_list(time)
    data = []
    for i in range(sto.getSize()):
        temp = osim_array_to_list(sto.getStateVector(i).getData())
        temp.insert(0, time[i])
        data.append(temp)

    df = pd.DataFrame(data, columns=labels)
    df.index = df.time
    return df


def index_containing_substring(list_str, pattern):
    """For a given list of strings finds the index of the element that
    contains the substring.

    Parameters
    ----------
    list_str: list of str

    pattern: str
         pattern


    Returns
    -------
    indices: list of int
         the indices where the pattern matches

    """
    return [i for i, item in enumerate(list_str)
            if re.search(pattern, item)]


def plot_sto_file(file_name, plot_file, plots_per_row=4, pattern=None,
                  title_function=lambda x: x):
    """Plots the .sto file (OpenSim) by constructing a grid of subplots.

    Parameters
    ----------
    sto_file: str
        path to file
    plot_file: str
        path to store result
    plots_per_row: int
        subplot columns
    pattern: str, optional, default=None
        plot based on pattern (e.g. only pelvis coordinates)
    title_function: lambda
        callable function f(str) -> str
    """
    df = read_from_storage(file_name)
    labels = df.columns.to_list()
    data = df.to_numpy()

    if pattern is not None:
        indices = index_containing_substring(labels, pattern)
    else:
        indices = range(1, len(labels))

    n = len(indices)
    ncols = int(plots_per_row)
    nrows = int(np.ceil(float(n) / plots_per_row))
    pages = int(np.ceil(float(nrows) / ncols))
    if ncols > n:
        ncols = n

    with PdfPages(plot_file) as pdf:
        for page in range(0, pages):
            fig, ax = plt.subplots(nrows=ncols, ncols=ncols,
                                   figsize=(8, 8))
            ax = ax.flatten()
            for pl, col in enumerate(indices[page * ncols ** 2:page *
                                             ncols ** 2 + ncols ** 2]):
                ax[pl].plot(data[:, 0], data[:, col])
                ax[pl].set_title(title_function(labels[col]))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()


def foot_on_ground(left_state, right_state, thresholds):
    """Return(Bool, Bool) that determines if each foot is on the floor

    Parameters
    ----------
    left_state: List(Tuples)
        Tuples containing (position, velocity)
        Each element of the list correspond to a body
    right_state: List(Tuples)
        Tuples containing (position, velocity)
        Each element of the list correspond to a body
    thresholds: List(Tuples)
        Tuples containing (position, velocity)
        Each element of the list correspond to a body
    """
    out_left = []
    out_right = []
    for left_body, right_body, threshold in zip(left_state, right_state, thresholds):
        #out_left.append((left_body[0] < threshold[0] and left_body[1] < threshold[1], left_body[2]))
        #out_right.append((right_body[0] < threshold[0] and right_body[1] < threshold[1], right_body[2]))
        out_left.append((left_body[0] < threshold[0], left_body[2]))
        out_right.append((right_body[0] < threshold[0], right_body[2]))
    return (out_left, out_right)


def import_from_storage(parentDir):
    # model file
    model_file = os.path.abspath(
        os.path.join(parentDir, 'scale/model_scaled.osim'))

    # model coordinates for this specific motion
    inverse_kinematics_file = os.path.abspath(
        os.path.join(parentDir, 'inverse_kinematics/task_InverseKinematics.mot'))
    ik_data = read_from_storage(model_file, inverse_kinematics_file)

    # this file contains the results from inverse dynamics ignoring the
    # ground reaction forces in the calculation.
    inverse_dynamics_file = os.path.abspath(
        os.path.join(parentDir, 'inverse_dynamics/task_InverseDynamics.sto'))
    id_data = read_from_storage(model_file, inverse_dynamics_file)

    # this file contains the generalized speeds
    velocity_file = os.path.abspath(
        os.path.join(parentDir, 'results/model_scaled_Kinematics_u.sto'))
    u = read_from_storage(model_file, velocity_file)

    # this file contains the generalized acceleration
    acceleration_file = os.path.abspath(
        os.path.join(parentDir, 'results/model_scaled_Kinematics_dudt.sto'))
    a = read_from_storage(model_file, acceleration_file)

    # this file contains the generalized acceleration
    experimental_file = os.path.abspath(
        os.path.join(parentDir, 'experimental_data/task_grf.mot'))
    exp_data = read_from_storage(model_file, experimental_file)

    return (model_file, ik_data, id_data, u, a, exp_data)

def color_background(ax, left_idx, right_idx, times):
    l = 0
    for i in left_idx:
        if i+1 < len(times):
            ax.axvspan(times[i], times[i+1], facecolor='g', alpha=0.2, label =  "_"*l + "Left foot on groud")
            l = 1
    l = 0
    for i in right_idx:
        if i+1 < len(times):
            ax.axvspan(times[i], times[i+1], facecolor='r', alpha=0.2, label =  "_"*l + "Right foot on ground")
            l = 1

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
    legend_x = 1
    legend_y = 0.5
    #ax.legend()
    ax.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))

# Calculate distance to closest point on plane XZ
def minDistance2d(pt1, pts2, pelvis):
    min_dist = np.inf
    for pt2 in pts2:
        dist = np.sqrt((pt1[0] - (pt2[0] - pelvis[0]))**2 + (pt1[2] - (pt2[2] - pelvis[2]))**2)
        if dist < min_dist:
            min_dist = dist
    return min_dist

# Calculate distance to closest point on axis Z
def minDistance1d(pt1, pts2, pelvis):
    min_dist = np.inf
    for pt2 in pts2:
        dist = abs(pt1[2] - (pt2[2] - pelvis[2]))
        if dist < min_dist:
            min_dist = dist
    return min_dist

def smooth_function(y_ratio, v_ratio):
    #y_smooth = 0.5 * (np.cos((y_ratio - 0.8)/((1-0.8)*np.pi))+1)
    #v_smooth = 0.5 * (np.cos((v_ratio - 0.15)/((1-0.15)*np.pi))+1)
    y_smooth = 0.5 * (np.tanh(-np.pi*(2*(y_ratio-0.8)-0.2)/0.2)+1)
    v_smooth = 0.5 * (np.tanh(-np.pi*(2*(v_ratio-0.15)-0.85)/0.85)+1)
    return (y_smooth, v_smooth)

def compute_force(states, thresholds, force, forces):
    """Return Bool that determines if feet is on the floor

    Parameters
    ----------


    """
    on_floor = True
    min_y_ratio = 1
    min_v_ratio = 1
    y_smooth = 0
    v_smooth = 0
    # find best contact point
    for state, threshold in zip(states, thresholds):
        y_ratio = state[0][1] / threshold[0]
        v_ratio = state[1][1] / threshold[1]
        y_tmp, v_tmp = smooth_function(y_ratio, v_ratio)
        if y_ratio <= min_y_ratio:
            y_smooth = y_tmp
            min_y_ratio = y_ratio
        if v_ratio <= min_v_ratio:
            v_smooth = v_tmp
            min_v_ratio = v_ratio

    if min_y_ratio < 1 and min_v_ratio < 1:
        force_smooth = force * y_smooth * v_smooth
    else:
        force_smooth = [0, 0, 0]
        on_floor = False
    forces.append(force_smooth)
    return on_floor

def compute_force_TEST(states, thresholds, force, forces):
    """Return Bool that determines if feet is on the floor

    Parameters
    ----------


    """
    on_floor = True
    mean_y_ratio = 0
    mean_v_ratio = 0
    mean_y_smooth = 0
    mean_v_smooth = 0
    nb_element = len(states)
    # find best contact point
    for state, threshold in zip(states, thresholds):
        y_ratio = state[0][1] / threshold[0]
        v_ratio = state[1][1] / threshold[1]
        y_smooth, v_smooth = smooth_function(y_ratio, v_ratio)
        #if y_ratio <= 1:
        mean_y_ratio += y_ratio / nb_element
        mean_y_smooth += y_smooth / nb_element
        #if v_ratio <= 1:
        mean_v_ratio += v_ratio / nb_element
        mean_v_smooth += v_smooth / nb_element

    if mean_y_ratio < 1 and mean_v_ratio < 1:
        force_smooth = force * mean_y_smooth * mean_v_smooth
    else:
        force_smooth = [0, 0, 0]
        on_floor = False
    forces.append(force_smooth)
    return on_floor

def plot_results(time_grdtruth = None, groundtruth = None, times = None, \
                    time_left_on_ground = None, time_right_on_ground = None, \
                    forces = None, left_forces = None, right_forces = None, cops = None,\
                    right_foot_position = None, left_foot_position = None, right_foot_usage = None):
    if times is not None and \
        time_left_on_ground is not None and \
        time_right_on_ground is not None:
        display_background = True
    else:
        display_background = False


    if groundtruth is not None and \
        time_grdtruth is not None and \
        forces is not None and \
        left_forces is not None and \
        right_forces is not None:
        groundtruth = np.asarray(groundtruth)
        forces = np.asarray(forces)
        left_forces = np.asarray(left_forces)
        right_forces = np.asarray(right_forces)
        plt.figure()
        plt.suptitle('Total force')
        ax = plt.subplot(311)
        ax.plot(time_grdtruth, groundtruth[:, 0] + groundtruth[:, 3], label = 'groundtruth')
        ax.plot(times, forces[:, 0], label = 'prediction')
        ax.set_title('Ground force (x-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)
        ax = plt.subplot(312)
        ax.plot(time_grdtruth, groundtruth[:, 1] + groundtruth[:, 4], label = 'groundtruth')
        ax.plot(times, forces[:, 1], label = 'prediction')
        ax.set_title('Ground force (y-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)
        ax = plt.subplot(313)
        ax.plot(time_grdtruth, groundtruth[:, 2] + groundtruth[:, 5], label = 'groundtruth')
        ax.plot(times, forces[:, 2], label = 'prediction')
        ax.set_title('Ground force (z-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)

        plt.figure()
        plt.suptitle('Left foot')
        ax = plt.subplot(311)
        ax.plot(time_grdtruth, groundtruth[:, 3], label = 'groundtruth')
        ax.plot(times, left_forces[:, 0], label = 'prediction')
        ax.set_title('Ground force (x-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)
        ax = plt.subplot(312)
        ax.plot(time_grdtruth, groundtruth[:, 4], label = 'groundtruth')
        ax.plot(times, left_forces[:, 1], label = 'prediction')
        ax.set_title('Ground force (y-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)
        ax = plt.subplot(313)
        ax.plot(time_grdtruth, groundtruth[:, 5], label = 'groundtruth')
        ax.plot(times, left_forces[:, 2], label = 'prediction')
        ax.set_title('Ground force (z-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)

        plt.figure()
        plt.suptitle('Right foot')
        ax = plt.subplot(311)
        ax.plot(time_grdtruth, groundtruth[:, 0], label = 'groundtruth')
        ax.plot(times, right_forces[:, 0], label = 'prediction')
        ax.set_title('Ground force (x-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)
        ax = plt.subplot(312)
        ax.plot(time_grdtruth, groundtruth[:, 1], label = 'groundtruth')
        ax.plot(times, right_forces[:, 1], label = 'prediction')
        ax.set_title('Ground force (y-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)
        ax = plt.subplot(313)
        ax.plot(time_grdtruth, groundtruth[:, 2], label = 'groundtruth')
        ax.plot(times, right_forces[:, 2], label = 'prediction')
        ax.set_title('Ground force (z-axis)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)

    if cops is not None and \
        right_foot_position is not None and \
        left_foot_position is not None:

        cops = np.asarray(cops)
        right_foot_position = np.asarray(right_foot_position)
        left_foot_position = np.asarray(left_foot_position)
        plt.figure()
        plt.suptitle('Position')
        ax = plt.subplot(211)
        ax.plot(times, cops[:, 0], label='center of pressure')
        ax.plot(times, left_foot_position[:, 0, 0], label='left foot calcn')
        ax.plot(times, right_foot_position[:, 0, 0], label='right foot calcn')
        ax.plot(times, left_foot_position[:, 1, 0], label='left foot toes')
        ax.plot(times, right_foot_position[:, 1, 0], label='right foot toes')
        ax.set_title('x-axis')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)

        ax = plt.subplot(212)
        ax.plot(times, cops[:, 2], label='center of pressure')
        ax.plot(times, left_foot_position[:, 0, 2], label='left foot calcn')
        ax.plot(times, right_foot_position[:, 0, 2], label='right foot calcn')
        ax.plot(times, left_foot_position[:, 1, 2], label='left foot toes')
        ax.plot(times, right_foot_position[:, 1, 2], label='right foot toes')
        ax.set_title('z-axis')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)


    if right_foot_usage is not None:
        right_foot_usage = np.asarray(right_foot_usage)
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(times, right_foot_usage*100, label='Right foot')
        ax.plot(times, 100 - right_foot_usage*100, label='Left foot')
        ax.set_title('Pourcentage of force applied on each foot')
        ax.set_ylabel('Weighting factor (in %)')
        if display_background:
            color_background(ax, time_left_on_ground, time_right_on_ground, times)


    plt.show(block = False)

    try:
        plt.pause(0.001) # Pause for interval seconds.
        input("hit [enter] to end.")
    finally:
        plt.close('all')
