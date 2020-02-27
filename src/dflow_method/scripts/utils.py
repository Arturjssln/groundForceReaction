# Utility functions.
#
# author: Dimitar Stanev <jimstanev@gmail.com>
##
import re
import os
import opensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText


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
