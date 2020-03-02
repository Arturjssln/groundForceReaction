import os, sys
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('../data/data_marker.csv') as file:
    data = csv.reader(file)
    data_array = []
    i = 0
    for row in data:
        if i == 0:
            i = 1
            header = np.asarray(row, dtype = np.dtype(str))
        else:
            data_array.append(np.asarray(row, dtype = np.float32))


    data_array = np.asarray(data_array)

    # Display data :
    fig1, axes1 = plt.subplots(3, 2)
    fig1.suptitle("Right foot")
    for i in range(len(axes1)):
        for j in range(len(axes1[i])):
            axes1[i,j].plot(np.linspace(0., 2.5, int(len(data_array)/2)), data_array[::2, i*2+j])
            axes1[i,j].set_title(header[i*2+j])

    # Display data :
    fig2, axes2 = plt.subplots(3, 2)
    fig2.suptitle("Left foot")
    for i in range(len(axes2)):
        for j in range(len(axes2[i])):
            axes2[i,j].plot(np.linspace(0., 2.5, int(len(data_array)/2)), data_array[1::2, i*2+j])
            axes2[i,j].set_title(header[i*2+j])
    plt.show()
