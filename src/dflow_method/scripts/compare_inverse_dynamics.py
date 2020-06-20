import os
import sys
import argparse
import opensim
from utils import *
import matplotlib.pyplot as plt

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(fileDir)

# model file
model_file = os.path.abspath(os.path.join(parentDir, 'scale/model_scaled.osim'))

# groundtruth file
groundtruth_file1 = os.path.abspath(os.path.join(parentDir, 'results/groundtruth_task_InverseDynamics.sto'))
groundtruth_data1 = read_from_storage(model_file, groundtruth_file1)

# exp file
exp_file1 = os.path.abspath(os.path.join(parentDir, 'results/exp_task_InverseDynamics.sto'))
exp_data1 = read_from_storage(model_file, exp_file1)

for col in exp_data1.columns:
    if 'moment' not in col:
        continue
    plt.figure()
    plt.plot(exp_data1['time'], exp_data1[col], label='predicted')
    plt.plot(groundtruth_data1['time'], groundtruth_data1[col], label='groundtruth')
    plt.legend()
    plt.title(col)
    plt.savefig(fname=col)
plt.show(block=False)
try:
    plt.pause(0.001)  # Pause for interval seconds.
    input("hit [enter] to end.")
finally:
    plt.close('all')

compare_data_muscle(groundtruth_data1, exp_data1, 'force.csv')



