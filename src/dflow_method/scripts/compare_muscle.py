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
groundtruth_file1 = os.path.abspath(os.path.join(parentDir, 'cmc/groundtruth_Actuation_force.sto'))
groundtruth_data1 = read_from_storage(model_file, groundtruth_file1)
groundtruth_file2 = os.path.abspath(os.path.join(parentDir, 'cmc/groundtruth_Actuation_speed.sto'))
groundtruth_data2 = read_from_storage(model_file, groundtruth_file2)
groundtruth_file3 = os.path.abspath(os.path.join(parentDir, 'cmc/groundtruth_Actuation_power.sto'))
groundtruth_data3 = read_from_storage(model_file, groundtruth_file3)
# exp file
exp_file1 = os.path.abspath(os.path.join(parentDir, 'cmc/exp_Actuation_force.sto'))
exp_data1 = read_from_storage(model_file, exp_file1)
exp_file2 = os.path.abspath(os.path.join(parentDir, 'cmc/exp_Actuation_speed.sto'))
exp_data2 = read_from_storage(model_file, exp_file2)
exp_file3 = os.path.abspath(os.path.join(parentDir, 'cmc/exp_Actuation_power.sto'))
exp_data3 = read_from_storage(model_file, exp_file3)
print(exp_data1.columns)
for col in exp_data1.columns:
    if col == 'time':
        continue
    plt.figure()
    plt.plot(exp_data1['time'], exp_data1[col], label='exp')
    plt.plot(groundtruth_data1['time'], groundtruth_data1[col], label='groundtruth')
    plt.legend()
    break
plt.show()
try:
    plt.pause(0.001)  # Pause for interval seconds.
    input("hit [enter] to end.")
finally:
    plt.close('all')

compare_data_muscle(groundtruth_data1, exp_data1, 'force.csv')
compare_data_muscle(groundtruth_data2, exp_data2, 'speed.csv')
compare_data_muscle(groundtruth_data3, exp_data3, 'power.csv')


