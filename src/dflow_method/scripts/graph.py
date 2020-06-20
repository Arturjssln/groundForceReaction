import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from utils import *

time = np.linspace(0,0.5, 50)

pts = np.zeros(50, dtype=np.float64)
pts[10:40] = 0.5
pts[40:] = 1

spline_interpolation_(pts)

plt.plot(time, pts*100)
plt.xticks([0.1, 0.4], [r'$t_{HS}$', r'$t_{TO}$'])
plt.axvline(0.1, ls=':', color='black')
plt.axvline(0.4, ls=':', color='black')
plt.title('Percentage of force applied on one foot')
plt.xlabel('Time [s]')
plt.ylabel('Distribution of force on one foot [%]')
plt.axvspan(0.1, 0.4, facecolor='gray', alpha=0.2)
plt.text(0.05, 15, 'Swinging\nfoot', horizontalalignment='center')
plt.text(0.25, 15, 'Double stance\nphase', horizontalalignment='center')
plt.text(0.45, 15, 'Stance\nfoot', horizontalalignment='center')

x = np.linspace(0,1.2,100)
y = 0.5 * (np.tanh(-np.pi*(2*(x-0.15)-0.85)/0.85)+1)

plt.figure()
plt.plot(x, y)
plt.xticks([0.15, 1], [r'$u_{th}$', '1'])
plt.axvline(0.15, ls=':', color='black')
plt.axvline(1, ls=':', color='black')
plt.title('Smoothing function')
plt.xlabel('Input ratio')
plt.ylabel('Smoothed ratio')
plt.show()
