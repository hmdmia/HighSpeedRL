import json

import matplotlib.pyplot as plt
import numpy as np


# Function that loads save file
def load_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    return data


# Load series
sol_set_alt = load_file('emergency_descent_alt.json')
sol_set_vel = load_file('emergency_descent_vel.json')
sol_set_fpa = load_file('emergency_descent_fpa.json')
sol_set_all = load_file('emergency_descent_all.json')

# Accessing Array Data: np.array(sol_set[continuation number][solution number]['value field'])[time index, value index]
# In solutions: 't': time, 'y': states, 'u': control
# Order of states: altitude, downrange angle, velocity, fpa

# Example plots of altitude
plt.figure()

for sol in sol_set_all[-1]:
    plt.plot(np.rad2deg(np.array(sol['y'])[:, 1]), np.array(sol['y'])[:, 0])

plt.title('Flight Path')
plt.xlabel('Downrage Angle [deg]')
plt.ylabel('Altitude [m]')

plt.figure()

for sol in sol_set_alt[-1]:
    plt.plot(np.array(sol['t']), np.rad2deg(np.array(sol['u'])[:, 0]))

plt.title('Control History')
plt.xlabel('Time [s]')
plt.ylabel('AoA [deg]')

plt.show()
