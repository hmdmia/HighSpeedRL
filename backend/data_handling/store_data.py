from beluga.utils import load
import numpy as np

files = [
    'emergency_descent_nominal'
]

for filename in files:
    sol_set = load(filename + '.beluga')['solutions']

    data = []
    for traj in sol_set[-1]:
        data.append(np.hstack((np.expand_dims(traj.t, 1), traj.y, traj.u)))

    with open(filename + '.npy', 'wb') as f:
        for datum in data:
            np.save(f, datum)
