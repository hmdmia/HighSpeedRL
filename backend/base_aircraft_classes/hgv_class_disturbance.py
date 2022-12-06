from .hgv_class import HGV
import numpy as np
import random

rng = float(random.randint(0, 30))

rng2 = float(random.randint(0, 30))

rand_v = [rng,rng2]


class HGV_Disturbance(HGV):
    '''
    Disturbance to observation data to velocity and fpa data by

    '''

    def dynamics(self, _t, _x):
        g_t = [rand_v[0], rand_v[0] + 1, rand_v[0] + 2]
        w_g = [rand_v[1], rand_v[1] + 1, rand_v[1] + 2]

        if 50.0 >= _t > 30.1:
            rng = float(random.randint(0, 30))
            rng2 = float(random.randint(0, 30))
            g_t = [rng, rng+1, rng+2]
            w_g = [rng2, rng2+1, rng2+2]
        else:
            pass
        # randomly vary velocity and fpa at random times for 3 seconds
        if _t in g_t:
            _x[3] = np.random.normal(_x[3],20)
        elif _t in w_g:
            _x[4] = np.random.normal(_x[4],1.5 * (3.14159 / 180))
        else:
            pass

        y_dot = super().dynamics(_t,_x)
        return y_dot