from .hgv_class import HGV
import numpy as np
import random as rd
from math import atan, cos, sin, tan
from backend.utils.standard_atmosphere import calc_speed_of_sound

a_ref = 0.4839  # HGV reference area [m**2]
mass = 907.20   # HGV mass [kg]
re = 6378000     # Radius of Earth [m]
mu = 3.986e14   # Earth gravitational constant [N/m**2]
#vw = rd.randint(0, 80)  # wind speed [m/s]
#aw = rd.randint(0, 45) * np.pi / 180  # wind angle [radians]
vw = 100  # wind speed [m/s]
aw = 45  # wind angle [radians]


class HGV_Wind(HGV):
    def _eom_func(self, t, x, u):
        alpha, sigma = u
        h, theta, phi, v, gam, psi = x

        temp, _, rho = self.atmosphere_func(h)
        r = re + h

        #
        alpha = alpha - atan(mu / r ** 2) * (vw * cos(aw) * sin(gam)) / (v - vw * cos(aw) * cos(gam))

        cl, cd = self.compute_aero(alpha, v / calc_speed_of_sound(temp))
        q = 0.5 * rho * v ** 2
        lift, drag = q * cl * a_ref, q * cd * a_ref

        x_dot = [
            v * sin(gam),  # altitude
            v * cos(gam) * cos(psi) / r,  # downrange
            v * cos(gam) * sin(psi) / (r * cos(theta)),
            -drag * cos(alpha) / mass - (mu * sin(gam) / r ** 2) - lift * sin(alpha) / mass,  # velocity
            #-drag / mass - mu * sin(gam) / r ** 2,  # velocity
            #lift * cos(sigma) / (mass * v) - mu / (v * r ** 2) * cos(gam) + v / r * cos(gam),  # change in fpa
            lift * cos(alpha) / (mass * v) - mu * cos(gam) / (v * r ** 2) - drag * sin(alpha) / (v * mass),  #change in fpa
            lift * sin(sigma) / (mass * cos(gam) * v) + v / r * cos(gam) * sin(psi) * tan(theta)
        ]
        return x_dot