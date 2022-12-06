import numpy as np
from abc import ABC, abstractmethod
from math import cos, sin
from scipy.integrate import solve_ivp
from random import uniform
from scipy.stats import beta

from backend.base_aircraft_classes.hgv_class import re


class TargetBaseClass(ABC):
    def __init__(self):
        self.target_location = None
        self.target_speed = None
        self.target_heading = None

    @abstractmethod
    def update_location(self, dt):
        """
        Update self.target_location with change in time dt
        :param dt: time difference [s]
        """
        pass

    @abstractmethod
    def reset_location(self):
        """
        Reset target back to initial conditions
        """
        pass


class MovingTarget(TargetBaseClass):
    def __init__(self, target_state):
        super().__init__()
        self.initial_state = target_state
        self.altitude = self.initial_state[0]
        self.reset_location()  # set target location, speed, heading

    def _eom(self, _, theta_phi):
        """
        Equations of motion for change in latitude (theta) and longitude (phi)
        :param _: time (not used)
        :param theta_phi: 2-vector state of theta and phi
        """
        r = re + self.altitude
        theta_phi_dot = (
            self.target_speed * cos(self.target_heading) / r,
            self.target_speed * sin(self.target_heading) / (r * cos(theta_phi[0]))
        )
        return theta_phi_dot

    def update_location(self, dt):
        """
        Update location of target according to _eom
        :param dt: change in time [s]
        """
        theta_phi_0 = self.target_location[1:3].copy()
        sol = solve_ivp(self._eom, (0, 0 + dt), theta_phi_0)
        self.target_location[1:3] = sol.y[:, -1]

    def reset_location(self):
        """
        Reset target back to initial conditions
        """
        target_state = self.initial_state.copy()
        self.target_location = target_state[:3]
        self.target_speed = target_state[3]
        self.target_heading = target_state[5]


class RandomMovingTarget(MovingTarget):
    def __init__(self, target_location, v_range, psi_range, distribution="uniform"):
        self.distribution = distribution
        self.v_range = v_range
        self.psi_range = psi_range

        v, psi = self.sample_state()
        super().__init__(np.append(target_location, (v, 0, psi)))

    def sample_state(self):
        if self.distribution.lower() == "uniform":
            v = uniform(self.v_range[0], self.v_range[1])
            psi = uniform(self.psi_range[0], self.psi_range[1])
        elif self.distribution.lower() == "beta":
            v = beta.rvs(a=1, b=1, size=1, loc=self.v_range[0], scale=self.v_range[1] - self.v_range[0])[0]
            psi = beta.rvs(a=1, b=1, size=1, loc=self.psi_range[0], scale=self.psi_range[1] - self.psi_range[0])[0]
        else:
            raise ValueError("Distribution not yet implemented!")
        return v, psi

    def reset_location(self):
        v, psi = self.sample_state()
        self.target_location = self.initial_state[:3].copy()
        self.target_speed = v
        self.target_heading = psi
