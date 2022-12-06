import numpy as np
from abc import abstractmethod

from backend.rl_base_classes.rl_base_class import RLBaseClass
from backend.base_aircraft_classes.scramjet_class import Scramjet


class ScramjetBaseClass(RLBaseClass, Scramjet):
    def __init__(self, initial_state):
        self.initial_state = initial_state

        RLBaseClass.__init__(self)
        Scramjet.__init__(self, self.initial_state)

        self._target_altitude = 3000
        self._fpa_tol = 1 * np.pi / 180
        self._max_time = 50.
        self._max_fpa = 89*np.pi/180
        self._fpa_scale = 15*np.pi/180

        self.n_actions = 11
        self._aoa_options = np.linspace(-10, 10, self.n_actions)/180*np.pi

        self.low = np.array([0, 0, 500, -89 * np.pi / 180, 600])  # time, altitude, velocity, fpa, mass
        self.high = np.array(
            [self._max_time, 30000, 4000, 89 * np.pi / 180, 1300])  # time, altitude, velocity, fpa, mass
        self.dt = 1

    def observe(self):
        t, h, v, gam, mass = self.time, self.state[0], self.state[3], self.state[4], self.state[6]

        if (h <= self._target_altitude) & (abs(gam) <= self._fpa_tol):
            self.success, self.done = True, True

        elif (t > self._max_time) or (h <= 0) or (abs(gam) >= 80*np.pi/180):
            self.done = True

        return np.array([t, h, v, gam, mass])

    def _reward1(self):
        dist_alt = (self.state[0] - self._target_altitude) / (self.initial_state[0] - self._target_altitude)
        if self.success:
            time_scaled = self.time/self._max_time
            fpa_scaled = abs(self.state[4]/self._max_fpa)
            scale = 10
            reward = scale * (1-time_scaled)*(1-fpa_scaled)

        elif self.done:
            # penalty

            reward = -2*dist_alt
        else:
            reward = 1 - dist_alt

        return reward

    def _reward2(self):
        dist_alt = (self.state[0] - self._target_altitude) / (self.initial_state[0] - self._target_altitude)
        fpa_scaled = abs(self.state[4]/self._max_fpa)
        if self.success:
            time_scaled = self.time / self._max_time
            reward = 100 * (1 - time_scaled)

        elif self.done:
            reward = -100

        else:
            reward = self.dt * (-1 - dist_alt - fpa_scaled**4*(1-dist_alt)**4)

        return reward

    def _reward3(self):
        scaled_alt = (self.state[0] - self._target_altitude) / (self.initial_state[0] - self._target_altitude)
        fpa_scaled = abs(self.state[4]/self._fpa_scale)
        if self.success:
            reward = 100 * (self._max_time - self.time) / self._max_time

        elif self.done:
            reward = - fpa_scaled**4 - 100 * self.time / self._max_time - 10 * scaled_alt

        else:
            reward = -10 * self.dt - scaled_alt

            # if dist_alt < 0.1:
            #     reward -= 10 * fpa_scaled ** 2

        return reward

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def _inner_step(self, action):
        pass

