import numpy as np
from abc import abstractmethod

from backend.rl_base_classes.rl_base_class import RLBaseClass
from backend.base_aircraft_classes.hgv_class import HGV


class AoABaseClass(RLBaseClass, HGV):
    """
    Class to contain data and methods common for AoA/emergency descent problem
    """
    def __init__(self, initial_state):
        """
        Initial HGV for AoA problem with given initial state
        :param initial_state: initial state of vehicle, [altitude, latitude, longitude, velocity, FPA, heading]
        """
        self.initial_state = initial_state

        RLBaseClass.__init__(self)
        HGV.__init__(self, self.initial_state)

        self._target_altitude = 3000
        self._fpa_tol = 0.25 * np.pi / 180
        self._max_time = 50.
        self._max_fpa = 89*np.pi/180
        self._fpa_scale = 15*np.pi/180

        # Define angle of attack options available for actions
        self.n_actions = 11
        self._aoa_options = np.linspace(-20, 20, self.n_actions)/180*np.pi

        # Lowest expected values of observation space
        self.low = np.array([0, 0, 500, -89 * np.pi / 180])  # time, altitude, velocity, fpa
        # Highest expected values of observation space
        self.high = np.array([self._max_time+1, 30000, 4000, 89 * np.pi / 180])  # time, altitude, velocity, fpa
        self.dt = 1

        self._target_altitude_threshold = 2*self._target_altitude

        # Define event to end integration to set done flag
        self.training_events = [self.generate_emergency_descent_event(trigger_alt=self._target_altitude_threshold)]

    def observe(self):
        """
        Define observation method for AoA problem.

        The observation contains the current time (t), altitude (h), velocity (v), and FPA (gam)

        :return: observation vector
        """
        t, h, v, gam = self.time, self.state[0], self.state[3], self.state[4]

        # Define success condition and set appropriate flags
        if (h <= self._target_altitude) & (abs(gam) <= self._fpa_tol):
            self.success, self.done = True, True

        # Define condition to unsuccessfully end episode
        elif (t > self._max_time) or (h <= 0) or (abs(gam) >= 80*np.pi/180):
            self.done = True

        elif (h <= self._target_altitude_threshold) & (abs(gam) <= self._fpa_tol):
            self.done = True

        return np.array([t, h, v, gam])

    def _reward1(self):
        """
        A potential reward method
        :return: reward value
        """
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
        """
        A potential reward method
        :return: reward value
        """
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
        """
        A potential reward method
        :return: reward value
        """
        dist_alt = (self.state[0] - self._target_altitude) / (self.initial_state[0] - self._target_altitude)
        if self.success:
            time_scaled = self.time / self._max_time
            fpa_scaled = abs(self.state[4] / (89 * np.pi / 180))
            t_scale = 80
            fpa_scale = 20
            reward = t_scale * (1 - time_scaled) + fpa_scale * (1 - fpa_scaled)

        elif self.done:
            # reward = -10 * dist_alt - 10 * self.time / self.max_time
            reward = - dist_alt
        else:
            reward = 1 - dist_alt

        return reward

    def _reward4(self):
        """
        A potential reward method
        :return: reward value
        """
        dist_alt = (self.state[0] - self._target_altitude) / (self.initial_state[0] - self._target_altitude)
        if self.success:
            time_scaled = self.time / self._max_time
            fpa_scaled = abs(self.state[4] / (89 * np.pi / 180))
            t_scale = 80
            fpa_scale = 20
            reward = t_scale * (1 - time_scaled) + fpa_scale * (1 - fpa_scaled)

        elif self.done:
            # reward = -10 * dist_alt - 10 * self.time / self.max_time
            reward = - 100 * dist_alt
        else:
            reward = 1 - dist_alt

        return reward

    def _reward5(self):
        """
        A potential reward method
        :return: reward value
        """
        dist_alt = abs(self.state[0] - self._target_altitude) / (self.initial_state[0] - self._target_altitude)

        time_scaled = self.time / self._max_time
        fpa_scaled = abs(self.state[4] / (5 * np.pi / 180))
        t_scale = 80
        fpa_scale = 20
        alt_scale = 100

        if self.success:
            reward = t_scale * (1 - time_scaled) + fpa_scale * (1 - fpa_scaled)

        elif self.done:
            # reward = -10 * dist_alt - 10 * self.time / self.max_time
            reward = - alt_scale * dist_alt + (t_scale * (1 - time_scaled) + fpa_scale * (1 - fpa_scaled)) / 5
        else:
            reward = - dist_alt

        return reward

    def generate_emergency_descent_event(self, trigger_alt=None):
        """
        Generate event to stop integration when FPA hits zero below trigger altitude

        :param trigger_alt: altitude beneath which zero FPA will end propagation
        :return: emergency_descent_event
        """

        # If trigger_alt is not set, set to target altitude
        if trigger_alt is None:
            trigger_alt = self._target_altitude

        def emergency_descent_event(_, x):
            """
            Event to stop integration when FPA hits zero below trigger altitude

            Event triggers when a sign change occurs in the return value
            Therefore, returning -1 will prevent triggering when above trigger_alt
            FPA must be negative to cross trigger altitude so event won't trigger when crossing altitude

            :param _: time (not used)
            :param x: states
            :return: -1 if above trigger_alt, FPA if below
            """
            h, gam = x[0], x[4]

            if h > trigger_alt:
                return -1
            else:
                return gam

        # Add attributes to event function to end propagation when return values goes from negative to positive
        emergency_descent_event.terminal, emergency_descent_event.direction = True, 1

        return emergency_descent_event

    @abstractmethod
    def reset(self):
        """
        Reset current agent
        :return:
        """
        pass

    @abstractmethod
    def reward(self):
        """
        Generate reward for step
        :return:
        """
        pass

    @abstractmethod
    def _inner_step(self, action):
        """
        A overwrittable function that specifies application specific operations to perform within the RL step

        :param action: chosen action
        :return:
        """
        pass
