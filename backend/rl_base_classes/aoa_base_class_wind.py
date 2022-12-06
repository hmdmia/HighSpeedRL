import numpy as np
from backend.rl_base_classes.aoa_base_class import AoABaseClass
from backend.base_aircraft_classes.hgv_class_wind import HGV_Wind
from backend.rl_base_classes.rl_base_class import RLBaseClass


class AoABaseClassWind(AoABaseClass):
    def __init__(self, initial_state):

        self.initial_state = initial_state

        RLBaseClass.__init__(self)
        HGV_Wind.__init__(self, self.initial_state)

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