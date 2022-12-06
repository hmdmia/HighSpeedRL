from typing import Sequence, Optional, Union
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from backend.rl_base_classes.rl_base_class import RLBaseClass
from backend.base_aircraft_classes.hgv_class import HGV
from backend.utils import circle_ang_dist, calc_bearing, wrap_ang
from backend.utils.coordinate_transform import lla_dist
from backend.base_aircraft_classes.hgv_class import re


class MovingTargetNonMP(RLBaseClass, HGV):
    """
    Defines an HGV RL agent that has MPs of the FPA trim and turn set
    """

    def __init__(self, initial_state, target,
                 num_actions: Optional[Union[int, dict, Sequence]] = None, dt: int = 1, use_box: bool = False):
        self.initial_state = initial_state
        self.target = target
        self.target_location = self.target.target_location
        self.dt = dt  # Length of time for actions
        self.use_box = use_box

        RLBaseClass.__init__(self)
        HGV.__init__(self, self.initial_state)

        self.done, self.success = False, False
        self.lastRewardTime = self.time  # Last time reward calculated, used for variable time step reward calc's
        self.old_reward = 0

        self.alt_tol = 1000  # tolerance for difference between HGV and target altitude [m]
        self.dist_tol = 5000.  # tolerance for circular distance to target [m]
        self.max_time = 800.

        self.min_alt = 20000
        self.max_alt = 75000
        self.min_vel = 2000
        self.max_vel = 6000
        self.max_dist_to_tar = re * np.pi * 1.05
        self.min_gam = np.deg2rad(-10)
        self.max_gam = np.deg2rad(5)
        self.max_bear_to_tar = np.pi / 2
        self.min_tar_velocity = 0
        self.max_tar_velocity = self.max_vel
        self.max_tar_bear_to_hgv = np.pi

        self.generate_bounds_event()  # Use min/max to make integration events

        self._tar_fpa = initial_state[4]

        # Observation space: [altitude, velocity, FPA, surface distance to target, bearing to target,
        # target velocity, target bearing to HGV]

        # Lowest admissible values of non-normalized observation space
        self.min_obs = np.array([self.min_alt, self.min_vel, self.min_gam, 0, -self.max_bear_to_tar,
                                 self.min_tar_velocity, -self.max_tar_bear_to_hgv])

        # Highest admissible values of non-normalized observation space
        self.max_obs = np.array([self.max_alt, self.max_vel, self.max_gam, self.max_dist_to_tar, self.max_bear_to_tar,
                                 self.max_tar_velocity, self.max_tar_bear_to_hgv])

        self.low = -np.ones(self.min_obs.shape)
        self.high = np.ones(self.max_obs.shape)

        self.min_aoa = np.deg2rad(0)
        self.max_aoa = np.deg2rad(20)
        self.min_bank = np.deg2rad(-75)
        self.max_bank = np.deg2rad(75)

        self.num_ctrl = 2
        if self.use_box:
            def action2control(action):
                return action[0] * ((self.max_aoa - self.min_aoa) + (self.max_aoa + self.min_aoa))/2,\
                       action[1] * ((self.max_bank - self.min_bank) + (self.max_bank + self.min_bank))/2
            self.action2control = action2control
        else:
            # Define available MPs ('type', amount)
            if num_actions is None:
                num_actions = {'aoa': 20, 'bank': 3}
            if isinstance(num_actions, int):
                self.aoa_options = np.deg2rad(np.linspace(-20, 20, num_actions))
                self.bank_options = np.deg2rad(np.linspace(-75, 75, num_actions))
            elif isinstance(num_actions, dict):
                self.aoa_options = np.deg2rad(np.linspace(-20, 20, num_actions['aoa']))
                self.bank_options = np.deg2rad(np.linspace(-75, 75, num_actions['bank']))
            elif isinstance(num_actions, Sequence):
                self.aoa_options = np.deg2rad(np.linspace(-20, 20, num_actions[0]))
                self.bank_options = np.deg2rad(np.linspace(-75, 75, num_actions[1]))

            self.n_actions = (len(self.aoa_options), len(self.bank_options))

            def action2control(action):
                return self.aoa_options[action[0]], self.bank_options[action[1]]
            self.action2control = action2control

        self.bear_history = []
        self.d_history = []

    def bear_to_tar(self):
        bear = calc_bearing(self.state[1], self.state[2], self.target_location[1], self.target_location[2])
        _bear_to_tar = wrap_ang(bear - self.state[5])
        return _bear_to_tar

    def dist_to_tar(self):
        _dist_to_tar = circle_ang_dist(self.state[1], self.state[2],
                                       self.target_location[1], self.target_location[2]) * re
        return _dist_to_tar

    def cart_dist_to_tar(self):
        _cart_dist_to_tar = lla_dist(self.state[1], self.state[2], self.state[0],
                                     self.target_location[1], self.target_location[2], self.target_location[0],
                                     radius_earth=re)
        return _cart_dist_to_tar

    def alt_to_tar(self):
        _alt_to_tar = self.state[0] - self.target_location[0]
        return _alt_to_tar

    def _observe(self):
        """
        Method to form observation (nonnormalized)
        [altitude, velocity, FPA, distance to target, difference between heading and bearing to target]

        :return: nonnormalized observation vector
        """
        h, theta, phi, v, gam, psi = self.state

        dist_to_tar = self.dist_to_tar()
        alt_to_tar = self.alt_to_tar()

        # Define success condition
        if dist_to_tar <= self.dist_tol and abs(alt_to_tar) <= self.alt_tol:
            self.success, self.done = True, True

        # Define conditions to end episode without success
        else:
            self.done = self.stop_integrating()

        tar_velocity = self.target.target_speed
        tar_head = self.target.target_heading
        tar_bear = calc_bearing(self.target_location[1], self.target_location[2], theta, phi)
        tar_bear_to_hgv = wrap_ang(tar_bear - tar_head)

        return np.array((h, v, gam, dist_to_tar, self.bear_to_tar(), tar_velocity, tar_bear_to_hgv))

    def observe(self):
        """
        Return normalized observations
        :return: normalized observation vector [-1, 1]
        """
        observation = self._observe()
        normalized_observation = (2 * observation - self.max_obs - self.min_obs) / (self.max_obs - self.min_obs)
        return normalized_observation

    def reset(self, initial_state=None):
        """
        Resets agent at end of episode

        :param initial_state: initial state to start reset agent
        :return:
        """
        self.time = 0
        self.lastRewardTime = 0
        self.old_reward = 0

        # If initial state not defined, last initial state used
        if initial_state is None:
            self.state = self.initial_state
        else:
            self.initial_state = initial_state
            self.state = self.initial_state

        self._tar_fpa = self.initial_state[4]

        self.time_history = []
        self.state_history = []
        self.control_history = []

        self.reward_total = 0

        self.done, self.success = False, False

        self.target.reset_location()
        self.target_location = self.target.target_location

    def reward_telescopic(self):
        """
        Iteration of telescopic reward, replacing h, theta, phi with alt_to_tar, dist_to_tar, bear_to_tar

        :return: reward value for step
        """
        h_scale = self.max_alt - self.min_alt  # use range of h as scale for altitude
        dist_scale = self.max_dist_to_tar - 0  # use range of dist_to_tar
        bear_scale = self.max_bear_to_tar - -self.max_bear_to_tar  # use range of bear_to_tar
        scale_array = np.array((h_scale, dist_scale, bear_scale))
        goal_state = np.zeros((3,)) / scale_array
        current_state = np.array((self.alt_to_tar(), self.dist_to_tar(), self.bear_to_tar())) / scale_array
        tolerance_state = np.array((self.alt_tol, self.dist_tol, 0)) / scale_array
        max_reward = 1 / np.linalg.norm(goal_state - tolerance_state, 2)
        new_reward = min(max_reward, 1 / np.linalg.norm(goal_state - current_state, 2))
        success_bonus = 200
        reward = (new_reward - self.old_reward) + success_bonus * self.success
        self.old_reward = new_reward
        return reward

    def reward(self):
        return self.reward_telescopic()

    def stop_integrating(self) -> bool:
        boolean = super().stop_integrating()
        self.done = boolean
        return boolean

    def _inner_step(self, action):
        """
        Inner step that execute MPs for action

        :param action: index of selected MP
        :return:
        """
        aoa, bank = self.action2control(action)
        self.constant_step(self.dt, (aoa, bank))

        _t0, _x0 = self.time, self.state
        sol = solve_ivp(self.dynamics, [_t0, _t0 + self.dt], _x0, events=self._get_terminal_events(),
                        max_step=self.max_step, method=self.odemethod)
        self.time_history.append(sol.t)
        self.state_history.append(sol.y)
        u = self.compute_control_series(sol.t, sol.y)
        self.control_history.append(u)
        self.time, self.state, self.control = sol.t[-1], sol.y[:, -1], u[:, -1]

        self.target.update_location(self.dt)
        self.target_location = self.target.target_location

        bear_to_tar = []
        d = []
        for i in range(len(sol.y[0, :])):
            tar_i = deepcopy(self.target)
            tar_i.reset_location()
            tar_i.update_location(sol.t[i])
            bear = calc_bearing(sol.y[1, i], sol.y[2, i], tar_i.target_location[1], tar_i.target_location[2])
            bear_to_tar.append(bear - sol.y[5, i])
            d.append(circle_ang_dist(sol.y[1, i], sol.y[2, i],
                                     tar_i.target_location[1], tar_i.target_location[2]) * re)
        self.d_history.append(np.asarray(d))
        self.bear_history.append(np.asarray(bear_to_tar))

    def generate_bounds_event(self):
        """
        Generates event to end propagation when min/max bounds violated, appends it to training events.
        """

        terminal_bounds_events = [lambda _, x: x[0] - self.min_alt,
                                  lambda _, x: x[3] - self.min_vel,
                                  lambda _, x: x[4] - self.min_gam,
                                  lambda _, x: self.max_bear_to_tar - abs(self.bear_to_tar())]

        for i in range(len(terminal_bounds_events)):
            terminal_bounds_events[i].terminal = True
            terminal_bounds_events[i].direction = -1  # track positive to negative

        self.training_events.extend(terminal_bounds_events)
