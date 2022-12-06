import numpy as np

from backend.rl_base_classes.rl_base_class import RLBaseClass
from backend.base_aircraft_classes.target_classes import MovingTarget
from backend.base_aircraft_classes.hgv_class import HGV
from backend.utils import circle_ang_dist, calc_bearing, wrap_ang
from backend.utils.coordinate_transform import lla_dist
from backend.base_aircraft_classes.hgv_class import re


class FPATrimsAndTurns(RLBaseClass, HGV):
    """
    Defines an HGV RL agent that has MPs of the FPA trim and turn set
    """

    def __init__(self, initial_state, target_location):
        self.initial_state = initial_state
        self.target_location = target_location
        self.initial_target_location = target_location

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

        self.generate_bounds_event()  # Use min/max to make integration events

        self._tar_fpa = initial_state[4]

        # Initial angular distance [deg] from vehicle to targets
        self._initial_ang_dist = circle_ang_dist(self.initial_state[1], self.initial_state[2],
                                                 self.target_location[1], self.target_location[2])
        initial_bear = calc_bearing(self.initial_state[1], self.initial_state[2], self.target_location[1],
                                    self.target_location[2])
        self._initial_bear_to_tar = wrap_ang(initial_bear - self.initial_state[5])

        # Observation space: [altitude, velocity, FPA, surface distance to target, bearing to target]

        # Lowest admissible values of nonnormalized observation space
        self.min_obs = np.array([self.min_alt, self.min_vel, self.min_gam, 0, -self.max_bear_to_tar])

        # Highest admissible values of nonnormalized observation space
        self.max_obs = np.array([self.max_alt, self.max_vel, self.max_gam, self.max_dist_to_tar, self.max_bear_to_tar])

        self.low = -np.ones(self.min_obs.shape)
        self.high = np.ones(self.max_obs.shape)

        self.max_dt = 25  # Max length of time for MPs to last
        self.min_dt = 0.5  # Min length of time for MPs to last
        self.dt = self.max_dt  # Length of time for MPs to last

        # Define available MPs ('type', amount)
        self.mp_options = [
            ('pull', np.deg2rad(0.5)),
            ('pull', np.deg2rad(0.)),
            ('pull', np.deg2rad(-0.5)),
            ('pull', np.deg2rad(-1)),
            ('pull', np.deg2rad(-1.5)),
            ('pull', np.deg2rad(-2)),
            ('turn', np.deg2rad(-5)),
            ('turn', np.deg2rad(-2.5)),
            ('turn', np.deg2rad(-1)),
            ('turn', np.deg2rad(-0.5)),
            ('turn', np.deg2rad(-0.1)),
            ('turn', np.deg2rad(0.1)),
            ('turn', np.deg2rad(0.5)),
            ('turn', np.deg2rad(1)),
            ('turn', np.deg2rad(2.5)),
            ('turn', np.deg2rad(5)),
        ]

        self.n_actions = len(self.mp_options)
        self.d_history = []
        self.bear_history = []

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

        return np.array([h, v, gam, dist_to_tar, self.bear_to_tar()])

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

    def reward_non_telescopic(self):
        """
        Return reward for each step

        :return: reward value for step
        """
        dist_rem = circle_ang_dist(self.state[1], self.state[2],
                                   self.target_location[1], self.target_location[2]) / self._initial_ang_dist

        vel_rem = (self.state[3] - self.min_vel) / (self.initial_state[3] - self.min_vel)

        if self.success:
            reward = 100 * (1 - dist_rem) + 100 * vel_rem

        elif self.done:
            reward = -10 * dist_rem + 10 * vel_rem

        else:
            bear = calc_bearing(self.state[1], self.state[2],
                                self.target_location[1], self.target_location[2])
            reward = - dist_rem - 10 * abs((wrap_ang(bear - self.initial_state[5]) / (np.pi / 4)))

        return reward

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

    def reward_telescopic_old(self):
        """
        Iteration on reward 8 including max reward

        :return: reward value for step
        """
        angle_scale = self.max_obs[2] - self.min_obs[2]  # use range of gamma as scale for angles
        altitude_scale = self.max_obs[0] - self.min_obs[0]  # use range of h as scale for altitude
        scale_array = np.array([altitude_scale, angle_scale, angle_scale])
        goal_state = np.array(self.target_location) / scale_array
        current_state = np.array(self.state[0:3]) / scale_array

        circ_ang_dist_tol = self.dist_tol / re
        tolerance_state = (np.array(self.target_location)
                           + np.array([self.alt_tol, circ_ang_dist_tol, 0])) / scale_array

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

    def update_dt(self):
        min_time_to_tar = self.cart_dist_to_tar() / self.state[3]
        dt = min(max(self.min_dt, min_time_to_tar), self.max_dt)

        time_to_event = self.time_to_event(dt)
        if self.min_dt < time_to_event < dt:
            self.dt = time_to_event
        else:
            self.dt = dt

    def _inner_step(self, action):
        """
        Inner step that execute MPs for action

        :param action: index of selected MP
        :return:
        """
        mp = self.mp_options[action]

        if mp[0] == 'pull' and abs(self.state[4] - mp[1]) > 1e-3:
            self._tar_fpa = mp[1]
            self.pull_up(self._tar_fpa)

            self.update_dt()
            _ti = self.time + self.dt

            self.sim_step(self.dt)
            self.record()

            if self.time + 1e-3 < _ti and not self.stop_integrating():
                self.fpa_trim(mp[1])
                self.sim_step(_ti - self.time)
                self.record()

        elif mp[0] == 'turn':
            self.turn(mp[1], self._tar_fpa)
            self.update_dt()
            _ti = self.time + self.dt

            self.sim_step(self.dt)
            self.record()

            if self.time + 1e-3 < _ti and not self.stop_integrating():
                self.fpa_trim()
                self.sim_step(_ti - self.time)
                self.record()

        elif not self.stop_integrating():
            self.fpa_trim(self._tar_fpa)
            self.update_dt()
            self.sim_step(self.dt)
            self.record()

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

    def record(self):
        bear_to_tar = []
        d = []
        y = self.state_history[-1]
        t = self.time_history[-1]
        for i in range(len(y[0, :])):
            tar_i = MovingTarget(np.concatenate((
                self.initial_target_location,
                (self.target.target_speed, 0, self.target.target_heading)
            )))
            tar_i.update_location(t[i])
            bear = calc_bearing(y[1, i], y[2, i], tar_i.target_location[1], tar_i.target_location[2])
            bear_to_tar.append(bear - y[5, i])
            d.append(circle_ang_dist(y[1, i], y[2, i],
                                     tar_i.target_location[1], tar_i.target_location[2]) * re)
        self.d_history.append(np.asarray(d))
        self.bear_history.append(np.asarray(bear_to_tar))

class MovingTargetFPATrimsAndTurns(FPATrimsAndTurns):
    def __init__(self, initial_state, target):
        super().__init__(initial_state, target.target_location)
        self.target = target

        # Expand observation to include the target's velocity and bearing to HGV
        self.low = np.append(self.low, (-1, -1))
        self.high = np.append(self.high, (1, 1))
        self.min_obs = np.append(self.min_obs, (0, -np.pi))
        self.max_obs = np.append(self.max_obs, (self.max_vel, np.pi))

    def _observe(self):
        observables = super()._observe()
        theta = self.state[1]
        phi = self.state[2]
        tar_velocity = self.target.target_speed
        tar_head = self.target.target_heading
        tar_bear = calc_bearing(self.target_location[1], self.target_location[2], theta, phi)
        tar_bear_to_hgv = wrap_ang(tar_bear - tar_head)
        return np.append(observables, (tar_velocity, tar_bear_to_hgv))

    def reset(self, initial_state=None):
        super().reset(initial_state)
        self.target.reset_location()
        self.target_location = self.target.target_location

    def _inner_step(self, action):
        super()._inner_step(action)
        self.target.update_location(self.dt)
        self.target_location = self.target.target_location


class RandomStartFPATrimsAndTurns(FPATrimsAndTurns):
    def __init__(self, nominal_initial_state, nominal_target_location, initial_variance, target_variance):

        self.nominal_initial_state = nominal_initial_state
        self.nominal_target_location = nominal_target_location
        self.initial_variance = initial_variance
        self.target_variance = target_variance

        initial_state = self.randomize_state(self.nominal_initial_state, self.initial_variance)
        target_location = self.randomize_state(self.nominal_target_location, self.target_variance)

        super().__init__(initial_state, target_location)

        self.target_tol = 50000.
        self.reward = self._reward_telescopic
        self._last_reward = 0
        self._last_reward = self.reward()

    @staticmethod
    def randomize_state(nominal, variance):
        state = nominal + np.random.random(len(variance)) * variance * 2 - variance
        return state

    def reset(self, initial_state=None, target_location=None):
        """
        Resets agent at end of episode

        :param initial_state: initial state to start reset agent
        :param target_location: location for vehicle to target
        :return:
        """
        self.time = 0

        # If initial state not defined, last initial state used
        if initial_state is None:
            initial_state = self.randomize_state(self.nominal_initial_state, self.initial_variance)

        if target_location is None:
            target_location = self.randomize_state(self.nominal_target_location, self.target_variance)

        self.initial_state = initial_state
        self.state = self.initial_state

        self.target_location = target_location

        self._initial_ang_dist = circle_ang_dist(self.initial_state[1], self.initial_state[2],
                                                 self.target_location[1], self.target_location[2])

        self._last_reward = 0
        self._last_reward = self.reward()

        self.time_history = []
        self.state_history = []
        self.control_history = []

        self.reward_total = 0

        self.done, self.success = False, False

    def _reward(self):
        """
        Return reward for each step

        :return: reward value for step
        """
        dist_rem = circle_ang_dist(self.state[1], self.state[2],
                                   self.target_location[1], self.target_location[2]) / self._initial_ang_dist

        vel_rem = (self.state[3] - self.min_vel) / (self.initial_state[3] - self.min_vel)

        bear = calc_bearing(self.state[1], self.state[2],
                            self.target_location[1], self.target_location[2])

        if self.success:
            reward = 1000 * (1 - dist_rem) + 1000 * vel_rem

        elif self.done:
            reward = -1000 * dist_rem

        else:
            reward = 10 - dist_rem - abs((bear - self.state[5]) / np.deg2rad(1))

        return reward

    def _reward_telescopic(self):
        """
        Return reward for each step

        :return: reward value for step
        """
        dist_rem = circle_ang_dist(self.state[1], self.state[2],
                                   self.target_location[1], self.target_location[2]) / self._initial_ang_dist

        vel_rem = (self.state[3] - self.min_vel) / (self.initial_state[3] - self.min_vel)

        bear = calc_bearing(self.state[1], self.state[2],
                            self.target_location[1], self.target_location[2])

        if self.success:
            reward = 10 * (1 - dist_rem) + 10 * vel_rem

        elif self.done:
            reward = - dist_rem * 10

        else:
            reward_i = - dist_rem - ((bear - self.state[5]) / np.deg2rad(5)) ** 2 / 5
            reward = reward_i - self._last_reward
            self._last_reward = reward_i

        return reward


if __name__ == '__main__':
    '''
    This is to test and compare MPs
    '''
    import matplotlib.pyplot as plt

    # MPs to run (incremented down from DAF runner)
    mps = [2, 4, 2, 6, 10, 7, 10, 8, 10, 5]

    y0 = np.array([40_000, 0, 0, 6_000, 0, 0])

    hgv = FPATrimsAndTurns(y0, np.array([0, np.pi, 0.]))

    hgv.mp_options = [
        ('turn', np.deg2rad(-10)),
        ('turn', np.deg2rad(-5)),
        ('turn', np.deg2rad(-2.5)),
        ('turn', np.deg2rad(-1)),
        ('pull', np.deg2rad(0.5)),
        ('pull', np.deg2rad(0.)),
        ('pull', np.deg2rad(-0.5)),
        ('pull', np.deg2rad(-1)),
        ('pull', np.deg2rad(-1.5)),
        ('pull', np.deg2rad(-2)),
        ('turn', np.deg2rad(1)),
        ('turn', np.deg2rad(2.5)),
        ('turn', np.deg2rad(5)),
        ('turn', np.deg2rad(10))
    ]

    for current_mp in mps:
        hgv.rl_step(current_mp)

    hgv.export_history(filename='mp_comp', export_format='mat')
    hgv.export_history(filename='mp_comp', export_format='csv')

    _t = np.concatenate(hgv.time_history)
    _y = np.hstack(hgv.state_history)
    _u = np.hstack(hgv.control_history)

    _h = _y[0, :] / 1000
    _theta = np.rad2deg(_y[1, :])
    _phi = np.rad2deg(_y[2, :])
    _v = _y[3, :]
    _gam = np.rad2deg(_y[4, :])
    _psi = np.rad2deg(_y[5, :])

    _alpha = np.rad2deg(_u[0, :])
    _sigma = np.rad2deg(_u[1, :])

    fig1 = plt.figure(1, figsize=(6, 6))
    plt.plot(_t, _psi, linestyle='-', label=r'$\psi$', color='C0')
    plt.plot(_t, _gam, linestyle='-', label=r'$\gamma$', color='C3')
    plt.plot(_t, _theta, linestyle='--', label=r'$\theta$', color='C1')
    plt.plot(_t, _phi, linestyle='--', label=r'$\phi$', color='C4')
    plt.plot(_t, _alpha, linestyle=':', label=r'$\alpha$', color='C2')
    plt.plot(_t, _sigma, linestyle=':', label=r'$\sigma$', color='C9')
    plt.legend()
    plt.title('MP series {}'.format([mp + 1 for mp in mps]))
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')
    plt.grid()
    plt.tight_layout()

    fig2 = plt.figure(2, figsize=(6, 6))
    gs2 = plt.GridSpec(3, 2, figure=fig2)

    fig2.suptitle('MP series {}'.format([mp + 1 for mp in mps]))

    ax2_00 = fig2.add_subplot(gs2[0, 0])
    ax2_00.plot(_t, _psi, linestyle='-', label=r'$\psi$', color='C0')
    ax2_00.set_xlabel('Time [s]')
    ax2_00.set_ylabel(r'Heading, $\psi$ [deg]')

    ax2_01 = fig2.add_subplot(gs2[0, 1])
    ax2_01.plot(_t, _gam, linestyle='-', label=r'$\gamma$', color='C3')
    ax2_01.set_xlabel('Time [s]')
    ax2_01.set_ylabel(r'FPA, $\gamma$ [deg]')

    ax2_10 = fig2.add_subplot(gs2[1, 0])
    ax2_10.plot(_t, _theta, linestyle='--', label=r'$\theta$', color='C1')
    ax2_10.set_xlabel('Time [s]')
    ax2_10.set_ylabel(r'Downrange, $\theta$ [deg]')

    ax2_11 = fig2.add_subplot(gs2[1, 1])
    ax2_11.plot(_t, _phi, linestyle='--', label=r'$\phi$', color='C4')
    ax2_11.set_xlabel('Time [s]')
    ax2_11.set_ylabel(r'Crossrange, $\phi$ [deg]')

    ax2_20 = fig2.add_subplot(gs2[2, 0])
    ax2_20.plot(_t, _alpha, linestyle=':', label=r'$\alpha$', color='C2')
    ax2_20.set_xlabel('Time [s]')
    ax2_20.set_ylabel(r'AoA, $\alpha$ [deg]')

    ax2_21 = fig2.add_subplot(gs2[2, 1])
    ax2_21.plot(_t, _sigma, linestyle=':', label=r'$\sigma$', color='C9')
    ax2_21.set_xlabel('Time [s]')
    ax2_21.set_ylabel(r'Bank, $\sigma$ [deg]')

    plt.tight_layout()

    fig3 = plt.figure(3, figsize=(6, 6))
    gs3 = plt.GridSpec(3, 1, figure=fig3)

    fig3.suptitle('MP series {}'.format([mp + 1 for mp in mps]))

    ax3_0 = fig3.add_subplot(gs3[0, 0])
    ax3_0.plot(_theta, _phi)
    ax3_0.set_xlabel('Downrange [deg]')
    ax3_0.set_ylabel('Crossrange [deg]')

    ax3_1 = fig3.add_subplot(gs3[1, 0])
    ax3_1.plot(_t, _h)
    ax3_1.set_xlabel('Time [s]')
    ax3_1.set_ylabel('Altitude [km]')

    ax3_2 = fig3.add_subplot(gs3[2, 0])
    ax3_2.plot(_t, _v)
    ax3_2.set_xlabel('Time [s]')
    ax3_2.set_ylabel('Velocity [m/s]')

    plt.show()
