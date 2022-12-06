import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from abc import ABC, abstractmethod
from typing import Callable, Iterable


class VehicleBase(ABC):
    """
    Abstract vehicle base for simulation.
    Intended to define the interfaces of an agent which can be simulated
    """
    def __init__(self, initial_state: np.array, initial_time: float = 0.):
        """
        Initialize the vehicle base class

        :param initial_state: Array of initial states (dynamic variables) to start simulation
        :param initial_time: Time to start simulation (most likely default 0)
        """

        # n is length of state vector
        if self.n is None:
            self.n = len(initial_state)

        # m is length of control vector. Intended to be defined in child class prior to running VehicleBase.__init__()
        if self.m is None:
            self.m = 0

        # self.compute_control returns the control vectol based on time and states
        self.compute_control = self.null_control

        self.time = initial_time
        self.state = np.array(initial_state)
        self.control = np.zeros(self.m)

        # Hard limits by which the control may be clipped
        self.control_bounds = np.array([
            [-np.inf] * self.m,
            [np.inf] * self.m
        ])

        # For storage of vehicle history for later plotting
        self.time_history = []
        self.state_history = []
        self.control_history = []

        # Max simulation step (to give to solve_ivp)
        self.max_step = 100.

        # Events which in simulation step
        self.constraint_events = []  # for constraint violations
        self.primitive_events = []   # to end current MP
        self.training_events = []    # to end RL training

        self.func_stop = None
        self.constraint_functions = []
        self.odemethod = 'RK45'

    @abstractmethod
    def _eom_func(self, t: float, x: np.array, u: np.array) -> np.array:
        """
        Define equations of motion (dx/dt) for vehicle as function of time, state, and control.

        Called within dynamics() with allows for _eom_func to be overwritten by child classes without affecting generic
        control calls.

        :param t: time
        :param x: state vector
        :param u: control vector
        :return: time derivative of x
        """
        pass

    def dynamics(self, t, x) -> np.array:
        """
        Dynamic function for scipy.integrate.solve_ivp()
        :param t: time
        :param x: state vector
        :return: time derivative of x
        """
        u = self.compute_control(t, x)
        x_dot = self._eom_func(t, x, u)
        return x_dot

    def bound_controls(self, u: np.array) -> np.array:
        """
        Clips control vector to lie within region defined by self.control_bounds
        :param u: controls
        :return: bounded controls
        """
        return np.clip(u, self.control_bounds[0], self.control_bounds[1])

    def null_control(self, _: np.array, __: np.array) -> np.array:
        """
        Return control vector filled with 0

        :param _: time (not needed)
        :param __: states (not needed)
        :return: controls of zeros
        """
        return np.zeros(self.m)

    def load_control_profile(self, t_data: np.array, u_data: np.array):
        """
        Generates control method prescribing a given control profile directly.

        PCHIP interpolates control between sample points.

        :param t_data: vector of time at sample points
        :param u_data: vector of control vectors at sample points
        :return:
        """
        self.primitive_events = self.generate_terminal_time_event(t_data[-1])

        interpolator = PchipInterpolator(t_data, u_data)

        def get_control_from_reference(t: float, _: np.array) -> np.array:
            """
            Function to return control from given profile

            :param t: time
            :param _: states (not used)
            :return: controls
            """
            return interpolator(t).T

        self.compute_control = get_control_from_reference

    def constant_step(self, dt: float, u_step: np.array):
        """
        Generate MP to specify a given control (u_step) for a specified time (dt)

        Used in emergency descent problem indicated as absolute

        :param dt: time for step/MP to last
        :param u_step: control to use for step/MP
        :return:
        """

        t_cur = self.time

        def _constant_step(_, __):
            """
            Function to return specified control for time step

            :param _: time (not used)
            :param __: state (not used)
            :return: controls
            """
            return u_step

        self.compute_control = _constant_step
        self.primitive_events = self.generate_terminal_time_event(t_cur + dt)

    def delta_step(self, dt: float, du: np.array):
        """
        Generate MP to specify a given control change from previous interval (du) for a specified time (dt)

        Used in emergency descent problem indicated as relative

        :param dt: time for step/MP to last
        :param du: change in controls from last interval
        :return:
        """

        t_cur = self.time

        u_prev = self.control
        u_cur = u_prev + du

        def _delta_step(_, __):
            """
            Function to return specified control for time step

            :param _: time (not used)
            :param __: state (not used)
            :return: controls
            """
            return u_cur

        self.compute_control = _delta_step
        self.primitive_events = self.generate_terminal_time_event(t_cur + dt)

    def linear_change(self, dt, u_tar):
        """
        Generate MP to specify a given control linear ramp from previous interval to u_tar for a specified time (dt)

        Used in emergency descent problem indicated as relative

        :param dt: time for step/MP to last
        :param u_tar: controls at end of interval
        :return:
        """

        t_cur = self.time
        t_tar = self.time + dt

        self.primitive_events = self.generate_terminal_time_event(t_tar)

        if len(self.control_history) > 0:
            u_cur = self.control_history[-1].T[-1]
        else:
            u_cur = u_tar

        def lin_step(t: float, _: np.array) -> np.array:
            """
            Function to return linear control ramp for time step
            :param t: time
            :param _: state (not used)
            :return: controls
            """
            if t < t_cur:
                return u_cur
            elif t > t_tar:
                return u_tar
            else:
                return u_cur + (u_tar - u_cur) * (t - t_cur) / dt

        self.compute_control = lin_step

    @staticmethod
    def generate_terminal_time_event(t_end: float) -> Callable[[float, np.array], float]:
        """
        Generates event to end propagation at specified time

        :param t_end: time to end propagation
        :return: event function for solve_ivp
        """
        def terminal_time_event(t, _):
            """
            Event to end propagation at specified time
            :param t: time
            :param _: state (not used)
            :return: difference between current time and end time
            """
            return t - t_end

        # Add attributes to event function to end propagation when return values goes from negative to positive
        terminal_time_event.terminal, terminal_time_event.direction = True, 1

        return terminal_time_event

    def _get_terminal_events(self) -> Iterable[Callable]:
        """
        Combines lists of terminal events to feed to IVP solver (solve_ivp())

        :return: List of events for solve_ivp()
        """

        _events = []
        for _event_list in [self.constraint_events, self.training_events, self.primitive_events]:
            if isinstance(_event_list, Iterable):
                _events += list(_event_list)
            else:
                _events.append(_event_list)

        return _events

    def stop_integrating(self) -> bool:
        """
        Checks each non-MP event to see if constraint or training event has been triggered (and thus integration and
        training should stop)

        :return: Boolean, True if terminal event is triggered
        """
        terminal_events = self.constraint_events.copy()
        terminal_events.extend(self.training_events)

        for event in terminal_events:
            if event(self.time, self.state) < 1e-3:
                return True

        return False

    @abstractmethod
    def plot_state_history(self, style=None, size=None, title=None):
        """
        Method to plot history of agent simulations

        :param style:
        :param size:
        :param title:
        :return:
        """
        pass

    def compute_control_series(self, _t: np.array, _y: np.array) -> np.array:
        """
        Calculate control for propagated trajectory for record and plotting.

        Required to recompute unfortunately because of restrictions of solve_ivp()

        :param _t: time vector
        :param _y: state vector (over time vector)
        :return: control vector
        """

        u = np.array([self.compute_control(_t_k, _y_k) for _t_k, _y_k in zip(_t, _y.T)]).T
        return u

    def sim_step(self, dt: float) -> [float, np.array]:
        """
        Run simulation of vehicle for a time interval dt

        Runs solve_ivp() with appropriate arguments and record data

        :param dt: time to simulate vehicle
        :return: time and state vector
        """
        _t0, _x0 = self.time, self.state

        sol = solve_ivp(self.dynamics, [_t0, _t0 + dt], _x0, events=self._get_terminal_events(),
                        max_step=self.max_step, method=self.odemethod)

        self.time_history.append(sol.t)
        self.state_history.append(sol.y)

        u = self.compute_control_series(sol.t, sol.y)
        self.control_history.append(u)

        time, state, control = sol.t[-1], sol.y[:, -1],  u[:, -1]
        self.time = time
        self.state = state
        self.control = control

        return time, state

    def time_to_event(self, dt: float) -> float:
        """
        :param dt: time to simulate vehicle
        :return: change in time until sim_step would trigger an event
        """
        _t0, _x0 = self.time, self.state
        sol = solve_ivp(self.dynamics, [_t0, _t0 + dt], _x0, events=self._get_terminal_events(),
                        max_step=self.max_step, method=self.odemethod)
        return sol.t[-1] - _t0

    def export_history(self, filename='data', export_format='csv'):
        """
        Create a data file from agent stored history

        :param filename: name of output file
        :param export_format: type of data output
        :return:
        """

        t = np.concatenate(self.time_history)
        y = np.hstack(self.state_history)
        u = np.hstack(self.control_history)

        if export_format == 'csv':
            import csv

            data = np.vstack((t, y, u))

            with open(filename + '.csv', 'w') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(data)

        elif export_format == 'npz':
            np.savez(filename, t, y, u)

        elif export_format == 'mat':
            from scipy.io import savemat

            out_dict = {'t': t, 'y': y, 'u': u}
            savemat(filename + '.mat', out_dict)

        else:
            raise NotImplementedError('Export format {} not implemented'.format(export_format))
