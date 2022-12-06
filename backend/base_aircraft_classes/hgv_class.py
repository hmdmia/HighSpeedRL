import numpy as np
from math import sin, cos, tan, exp

from backend.utils.standard_atmosphere import StandardAtmosphere, calc_speed_of_sound

from .base_class import VehicleBase

dtype = np.float64

# Define Aerodynamic Coefficients
cl0, cl1, cl2, cl3 = -0.2317, 0.0513 * 180 / 3.14159, 0.2945, -0.1028
cd0, cd1, cd2, cd3 = 0.024, 7.24e-4 * 180 ** 2 / 3.14159 ** 2, 0.406, -0.323

a_ref = 0.4839  # HGV reference area [m**2]
mass = 907.20  # HGV mass [kg]
re = 6378000  # Radius of Earth [m]
mu = 3.986e14  # Earth gravitational constant [N/m**2]


class HGV(VehicleBase):
    """
    Class to simulate hypersonic glide vehicle
    """

    def __init__(self, initial_state, initial_time=0.):
        """
        Initialize the HGV class

        :param initial_state: Array of initial states (dynamic variables) to start simulation
        :param initial_time: Time to start simulation (most likely default 0)
        """

        self.n = 6
        self.m = 2

        VehicleBase.__init__(self, initial_state, initial_time=initial_time)

        self.atmosphere_func = StandardAtmosphere().calc_values
        self.constraint_events.append(self.hit_ground_event)

        self.control_bounds = np.array([
            [-20 * np.pi / 180, -89 * np.pi / 180],
            [20 * np.pi / 180, 89 * np.pi / 180]
        ])

    @staticmethod
    def compute_aero(alpha, mach):
        """
        Computes lift coefficient (cl) and drag coefficient (cd) for HGV

        :param alpha: angle of attack
        :param mach: vehicle mach number (velocity/speed of sound)
        :return: lift coefficient, drag  coefficient
        """
        cl = cl1 * alpha + cl2 * exp(cl3 * mach) + cl0
        cd = cd1 * alpha ** 2 + cd2 * exp(cd3 * mach) + cd0
        return cl, cd

    def _eom_func(self, t, x, u):
        """
        Equations of motion function for HGV.
        Called within dynamics() with allows for _eom_func to be overwritten by child classes without affecting generic
        control calls.

        :param t: time
        :param x: states
        :param u: controls
        :return: time derivative of x
        """

        alpha, sigma = u
        h, theta, phi, v, gam, psi = x

        temp, _, rho = self.atmosphere_func(h)
        cl, cd = self.compute_aero(alpha, v / calc_speed_of_sound(temp))

        r = re + h
        q = 0.5 * rho * v ** 2
        lift, drag = q * cl * a_ref, q * cd * a_ref

        x_dot = [
            v * sin(gam),
            v * cos(gam) * cos(psi) / r,
            v * cos(gam) * sin(psi) / (r * cos(theta)),
            -drag / mass - mu * sin(gam) / r ** 2,
            lift * cos(sigma) / (mass * v) - mu / (v * r ** 2) * cos(gam) + v / r * cos(gam),
            lift * sin(sigma) / (mass * cos(gam) * v) + v / r * cos(gam) * sin(psi) * tan(theta)
        ]

        return x_dot

    def plot_state_history(self, style='segmented', size=None, title=None):
        """
        Plot state history of vehilce through matplotlib.

        :param style: which collection of plots to show
        :param size: size of plot figure with predefined options
        :param title: title for collection of plots
        :return:
        """

        # Imports matplotlib only when needed to plot
        import matplotlib.pyplot as plt

        # Sizes ppt and half_ppt are set to fit on a PowerPoint slide and half a slide respectively
        if isinstance(size, tuple):
            figsize = size
        elif size == 'ppt':
            figsize = (12.58, 5.15)
        elif size == 'half_ppt':
            figsize = (6.15, 4.75)
        else:
            figsize = None

        fig = plt.figure(figsize=figsize)

        if style == 'planar':
            t = np.concatenate(self.time_history)
            y = np.hstack(self.state_history)
            u = np.hstack(self.control_history)

            ax21 = fig.add_subplot(211)
            ax21.plot(y[1, :] * 180 / np.pi, y[0, :] / 1000)
            ax21.set_xlabel('Downrange [deg]')
            ax21.set_ylabel('Altitude [km]')

            ax22 = fig.add_subplot(212)
            ax22.plot(t, u[0, :] * 180 / np.pi)
            ax22.set_xlabel('Time [s]')
            ax22.set_ylabel('Angle of Attack [deg]')
            ax22.set_title('Control History')

        elif style == 'segmented':

            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            for ti, yi, ui in zip(self.time_history, self.state_history, self.control_history):
                ax1.plot(yi[1, :] * 180 / np.pi, yi[0, :] / 1000)
                ax1.set_xlabel('Downrange [deg]')
                ax1.set_ylabel('Altitude [km]')
                ax1.set_title('Trajectory')

                ax2.plot(ti, ui[0, :] * 180 / np.pi)
                ax2.set_xlabel('Time [s]')
                ax2.set_ylabel('Angle of Attack [deg]')
                ax2.set_title('Control History')

                ax3.plot(ti, yi[4, :] * 180 / np.pi)
                ax3.set_xlabel('Time [s]')
                ax3.set_ylabel('Flight Path Angle [deg]')
                ax3.set_title('Flight Path Angle')

        elif style == '3d':

            ax1 = fig.add_subplot(231, projection='3d')
            ax2 = fig.add_subplot(234)
            ax3 = fig.add_subplot(232)
            ax4 = fig.add_subplot(233)
            ax5 = fig.add_subplot(235)
            ax6 = fig.add_subplot(236)

            for ti, yi, ui in zip(self.time_history, self.state_history, self.control_history):
                ax1.plot(yi[1, :] * 180 / np.pi, yi[2, :] * 180 / np.pi, yi[0, :] / 1000)
                ax1.set_xlabel('Downrange [deg]')
                ax1.set_ylabel('Crossrange [deg]')
                ax1.set_zlabel('Altitude [km]')
                ax1.set_title('Trajectory')

                ax2.plot(yi[3, :] / 1000, yi[0, :] / 1000)
                ax2.set_xlabel('Velocity [km/s]')
                ax2.set_ylabel('Altitude [km]')
                ax2.set_title('h-v Diagram')

                ax3.plot(ti, yi[4, :] * 180 / np.pi)
                ax3.set_xlabel('Time [s]')
                ax3.set_ylabel('Flight Path Angle [deg]')
                ax3.set_title('Flight Path Angle')

                ax4.plot(ti, yi[5, :] * 180 / np.pi)
                ax4.set_xlabel('Time [s]')
                ax4.set_ylabel('Heading Angle [deg]')
                ax4.set_title('Heading Angle')

                ax5.plot(ti, ui[0, :] * 180 / np.pi)
                ax5.set_xlabel('Time [s]')
                ax5.set_ylabel('Angle of Attack [deg]')
                ax5.set_title('Angle of Attack History')

                ax6.plot(ti, ui[1, :] * 180 / np.pi)
                ax6.set_xlabel('Time [s]')
                ax6.set_ylabel('Bank Angle [deg]')
                ax6.set_title('Bank Angle History')

        if title is not None:
            plt.suptitle(title)

        plt.show()

    @staticmethod
    def hit_ground_event(_, x):
        """
        Integration event to stop integration when vehicle hits the ground
        :param _: time (not used)
        :param x: states
        :return: altitude
        """
        return x[0]

    # Add attributes to event function to end propagation when return values goes from positive to negative
    hit_ground_event.terminal, hit_ground_event.direction = True, -1

    @staticmethod
    def generate_fpa_event(fpa_tar):
        """
        Generate event to stop integration when FPA equals fpa_tar
        :param fpa_tar: target flight path angle
        :return: terminal_fpa_event function
        """

        def terminal_fpa_event(_, _x):
            """
            Event to stop integration when FPA equals fpa_tar
            :param _: time (not used)
            :param _x: states
            :return: difference between current and target FPA
            """
            return _x[4] - fpa_tar

        # Add attributes to event function to end propagation when return values crosses zero
        terminal_fpa_event.terminal, terminal_fpa_event.direction = True, 0

        return terminal_fpa_event

    @staticmethod
    def generate_head_event(head_tar):
        """
        Generate event to stop integration when heading equals head_tar
        :param head_tar: target heading angle
        :return: terminal_head_event function
        """

        def terminal_head_event(_, _x):
            """
            Event to stop integration when heading equals head_tar
            :param _: time (not used)
            :param _x: states
            :return: difference between current and target heading
            """
            return _x[5] - head_tar

        # Add attributes to event function to end propagation when return values goes crosses zero
        terminal_head_event.terminal, terminal_head_event.direction = True, 0

        return terminal_head_event

    @staticmethod
    def generate_fpa_event_w_tol(fpa_tar, tol=np.deg2rad(0.25)):
        """
        Generate event to stop integration when FPA is within a tolerance around fpa_tar
        :param fpa_tar: target flight path angle
        :param tol: tolerance around fpa_tar
        :return: terminal_fpa_event function
        """

        def terminal_fpa_event(_, _x):
            """
            Event to stop integration when FPA is within a tolerance around fpa_tar
            :param _: time (not used)
            :param _x: states
            :return: difference between absolute difference FPA and target and tolerance
            """
            return abs(_x[4] - fpa_tar) - tol

        # Add attributes to event function to end propagation when return values goes crosses zero
        terminal_fpa_event.terminal, terminal_fpa_event.direction = True, -1

        return terminal_fpa_event

    def fpa_trim(self, fpa_tar=None):
        """
        Generate MP to hold a FPA (fpa_tar) for a set amount of time (dt)

        :param fpa_tar: FPA to trim around
        :return: None
        """

        # Set fpa_tar to the current FPA if not specified
        if fpa_tar is None:
            fpa_tar = self.state[4]

        bound_controls = self.bound_controls
        atmosphere_func = self.atmosphere_func

        def _fpa_trim(_, _x):
            """
            Function to return control to hold the specified FPA for a set amount of time

            :param _: time (not used)
            :param _x: state
            :return: controls
            """
            sigma = 0
            h, theta, phi, v, gam, psi = _x

            temp, _, rho = atmosphere_func(h)
            mach = v / calc_speed_of_sound(temp)

            r = re + h
            q = 0.5 * rho * v ** 2

            alpha_trim = (-a_ref*q*r**2*(cl0 + cl2*exp(cl3*mach))*cos(sigma) + mass*mu*cos(fpa_tar)
                          - mass*r*v**2*cos(fpa_tar)) / (a_ref*cl1*q*r**2*cos(sigma))

            alpha = alpha_trim + 25 * (fpa_tar - gam)

            return bound_controls(np.array([alpha, sigma]))

        self.compute_control = _fpa_trim
        self.primitive_events = []

    def pull_up(self, fpa_tar):
        """
        Generate MP to change FPA to reach fpa_tar

        :param fpa_tar: FPA to target
        :return: None
        """

        bound_controls = self.bound_controls
        atmosphere_func = self.atmosphere_func

        # direction specifies whether to pull up or down
        direction = np.sign(fpa_tar - self.state[4])

        def _pull_up(_, _x):
            """
            Function to give control vector to either pull-up or pull-down the vehicle

            :param _: time (not used)
            :param _x: states
            :return: controls
            """
            sigma = 0
            h, theta, phi, v, gam, psi = _x

            temp, _, rho = atmosphere_func(h)
            mach = v / calc_speed_of_sound(temp)

            r = re + h
            q = 0.5 * rho * v ** 2

            alpha_trim = (-a_ref * q * r ** 2 * (cl0 + cl2 * exp(cl3 * mach)) * cos(sigma) + mass * mu * cos(fpa_tar)
                          - mass * r * v ** 2 * cos(fpa_tar)) / (a_ref * cl1 * q * r ** 2 * cos(sigma))

            alpha = alpha_trim + 2.5 * np.pi / 180 * direction

            return bound_controls(np.array([alpha, sigma]))

        self.compute_control = _pull_up
        self.primitive_events = self.generate_fpa_event(fpa_tar)

    def turn(self, d_head, fpa_tar=None):
        """
        Generate MP to turn the HGV (change heading) by d_head at FPA fpa_tar

        :param d_head: desired change in heading angle
        :param fpa_tar: FPA at which to execute turn
        :return: None
        """

        bound_controls = self.bound_controls
        atmosphere_func = self.atmosphere_func

        # Set fpa_tar to the current FPA if not specified
        if fpa_tar is None:
            fpa_tar = self.state[4]

        direction = np.sign(d_head)
        head_tar = self.state[5] + d_head

        def _turn(_, _x):
            """
            Function to give control vector to turn the vehicle to head_tar

            :param _: time (not used)
            :param _x: states
            :return: controls
            """
            h, theta, phi, v, gam, psi = _x

            temp, _, rho = atmosphere_func(h)
            mach = v / calc_speed_of_sound(temp)

            r = re + h
            q = 0.5 * rho * v ** 2

            sigma = direction * np.deg2rad(75)
            alpha_trim = (-a_ref * q * r ** 2 * (cl0 + cl2 * exp(cl3 * mach)) * cos(sigma) + mass * mu * cos(fpa_tar)
                          - mass * r * v ** 2 * cos(fpa_tar)) / (a_ref * cl1 * q * r ** 2 * cos(sigma))

            alpha = alpha_trim + 5 * (fpa_tar - gam)

            return bound_controls(np.array([alpha, sigma]))

        self.compute_control = _turn
        self.primitive_events = self.generate_head_event(head_tar)
