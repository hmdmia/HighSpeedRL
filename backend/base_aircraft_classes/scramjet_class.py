import numpy as np
from numpy import sin, cos, tan

from backend.utils.standard_atmosphere import StandardAtmosphere, calc_speed_of_sound

from backend.base_aircraft_classes.base_class import VehicleBase

"""
Scramjet model (from Joseph Williams' Dissertation)

Constraints on states:
600 < mass < 1300 (kg)

Constraints on control:
0 < area < 1 (nd)
-10 < alpha < 10 (deg)
"""

# Define Constants
re, a_ref, mu = 6378000, 0.35, 3.986e14
amax = 0.3  # m2
mass0 = 1300  # kg
massMin = 600 # kg
eps2 = 1e-13  # kg/m2s
tempMax = 1600  # max temp (K)
mc = 3  # combustion mach number
hcr = 1.4  # heat capacity ratio
cp = 1004  # m2/s2-K
hpr = 43903250
qmin = 20000 # kg/m-s2

class Scramjet(VehicleBase):
    def __init__(self, initial_state, initial_time=0.):

        self.n = 7
        self.m = 3

        VehicleBase.__init__(self, initial_state, initial_time=initial_time)

        self.atmosphere_func = StandardAtmosphere().calc_values
        self.func_stop = self.hit_ground_event

    @staticmethod
    def compute_aero(alpha, mach):
        # cd set to mach = 6
        cl = (-0.008*mach**3 + 0.133*mach**2 - 0.793*mach + 2.648)*(0.001*alpha**2 + 0.2*alpha + 0.19)
        cd = 18.231*alpha**2 - 0.4113*alpha + 0.26943
        return cl, cd

    def _eom_func(self, t, x, u):
        # Control input. A is nondimensional engine area 1>=A>=0
        alpha, beta, area = u
        h, theta, phi, v, gam, psi, mass = x

        temp, _, rho = self.atmosphere_func(h)
        mach = v / calc_speed_of_sound(temp)
        cl, cd = self.compute_aero(alpha, mach)

        taulam = tempMax*(1 + mc**2*(hcr - 1)/2) / temp
        taur = 1 + mach**2*(hcr - 1)/2

        # Make sure taulam is in acceptable range, else produce 0 thrust
        taulam = {True: taulam, False: taur}[taulam > taur and 0.5*rho*v**2 > qmin and mass > massMin]

        f = cp*temp*(taulam - taur)/hpr

        mdot0 = rho*amax*area*v
        thrust = mdot0*v*(np.sqrt(taulam/taur) - 1)  # gc term in thesis is conversion constant (1 for SI)

        r = re + h
        q = 0.5 * rho * v ** 2
        lift, drag = q * cl * a_ref, q * cd * a_ref

        x_dot = [
            v * sin(gam),  # h dot
            v * cos(gam) * cos(psi) / r,  # theta dot
            v * cos(gam) * sin(psi) / (r * cos(theta)),  # phi dot
            (thrust*cos(alpha) - drag) / mass - mu * sin(gam) / r ** 2,  # v dot
            (lift + thrust*sin(alpha)) * cos(beta) / (mass * v) + (v / r - mu / (v * r ** 2)) * cos(gam),  # gam dot
            (lift + thrust*sin(alpha)) * sin(beta) / (mass * cos(gam) * v) + v / r * cos(gam) * sin(psi) * tan(theta),  # psi dot
            -mdot0*f + eps2*amax*area]  # m dot

        return x_dot

    def plot_state_history(self, style='planar', size='ppt', title=None):
        import matplotlib.pyplot as plt

        if size == 'ppt':
            figsize = (6.15, 4.75)
        else:
            figsize = None

        fig = plt.figure(0, figsize=figsize)

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

        if style == 'segmented':

            ax1 = fig.add_subplot(511)
            ax2 = fig.add_subplot(512)
            ax3 = fig.add_subplot(513)
            ax4 = fig.add_subplot(514)
            ax5 = fig.add_subplot(515)

            for ti, yi, ui in zip(self.time_history, self.state_history, self.control_history):
                ax1.plot(yi[1, :] * 180 / np.pi, yi[0, :] / 1000)
                ax1.set_xlabel('Downrange [deg]')
                ax1.set_ylabel('Altitude [km]')
                #ax1.set_title('Trajectory')

                ax2.plot(ti, yi[6, :])
                ax2.set_ylabel('Mass [kg]')
                #ax2.plot(ti, ui[0, :] * 180 / np.pi)
                ax2.set_xlabel('Time [s]')
                #ax2.set_ylabel('Angle of Attack [deg]')
                #ax2.set_title('Control History')

                ax3.plot(ti, yi[4, :] * 180 / np.pi)
                ax3.set_xlabel('Time [s]')
                ax3.set_ylabel('Flight Path Angle [deg]')
                #ax3.set_title('Flight Path Angle')

                ax4.plot(ti, yi[3, :])
                ax4.set_xlabel('Time [s]')
                ax4.set_ylabel('Velocity [m/s]')

                ax5.plot(ti, ui[0, :] * 180 / np.pi)
                ax5.set_xlabel('Time [s]')
                ax5.set_ylabel('Angle of Attack [deg]')

        if title is not None:
            plt.suptitle(title)

        if size == 'half_ppt':
            fig.subplots_adjust(hspace=0.5)

        plt.show()

    @staticmethod
    def hit_ground_event(_, x):
        return x[0]

    hit_ground_event.terminal, hit_ground_event.direction = True, -1


if __name__ == '__main__':
    from backend.data_handling.load_data import load_reference_set

    # ref = load_reference_set('../data_handling/emergency_descent_nominal')[-1]
    # t_ref = ref['time']
    # y_ref = np.insert(ref['states'], [2, 4], 0, axis=1)
    # u_ref = np.insert(np.array(ref['controls'], ndmin=2).T, [1], 0, axis=1)
    t_ref = 0
    y_ref = np.array([20e3, 0, 0, 1300, 0, 0, mass0])  # 20 km altitude, 1300 m/s velocity
    u_ref = np.array([0, 0, 0.5])

    scramjet = Scramjet(y_ref, t_ref)

    # test_routine = 'direct'
    # test_routine = 'sim_step'
    test_routine = 'delta_step'
    # test_routine = 'linear'

    if test_routine == 'direct':
        scramjet.load_control_profile(t_ref, u_ref)
        for _ in range(11):
            scramjet.sim_step(5)

        scramjet.plot_state_history('segmented')

    elif test_routine == 'sim_step':
        step_length = 10
        for aoa_step in np.array([7.5, 10, 9.5, 7.5, 6, 5, 9.5]) * np.pi/180:
            scramjet.constant_step(step_length, np.array([aoa_step, 0, 0.5]))
            scramjet.sim_step(step_length)

        scramjet.plot_state_history('segmented', title='Step Primitives')

    elif test_routine == 'delta_step':
        #step_length = 10
        step_length = 60
        #for delta_aoa in np.array([7.5, -10, 9.5, -7.5, -6, 5, 9.5]) * np.pi / 180:
        for aoa_change in np.array([2]) * np.pi / 180:
            #scramjet.delta_step(step_length, np.array([delta_aoa, 0, 0.5]))
            scramjet.linear_change(step_length, np.array([aoa_change, 0, 0.1]))
            scramjet.sim_step(step_length)

        scramjet.plot_state_history('segmented', title='Delta Step Primitives')

    elif test_routine == 'linear':
        step_length = 10
        #for aoa_change in np.array([7.5, 10, 9.5, 7.5, 6, 5, 9.5]) * np.pi/180:
        for aoa_change in np.array([7.5, 7.5, 7.5, 7.5]) * np.pi / 180:
            #scramjet.linear_change(step_length, np.array([aoa_change, 0, 0.5]))
            scramjet.linear_change(step_length, np.array([aoa_change, 0, 0.1]))
            scramjet.sim_step(step_length)

        scramjet.plot_state_history('segmented', title='Linear Primitives')

        scramjet.plot_state_history('segmented')
