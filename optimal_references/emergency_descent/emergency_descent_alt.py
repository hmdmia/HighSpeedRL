from math import pi
import logging

import matplotlib.pyplot as plt
import numpy as np

import beluga

ocp = beluga.Problem()

# Define independent variables
ocp.independent('t', 's')


# Define equations of motion
ocp.state('h', 'v*sin(gam)', 'm')
ocp.state('theta', 'v*cos(gam)/r', 'rad')
ocp.state('v', '-D/mass - mu*sin(gam)/r**2', 'm/s')
ocp.state('gam', 'L/(mass*v) + (v/r - mu/(v*r^2))*cos(gam)', 'rad')


# Define quantities used in the problem
ocp.quantity('rho', 'rho0*exp(-h/H)')
ocp.quantity('Cl', 'cl1 * alpha + cl2 * exp(cl3 * v / a) + cl0')
ocp.quantity('Cd', 'cd1 * alpha **2 + cd2 * exp(cd3 * v / a) + cd0')
ocp.quantity('D', '0.5*rho*v**2*Cd*Aref')
ocp.quantity('L', '0.5*rho*v**2*Cl*Aref')
ocp.quantity('r', 're+h')

# Define controls
ocp.control('alpha', 'rad')

# Define constants
ocp.constant('h_0', 50000, 'm')
ocp.constant('v_0', 4000, 'm/s')
ocp.constant('gam_0', (-90)*pi/180, 'rad')

ocp.constant('mu', 3.986e5*1e9, 'm^3/s^2')  # Gravitational parameter, m^3/s^2
ocp.constant('rho0', 1.2, 'kg/m^3')  # Sea-level atmospheric density, kg/m^3
ocp.constant('H', 7500, 'm')  # Scale height for atmosphere of Earth, m
ocp.constant('mass', 907.20, 'kg')  # Mass of vehicle, kg
ocp.constant('re', 6378000, 'm')  # Radius of planet, m
ocp.constant('Aref', 0.4839, 'm^2')  # Reference area of vehicle, m^2
ocp.constant('cl0', -0.2317, '1')
ocp.constant('cl1', 0.0513*180/3.14159, '1')
ocp.constant('cl2', 0.2945, '1')
ocp.constant('cl3', -0.1028, '1')
ocp.constant('a', 300, 'm/s')
ocp.constant('cd0', 0.024, '1')
ocp.constant('cd1', 7.24e-4*180**2/3.14159**2, '1')
ocp.constant('cd2', 0.406, '1')
ocp.constant('cd3', -0.323, '1')

ocp.constant('h_f', 10000, 'm')
ocp.constant('gam_f', (-30)*pi/180, 'rad')

ocp.constant('amax', 20 * pi / 180, 'rad')

ocp.constant('eps', 0.01, '1')

ocp.constant('k', 1, '1/rad**2')

# Define costs
ocp.path_cost('1', '1')

# Define constraints
ocp.initial_constraint('h-h_0', 'm')
ocp.initial_constraint('theta', 'rad')
ocp.initial_constraint('v-v_0', 'm/s')
ocp.initial_constraint('gam-gam_0', 'rad')
ocp.initial_constraint('t', 's')
ocp.terminal_constraint('h-h_f', 'm')
ocp.terminal_constraint('gam-gam_f', 'rad')

ocp.path_constraint('alpha', 'rad', lower='-amax', upper='amax', activator='eps', method='utm')

ocp.scale(m=40000, s=10, kg='mass', rad=1, W=4e8)

bvp_solver = beluga.bvp_algorithm('spbvp')

guess_maker = beluga.guess_generator(
    'auto',
    start=[30500, 0, 3000, -10*pi/180],
    control_guess=[0],
    use_control_guess=True,
    direction='forward',
    costate_guess=-0.01
)

continuation_steps = beluga.init_continuation()

continuation_steps.add_step('bisection') \
                .num_cases(300) \
                .const('h_f', 3000) \
                # .const('h_f', 10000/3.28084)

continuation_steps.add_step('bisection') \
                .num_cases(100) \
                .const('gam_f', 0*pi/180)

continuation_steps.add_step('bisection') \
                .num_cases(100) \
                .const('gam_0', 0*pi/180)

continuation_steps.add_step('bisection') \
                .num_cases(100, spacing='log') \
                .const('eps', 2e-5)

continuation_steps.add_step('bisection') \
                .num_cases(11) \
                .const('h_0', 29500) \

beluga.add_logger(display_level=logging.INFO)


sol_set = beluga.solve(
    ocp=ocp,
    method='indirect',
    bvp_algorithm=bvp_solver,
    steps=continuation_steps,
    guess_generator=guess_maker,
    optim_options={'control_method': 'differential', 'analytical_jacobian': False},
    save_sols='emergency_descent_alt.json', initial_helper=True
)

traj = sol_set[-1][-1]

t_bounds = (traj.t[0], traj.t[-1])
amax = traj.k[20]*180/np.pi
q_dot_max = 0

plt.figure(1)
plt.subplot(221)
plt.plot(traj.y[:, 1]*180/np.pi, traj.y[:, 0]/1000)

plt.xlabel('Downrange [deg]')
plt.ylabel('Altitude [km]')
plt.title('Cav-H Trajectories')
# plt.show()

# plt.figure(2)
plt.subplot(222)
plt.plot(traj.t, traj.u[:, 0]*180/pi, label='Control History')
plt.hlines([-amax, amax], t_bounds[0], t_bounds[1], linestyle='--', label='Bounds')

plt.xlabel('Time [s]')
plt.ylabel('Control [deg]')
plt.title('Control History')
# plt.show()

plt.subplot(223)
plt.plot(traj.y[:, 2]/1000, traj.y[:, 0]/1000)

plt.xlabel('Velocity [km/s]')
plt.ylabel('Altitude [km]')
plt.title('h-v Diagram')


# plt.figure(3)
plt.subplot(224)
plt.plot(traj.t, (1.74153e-4*(1.2*np.exp(-traj.y[:, 0]/7500)/(1/120*0.3048))**(1/2)*traj.y[:, 2]**3)/1e7)
# plt.hlines(q_dot_max, t_bounds[0], t_bounds[1], linestyle='--', label='Bounds')

plt.xlabel('Time [s]')
plt.ylabel('Heating [kW/cm^2]')
plt.title('Heat Rate at Stagnation Point')

plt.suptitle('Emergency Decent Problem')

plt.show()

