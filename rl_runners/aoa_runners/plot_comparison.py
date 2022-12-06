import numpy as np
import pandas
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from ppo_aoa_random_start import AoARandomStart

beluga_df = pandas.read_csv('emergency_descent.csv')

t_b = np.array(beluga_df.time)
h_b = np.array(beluga_df.altitude)
theta_b = np.rad2deg(np.array(beluga_df.downrange))
alpha_b = np.rad2deg(np.array(beluga_df.angle_of_attack))

env = DiscreteEnv(AoARandomStart())
model = PPO.load('ppo_aoa_random_start.zip')

done = False
obs = env.reset(initial_state=np.array([30000, 0, 0, 3000, 0, 0]))
for _ in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, _, done, __ = env.step(action)
    if done:
        break

t_r = np.concatenate(env.agent.time_history)
y_r = np.hstack(env.agent.state_history)
u_r = np.hstack(env.agent.control_history)

h_r = y_r[0, :]
theta_r = np.rad2deg(y_r[1, :])
alpha_r = np.rad2deg(u_r[0, :])

fig1 = plt.figure(figsize=(6.15, 4.76))
gs = plt.GridSpec(2, 1, figure=fig1,
                  left=None, bottom=None, right=None, top=None, wspace=None, hspace=1,
                  width_ratios=None, height_ratios=[1, 1])

fig1.suptitle('Validation of RL with Optimal Solution')

ax1 = fig1.add_subplot(gs[0, 0])
ax1.plot(theta_b, h_b, label='Optimal Solution')
ax1.plot(theta_r, h_r, label='RL Solution')
ax1.legend()
ax1.set_xlabel(r'Downrange Angle, $\theta$ [deg]')
ax1.set_ylabel(r'Altitude, $h$ [m]')
ax1.set_title('Flight Path')

ax2 = fig1.add_subplot(gs[1, 0])
ax2.plot(t_b, alpha_b, label='Optimal Solution')
ax2.plot(t_r, alpha_r, label='RL Solution')
ax2.set_xlabel(r'Time, $t$ [s]')
ax2.set_ylabel(r'Angle of Attack, $\alpha$ [deg]')
ax2.set_title('Control History')

plt.show()

