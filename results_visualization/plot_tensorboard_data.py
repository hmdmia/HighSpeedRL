import numpy as np
from matplotlib import pyplot as plt

# Read MP Data
mp_directory = 'canonical/05_16_22_moving_target'
mp_mean_rew_steps = []
mp_mean_rews = []
mp_mean_ep_rew_steps = []
mp_mean_ep_rews = []
last_step = 0
last_ep_step = 0
for i in range(1, 12):
    data = np.loadtxt(mp_directory + '/mean_rew' + str(i) + '.csv', delimiter=',', skiprows=1, ndmin=2)
    mp_mean_rew_steps.append(data[:, 1] + last_step)
    mp_mean_rews.append(data[:, 2])
    last_step += data[-1, 1]
    print(last_step)

    print(f'Stage {i}: Steps = {data[-1, 1]}, Final Mean Reward: {data[-1, 2]}')

    data = np.loadtxt(mp_directory + '/mean_ep_rew' + str(i) + '.csv', delimiter=',', skiprows=1, ndmin=2)
    mp_mean_ep_rew_steps.append(data[:, 1] + last_ep_step)
    mp_mean_ep_rews.append(data[:, 2])
    last_ep_step += data[-1, 1]
    print(last_ep_step)

# Read non-MP Data
non_mp_directory = 'canonical/05_17_22_non_mp'
non_mp_mean_rew_steps = []
non_mp_mean_rews = []
non_mp_mean_ep_rew_steps = []
non_mp_mean_ep_rews = []
last_step = 0
last_ep_step = 0
for i in range(1, 12):
    data = np.loadtxt(non_mp_directory + '/mean_rew' + str(i) + '.csv', delimiter=',', skiprows=1, ndmin=2)
    non_mp_mean_rew_steps.append(data[:, 1] + last_step)
    non_mp_mean_rews.append(data[:, 2])
    last_step += data[-1, 1]

    print(f'Stage {i}: Steps = {data[-1, 1]}, Final Mean Reward: {data[-1, 2]}')

    data = np.loadtxt(non_mp_directory + '/mean_ep_rew' + str(i) + '.csv', delimiter=',', skiprows=1, ndmin=2)
    non_mp_mean_ep_rew_steps.append(data[:, 1] + last_ep_step)
    non_mp_mean_ep_rews.append(data[:, 2])
    last_ep_step += data[-1, 1]

# Plot Data
# fig = plt.figure(figsize=(6.15, 4.75))
fig = plt.figure(figsize=(6.5, 2))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for mp_mean_ep_rew_step, mp_mean_ep_rew, non_mp_mean_ep_rew_step, non_mp_mean_ep_rew in zip(mp_mean_ep_rew_steps, mp_mean_ep_rews, non_mp_mean_ep_rew_steps, non_mp_mean_ep_rews):
    ax1.plot(non_mp_mean_ep_rew_step / 1000, non_mp_mean_ep_rew)
    ax2.plot(mp_mean_ep_rew_step / 1000, mp_mean_ep_rew)

ax1.set_xlabel('RL Step (10k)')
ax1.set_ylabel('Mean Reward')
ax1.set_title('Non-MP Reward Curve')
ax1.grid()

ax2.set_xlabel('RL Step (10k)')
ax2.set_ylabel('Mean Reward')
ax2.set_title('MP Reward Curve')
ax2.grid()

plt.tight_layout()
fig.savefig(fname='tmp/reward_curves.eps', format='eps')

plt.show()
