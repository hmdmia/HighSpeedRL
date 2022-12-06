import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from backend.rl_environments.discrete_environment import DiscreteEnv

from model_testing.disturbance.obs.obs_disturbance_fail_detect import AoAAbsoluteObsDist

factors = []

# Create environment
env = DiscreteEnv(AoAAbsoluteObsDist())
# Load the trained agent

nominal_start_1 = np.array([30000, 0, 0, 3000, 0, 0])

model = PPO.load('../../../trained_agents/ppo_aoa_random_start_uni.zip', env=env)

obs1 = []
success = []
ctr = 0


while 0 not in success:
    time = random.sample(range(0,40),7)
    for i in range(len(time)):
        time[i] = float(time[i])
    done = False
    obs = env.reset(initial_state=nominal_start_1)
    obs1.append(obs[1])
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, _, done, __ = env.step(action)
        obs1.append(obs[1])
    if done == True:
        if obs1[-1] <= 3000:
            success.append(1)
        else:
            env.agent.plot_control(style='segmented')
            success.append(0)
            pert = np.array(time)

    ctr+=1
    print(ctr)
    if ctr > 3000:
        break