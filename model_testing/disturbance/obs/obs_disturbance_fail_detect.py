import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from backend.rl_environments.discrete_environment import DiscreteEnv
from rl_runners.aoa_runners.ppo_aoa_absolute import AoAAbsolute
from model_testing.trained_model_csv import fail_detect

factors = []
class AoAAbsoluteObsDist(AoAAbsolute):
    def observe(self):
        obs = super().observe()
        time = random.sample(range(0, 50), 10)
        factor = random.uniform(0.0,6.5)
        if obs[0] in time:
            obs[1] = obs[1] * factor
            factors.append(factor)
        else:
            pass
        return obs

if __name__ == '__main__':
    # Create environment
    env = DiscreteEnv(AoAAbsoluteObsDist())
    # Load the trained agent

    nominal_start_1 = np.array([30000, 0, 0, 3000, 0, 0])

    model = PPO.load('../../../trained_agents/ppo_aoa_random_start_uni.zip', env=env)
    fail_detect(env, model, 'observation_disturbance')