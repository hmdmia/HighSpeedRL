import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from backend.rl_environments.discrete_environment import DiscreteEnv
from aoa_variation_ import AoAModification
from model_testing.trained_model_csv import fail_detect


nominal_start = np.array([30000, 0, 0, 3000, 0, 0])

# Create environment
env = DiscreteEnv(AoAModification())

# Load the trained agent
model = PPO.load('../../../trained_agents/ppo_aoa_random_start_uni.zip', env=env)

fail_detect(env, model, 'aoa_fail_detect')


