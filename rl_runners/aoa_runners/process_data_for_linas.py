import numpy as np
from stable_baselines3 import PPO
import pandas

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.utils.analysis import run_network_for_shap
from ppo_aoa_random_start import AoARandomStart

env = DiscreteEnv(AoARandomStart())
model = PPO.load('ppo_aoa_random_start.zip')

obs, act, rews, dones = run_network_for_shap(env, model, num_trials=1000)

aoa_options = np.linspace(-20, 20, 21)
aoa = np.array([aoa_options[idx] for idx in act])

panda_obs = pandas.DataFrame(data=np.vstack((np.array(obs).T, np.array(act), aoa, np.array(rews), np.array(dones))).T,
                             columns=['time', 'altitude', 'velocity', 'FPA', 'action', 'aoa', 'reward', 'done'])

panda_obs.to_csv('ppo_aoa_random_start_data_set.csv')
