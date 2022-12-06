import numpy as np
from stable_baselines3 import PPO
import shap
import pandas

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.utils.analysis import run_network_for_shap
from ppo_aoa_random_start import AoARandomStart

env = DiscreteEnv(AoARandomStart())
model = PPO.load('ppo_aoa_random_start.zip')

obs, act, _, __ = run_network_for_shap(env, model, num_trials=100)


def _predict(_ob):
    return model.predict(_ob)[0]


panda_obs = pandas.DataFrame(data=obs, columns=['t', 'h', 'v', 'gam'])

explainer = shap.Explainer(_predict, panda_obs)
shap_values = explainer(panda_obs)

shap.plots.waterfall(shap_values[0])
