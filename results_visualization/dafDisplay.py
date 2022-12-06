import numpy as np

from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.mp_base_classes import FPATrimsAndTurns
from rl_runners.mp_runners.run_saved_model import run_network

from rl_runners.dafHGVmpTest import initial_state, target_location

env = DiscreteEnv(FPATrimsAndTurns(initial_state, target_location))
model = PPO.load('../../tmp/DAF/dafHGVmpTestAgent1')
run_network(initial_state, env, model)
