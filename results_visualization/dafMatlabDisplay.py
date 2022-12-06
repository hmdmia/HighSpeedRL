import numpy as np

from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_environments.box_environment import BoxEnv
from backend.rl_base_classes.daf_base_classes import DafBaseClass, DafContinuousClass
from rl_runners.mp_runners.run_saved_model import run_network
from rl_runners.dafHGVmpTest import initial_state, target_location, TARGET_VELOCITY, MATLAB_RUNNER

moving_target = True
continuous_env = False

if moving_target:
    target_state = np.append(target_location, TARGET_VELOCITY)
else:  # stationary target
    target_state = np.append(target_location, np.array([0, 0, 0]))

if continuous_env:
    env = BoxEnv(DafContinuousClass(initial_state, target_state, MATLAB_RUNNER, total_episodes=1))
else:
    env = DiscreteEnv(DafBaseClass(initial_state, target_state, MATLAB_RUNNER, total_episodes=1))

env.agent.daf_logging = True
env.agent.curriculum = env.agent.generate_curriculum(1)
model = PPO.load('canonical/10_29_2021_hgvmp/dhm1to500mps800k', env)
run_network(initial_state, env, model)

# Receive out message from MATLAB, close socket
env.agent.daf_client.receive()
env.agent.daf_client.exit_daf()
