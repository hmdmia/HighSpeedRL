import numpy as np
from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_environments.box_environment import BoxEnv
from backend.rl_base_classes.daf_base_classes import DafBaseClass, DafContinuousClass
from rl_runners.dafHGVmpTest import initial_state, target_location, TARGET_VELOCITY, MATLAB_RUNNER


def observe_network_for_dist(_initial_state, _model):
    done = False
    obs = _model.env.reset(initial_state=_initial_state)
    while not done:
        action, _state = _model.predict(obs, deterministic=True)
        obs, _, done, __ = _model.env.step(action)

    return model.env.agent.dist

continuous_env = False

target_state = np.append(target_location, TARGET_VELOCITY)
if continuous_env:
    env = BoxEnv(DafContinuousClass(initial_state, target_state, MATLAB_RUNNER, total_episodes=1))
else:
    env = DiscreteEnv(DafBaseClass(initial_state, target_state, MATLAB_RUNNER, total_episodes=1))

env.agent.curriculum = env.agent.generate_curriculum(1)
model = PPO.load('canonical/10_29_2021_hgvmp/dhm1to500mps800k')
target_velocities = np.linspace(1, 1500, 20)
target_headings = np.linspace(np.deg2rad(-150), np.deg2rad(150), 20)
data = []

for target_velocity in target_velocities:
    for target_heading in target_headings:
        target_state = np.append(target_location, np.array([target_velocity, 0, target_heading]))
        env.agent.target_state = target_state
        env.agent.curriculum = env.agent.generate_curriculum(1)
        model.env = env
        dist = observe_network_for_dist(initial_state, model)

        data.append([target_velocity, target_heading, dist])
        print(f'vel: {target_velocity}, head: {target_heading}, dist: {dist}')

np.savetxt("tmp/DAF/varying_ICs.csv", np.asarray(data), delimiter=",")

# Receive out message from MATLAB, close socket
env.agent.daf_client.receive()
env.agent.daf_client.exit_daf()
