# daf_test_start.py - DAF connection test based on ppo_aoa_random_start.py

import logging
import numpy as np
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
import torch

from rl_runners.mp_runners.ppo_mp_fpa_trim_and_turn import initial_state, target_location
from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.daf_base_classes import DafBaseClass
from backend.utils.analysis import plot_average_reward_curve
from backend.utils.daf_client import DafClient
from backend.utils.logger import start_log
from backend.call_backs.mp_tensorboard_callback import MPTensorboardCallback
from backend.utils.hyperparams import load_ppo_hyperparams

# INITIAL_STATE = np.array([40000, np.deg2rad(35.1), np.deg2rad(-106.6), 6000, 0, np.deg2rad(45)])
# TARGET_LOCATION = np.array([20000, np.deg2rad(40.4), np.deg2rad(-86.9)])
TARGET_VELOCITY = np.array([1_000, 0, np.deg2rad(0)])  # [v, gam, psi] [m/s, rad, rad] for moving target

target_min_velocity = 1
target_max_velocity = 1_500
target_min_psi = np.deg2rad(-90)
target_max_psi = np.deg2rad(90)

MATLAB_RUNNER = 'a4h.runners.hgvmpTest'
EARLY_EXIT_ACTION = 86

TEST_ACTION = 5
TEST_NUM_RUNS = 2
TEST_SIM_SECS = 100


def load_continuous(file_path, ppo_env):
    """loads a PPO model and readjusts the variance for continuous action spaces"""

    ppo_model = PPO.load(file_path, ppo_env)

    # Reset initial std
    policy_std = np.exp(ppo_model.policy.log_std.cpu().detach().numpy())  # right now, log_std not an attribute
    for i in range(len(policy_std)):
        policy_std[i] = 0.4
    print('Initial policy std = ', policy_std)
    ppo_model.policy.log_std = torch.nn.parameter.Parameter(torch.tensor(np.log(policy_std),
                                                            device='cuda:0', requires_grad=True))
    ppo_model.save('tmp/ppo_tmp')
    ppo_model = PPO.load('tmp/ppo_tmp', ppo_env)
    return ppo_model


if __name__ == '__main__':

    if len(sys.argv) > 1 and \
            (sys.argv[1].lower() == '-d' or sys.argv[1].lower() == '--debug'):
        start_log(logging.DEBUG)
    else:
        start_log(logging.INFO)

    log = logging.getLogger('a4h')
    integrated_test = True
    moving_target = False
    full_run = False  # true for long runs with logging/saving model

    if integrated_test:  # SB3 and DAF
        if moving_target:
            target_state = np.append(target_location, TARGET_VELOCITY)
        else:  # stationary target
            target_state = np.append(target_location, np.array([0, 0, 0]))

        hyperparams = load_ppo_hyperparams("saved_hyperparams/discreteDAF")

        if full_run:
            # model = PPO.load('tmp/DAF/dafHGVmpTestAgent250k', env)
            total_episodes = 500_000
            env = DiscreteEnv(DafBaseClass(initial_state, target_state, MATLAB_RUNNER, total_episodes))
            model = PPO('MlpPolicy', env, verbose=1, **hyperparams, tensorboard_log='tmp/DAF/dafHGVmpTest')
            model.action_noise = NormalActionNoise(mean=np.array(0), sigma=np.array(1))
            model.learn(total_timesteps=total_episodes, callback=MPTensorboardCallback())
            model.save('tmp/DAF/dafHGVmpTestAgent500k')
        else:  # Quick runs without logging/saving model
            total_episodes = 2_048
            env = DiscreteEnv(DafBaseClass(initial_state, target_state, MATLAB_RUNNER, total_episodes))
            model = PPO('MlpPolicy', env, verbose=1, **hyperparams)
            model.action_noise = NormalActionNoise(mean=np.array(0), sigma=np.array(1))
            model.learn(total_timesteps=total_episodes)

        # TODO Better way than reaching into env.agent.daf_client?
        env.agent.daf_client.send_action(EARLY_EXIT_ACTION)
        env.agent.daf_client.receive()  # Get end msg
        env.agent.daf_client.exit_daf()

        plot_average_reward_curve(env.saved_agents, 100)
        print('Done RL mp, homie')

    else:  # DAF only
        log.info('Creating DAF connection...')
        daf_client = DafClient()
        python_params = {
            'initialState': initial_state.tolist(),
            'targetState':  np.append(target_location, [0, 0, 0]).tolist(),
            'isLogging': True}

        for run in range(TEST_NUM_RUNS):
            log.info('Requesting sim run #%d...' % (run + 1,))
            params = daf_client.run_sim(MATLAB_RUNNER, TEST_SIM_SECS, python_params)
            log.info('Received params %s...' % params)
            fini = False
            action_counter = 0

            while not fini:
                state = daf_client.get_state()

                # TODO Definitive end flag rcampbel20210722
                if ('fini' in state and not state['fini'] == '') or \
                        ('done' in state and state['done']):
                    fini = True
                else:
                    if run == 0 and action_counter > 1:  # Test prematurely terminating sim
                        action = EARLY_EXIT_ACTION
                        fini = True
                    else:
                        action = TEST_ACTION

                    daf_client.send_action(action)
                    log.debug('Sent action ' + str(action))
                    action_counter += 1

            msg = daf_client.receive()
            log.info('DAF end msg ' + str(msg))

        log.info('Requesting MATLAB exit...')
        daf_client.exit_daf()
        print('Done mp, homie')
