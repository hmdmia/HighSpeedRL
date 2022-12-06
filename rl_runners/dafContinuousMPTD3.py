import sys
import logging
import numpy as np

from stable_baselines3 import TD3
import torch

from backend.utils.logger import start_log
from backend.rl_environments.box_environment import BoxEnv
from backend.rl_base_classes.daf_base_classes import DafContinuousClass
from rl_runners.mp_runners.ppo_mp_fpa_trim_and_turn import initial_state, target_location
from backend.utils.hyperparams import load_ppo_hyperparams
from backend.call_backs.mp_tensorboard_callback import MPTensorboardCallback
from backend.utils.analysis import plot_average_reward_curve

MATLAB_RUNNER = 'a4h.runners.hgvmpTest'
EARLY_EXIT_ACTION = 86


def load_continuous(file_path, ppo_env):
    """loads a PPO model and readjusts the variance for continuous action spaces"""

    ppo_model = TD3.load(file_path, ppo_env)

    # Reset initial std
    policy_std = np.exp(ppo_model.policy.log_std.cpu().detach().numpy())  # right now, log_std not an attribute
    for i in range(len(policy_std)):
        policy_std[i] = 0.4
    print('Initial policy std = ', policy_std)
    ppo_model.policy.log_std = torch.nn.parameter.Parameter(torch.tensor(np.log(policy_std),
                                                                         device='cuda:0', requires_grad=True))
    ppo_model.save('tmp/ppo_tmp')
    ppo_model = TD3.load('tmp/ppo_tmp', ppo_env)
    return ppo_model


if __name__ == '__main__':
    # Logging
    if len(sys.argv) > 1 and \
            (sys.argv[1].lower() == '-d' or sys.argv[1].lower() == '--debug'):
        start_log(logging.DEBUG)
    else:
        start_log(logging.INFO)
    log = logging.getLogger('a4h')

    # Flags
    full_run = False

    # Define parameters
    target_velocity = np.array([1, 0, np.deg2rad(0)])  # [v, gam, psi] [m/s, rad, rad]

    hyperparams = load_ppo_hyperparams("saved_hyperparams/discreteDAF")
    target_state = np.append(target_location, target_velocity)

    def train_model(total_observations, previous_model=None, tensorboard_log=None):
        """
        :param: total_observations (int) number of observations taken when learning
        :param: previous_model (String) local path to model to load and continue training
        :param: tensorboard_log (String) local path to location to put folder with tensor log data
        """
        training_env = BoxEnv(DafContinuousClass(initial_state, target_state, MATLAB_RUNNER, total_observations))
        if not previous_model:
            training_model = TD3('MlpPolicy', training_env, verbose=1, tensorboard_log=tensorboard_log)
        else:
            training_model = TD3.load(previous_model, env=training_env)
        training_model.learn(total_timesteps=total_observations, callback=MPTensorboardCallback())
        return training_model, training_env

    if full_run:
        model, env = train_model(total_observations=3_000_000, previous_model=None, tensorboard_log='tmp/DAF/dafHGVmpTest')
        model.save('tmp/DAF/continuousMP3M')
    else:
        model, env = train_model(total_observations=2_048)

    env.agent.daf_client.send_action(EARLY_EXIT_ACTION)
    env.agent.daf_client.receive()  # Get end msg
    env.agent.daf_client.exit_daf()

    plot_average_reward_curve(env.saved_agents, 100)
    print('Done RL continuous mp, homie')
