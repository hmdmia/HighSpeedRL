from copy import deepcopy

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold,\
    StopTrainingOnNoModelImprovement
from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.mp_base_classes import MovingTargetFPATrimsAndTurns
from backend.base_aircraft_classes.target_classes import RandomMovingTarget, MovingTarget
from backend.utils.analysis import plot_average_reward_curve
from backend.call_backs.mp_tensorboard_callback import MPTensorboardCallback
from backend.utils.hyperparams import load_ppo_hyperparams
from rl_runners.mp_runners.run_saved_model import run_network

initial_state = np.array((40e3,  # h [m]
                          np.deg2rad(35.1),  # theta [rad]
                          np.deg2rad(-106.6),  # phi [rad]
                          6000,  # v [m/s]
                          0,  # gamma [rad]
                          np.pi / 4))  # psi [rad]
target_state = np.array((20e3,  # h [m]
                         np.deg2rad(40.4),  # theta [rad]
                         np.deg2rad(-86.9),  # phi [rad]
                         100,  # v [m/s]
                         np.deg2rad(0),  # gamma [rad]
                         np.deg2rad(0)))  # psi [rad]
target_location = target_state[:3]
v_range = (0, 500)
psi_range = np.deg2rad((-180, 180))
# v_range = (250, 250)
# psi_range = np.deg2rad((0, 0))
target = RandomMovingTarget(target_location, v_range, psi_range, distribution="uniform")
# target = MovingTarget(target_state)
hyperparams = load_ppo_hyperparams("saved_hyperparams/movingTarget")


def training_session(_env, _model, _v_range, _psi_range):
    _psi_range_deg = np.rad2deg(_psi_range)
    name_str = str(int(_v_range[0])) + 'to' + str(int(_v_range[1])) + 'v' + str(int(_psi_range_deg[0])) \
               + 'to' + str(int(_psi_range_deg[1])) + 'psi'
    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=200)
    eval_callback = EvalCallback(deepcopy(_env),
                                 best_model_save_path='tmp/saved_models/movingTarget/' + name_str,
                                 eval_freq=hyperparams['n_steps'], n_eval_episodes=25,
                                 callback_on_new_best=reward_threshold_callback)
    _model.learn(total_timesteps=int(1e6), callback=eval_callback)

    return _model


if __name__ == '__main__':
    env = Monitor(DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, target)))
    model = PPO('MlpPolicy', env, verbose=1, **hyperparams, tensorboard_log="tmp/movingTarget/PPO/")
    # model = PPO.load('tmp/saved_models/movingTarget/100to400v-180to180psi/best_model', env)

    v_mean = 250
    psi_mean = np.deg2rad(0)
    delta_velocity_curriculum = (0, 50, 50, 50, 50, 50, 50, 100, 150, 200, 250)
    delta_psi_curriculum = np.deg2rad((0, 30, 60, 90, 120, 150, 180, 180, 180, 180, 180))

    for delta_velocity, delta_psi in zip(delta_velocity_curriculum, delta_psi_curriculum):
        v_range = (v_mean - delta_velocity, v_mean + delta_velocity)
        psi_range = (psi_mean - delta_psi, psi_mean + delta_psi)

        env.unwrapped.agent.target = RandomMovingTarget(target_location, v_range, psi_range, distribution="uniform")
        model.env.envs[0].unwrapped.agent.target = RandomMovingTarget(
            target_location, v_range, psi_range, distribution="uniform")  # Redundant bc same object
        model = training_session(env, model, v_range, psi_range)

    # plot_average_reward_curve(env.saved_agents, 100)
    # run_network(initial_state, env, model)
