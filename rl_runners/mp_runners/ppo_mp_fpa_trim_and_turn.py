import numpy as np
from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.mp_base_classes import FPATrimsAndTurns
from backend.utils.analysis import plot_average_reward_curve
from backend.call_backs.mp_tensorboard_callback import MPTensorboardCallback
from backend.utils.hyperparams import load_ppo_hyperparams

initial_state = np.array([40000, np.deg2rad(35.1), np.deg2rad(-106.6), 6000, 0, np.pi/4])
target_location = np.array([20000, np.deg2rad(40.4), np.deg2rad(-86.9)])


if __name__ == '__main__':
    from run_saved_model import run_network

    env = DiscreteEnv(FPATrimsAndTurns(initial_state, target_location))

    hyperparams = load_ppo_hyperparams("saved_hyperparams/ppo_absolute_mp")
    hyperparams["policy_kwargs"]["net_arch"] = [256]
    model = PPO('MlpPolicy', env, verbose=1, **hyperparams)  # , tensorboard_log="tmp/mp/PPO/")
    # model = PPO.load('tmp/saved_models/ppo_mp_fpa_trim_and_turns400k', env)
    model.learn(total_timesteps=int(200e3))  # , callback=MPTensorboardCallback())
    # model.save('tmp/saved_models/ppo_mp_fpa_trim_and_turns600k')

    plot_average_reward_curve(env.saved_agents, 100)
    run_network(initial_state, env, model)
