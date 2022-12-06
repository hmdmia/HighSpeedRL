
import numpy as np
from stable_baselines3 import PPO
import time

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.aoa_base_class import AoABaseClass
from backend.utils.analysis import choose_best, plot_average_reward_curve
from backend.utils.hyperparams import load_ppo_hyperparams
from backend.call_backs.tensorboard_callback import TensorboardCallback


class AoAAbsolute(AoABaseClass):
    def __init__(self):
        self.initial_state = np.array([30000, 0, 0, 3000, 0, 0])
        AoABaseClass.__init__(self, self.initial_state)

        self.n_actions = 11
        self._aoa_options = np.linspace(-20, 20, self.n_actions)/180*np.pi
        self.dt = 1

    def reset(self):
        self.__init__()

    def reward(self):
        return self._reward3()

    def _inner_step(self, action):
        u = np.array([self._aoa_options[action], 0.])
        self.delta_step(self.dt, u)
        self.sim_step(self.dt)


if __name__ == '__main__':
    env = DiscreteEnv(AoAAbsolute())
    hyperparams = load_ppo_hyperparams("ppo_absolute_params")

    model = PPO('MlpPolicy', env, verbose=1, **hyperparams, tensorboard_log="../../tmp/test/")
    model.learn(total_timesteps=100000, callback=TensorboardCallback())
    # model.save('../tmp/ppo_aoa_absolute')

    choose_best(env.saved_agents).plot_state_history(style='segmented')
    plot_average_reward_curve(env.saved_agents, 100)
