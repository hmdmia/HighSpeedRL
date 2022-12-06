import numpy as np
from stable_baselines3 import PPO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.scramjet_base_class import ScramjetBaseClass
from backend.utils.analysis import run_network, plot_average_reward_curve


class ScramjetAbsolute(ScramjetBaseClass):
    def __init__(self):
        self.initial_state = np.array([30000, 0, 0, 3000, 0, 0, 1300])
        ScramjetBaseClass.__init__(self, self.initial_state)

        self._fpa_tol = 5 * np.pi / 180

        self.n_actions = 21
        self._aoa_options = np.linspace(-10, 10, self.n_actions)/180*np.pi

    def reset(self, **kwargs):
        self.__init__()

    def reward(self):
        return self._reward3()

    def _inner_step(self, action):
        u = np.array([self._aoa_options[action], 0., 0.5])
        self.constant_step(self.dt, u)
        self.sim_step(self.dt)


agent = ScramjetAbsolute()
env = DiscreteEnv(agent)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=500)
#model.save('ppo_scramjet_absolute')

#choose_best(env.saved_agents).plot_state_history(style='segmented')
run_network(agent.initial_state, env, model)
plot_average_reward_curve(env.saved_agents, 100)