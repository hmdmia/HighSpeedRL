import numpy as np
from stable_baselines3 import DQN

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.aoa_base_class import AoABaseClass
from backend.utils.analysis import choose_best
from backend.utils.analysis import plot_average_reward_curve
from backend.utils.hyperparams import load_dqn_hyperparams
from backend.call_backs.tensorboard_callback import TensorboardCallback


class AoARelative(AoABaseClass):
    def __init__(self):
        self.initial_state = np.array([30000, 0, 0, 3000, 0, 0])
        AoABaseClass.__init__(self, self.initial_state)

        self.n_actions = 11
        self._aoa_options = np.linspace(-5, 5, self.n_actions) / 180 * np.pi
        self.dt = 5

    def _reward1(self):
        dist_alt = (self.state[0] - self._target_altitude) / (self.initial_state[0] - self._target_altitude)
        if self.success:
            time_scaled = self.time / self._max_time
            fpa_scaled = abs(self.state[4] / (89 * np.pi / 180))
            scale = 40
            reward = scale * (1 - time_scaled) * (1 - fpa_scaled)

        elif self.done:
            reward = - dist_alt
        else:
            reward = 1 - dist_alt

        return reward

    def reset(self):
        self.__init__()

    def reward(self):
        return self._reward3()

    def _inner_step(self, action):
        u = np.array([self._aoa_options[action], 0.])
        self.delta_step(self.dt, u)
        self.sim_step(self.dt)


if __name__ == '__main__':
    env = DiscreteEnv(AoARelative())
    hyperparams = load_dqn_hyperparams("../../saved_hyperparams/dqn_relative_params.txt")

    model = DQN('MlpPolicy', env, verbose=1, **hyperparams, tensorboard_log="../../tmp/DQN/")
    model.learn(total_timesteps=50000, callback=TensorboardCallback())
    # model.save("../../tmp/testmodel")
    choose_best(env.saved_agents).plot_state_history(style='segmented')
    plot_average_reward_curve(env.saved_agents)
