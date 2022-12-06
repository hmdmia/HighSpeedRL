import numpy as np
from stable_baselines3 import DQN

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.aoa_base_class import AoABaseClass
from backend.utils.analysis import choose_best, plot_average_reward_curve, run_network
from backend.utils.hyperparams import load_dqn_hyperparams
from backend.call_backs.tensorboard_callback import TensorboardCallback

nominal_start = np.array([30000, 0, 0, 3000, 0, 0])


class AoAAbsolute(AoABaseClass):
    def __init__(self, initial_state=np.array([30000, 0, 0, 3000, 0, 0])):
        self.initial_state = initial_state
        AoABaseClass.__init__(self, self.initial_state)

        self._fpa_tol = 5 * np.pi / 180

        self.n_actions = 21
        self._aoa_options = np.linspace(-20, 20, self.n_actions)/180*np.pi
        self.dt = 1

    def reset(self, initial_state=nominal_start):
        self.__init__(initial_state=initial_state)

    def reward(self):
        return self._reward3()

    def _inner_step(self, action):
        u = np.array([self._aoa_options[action], 0.])
        self.constant_step(self.dt, u)
        self.sim_step(self.dt)


if __name__ == '__main__':
    env = DiscreteEnv(AoAAbsolute())
    hyperparams = load_dqn_hyperparams("../../saved_hyperparams/dqn_relative_params")

    model = DQN('MlpPolicy', env, verbose=1, **hyperparams, tensorboard_log="../../tmp/dqn/")
    model.learn(total_timesteps=500000, callback=TensorboardCallback())
    model.save('dqn_shap')
    model.save_replay_buffer('dqn_shap_replay_buffer')

    run_network(nominal_start, env, model)
    # choose_best(env.saved_agents).plot_state_history(style='segmented')
    plot_average_reward_curve(env.saved_agents, 100)
