import numpy as np
from stable_baselines3 import TD3

from backend.rl_environments.box_environment import BoxEnv
from backend.rl_base_classes.aoa_base_class import AoABaseClass
from backend.utils.analysis import choose_best, plot_average_reward_curve, run_network
from backend.call_backs.tensorboard_callback import TensorboardCallback

nominal_start = np.array([30000, 0, 0, 3000, 0, 0])


class AoAAbsolute(AoABaseClass):
    def __init__(self, initial_state=nominal_start):
        AoABaseClass.__init__(self, initial_state)

        self._fpa_tol = 5 * np.pi / 180
        self.num_ctrl = 1
        self.min_ctrl = -20 * np.pi/180
        self.max_ctrl = 20 * np.pi/180
        self.dt = 1

    def reset(self, initial_state=None):

        if initial_state is None:
            initial_state = nominal_start

        self.__init__()

    def reward(self):
        return self._reward3()

    def _inner_step(self, action):
        # Convert normalized action [-1, 1] to radians [min_ctrl, max_ctrl]
        alpha = 0.5 * (action[0] * (self.max_ctrl - self.min_ctrl) + (self.min_ctrl + self.max_ctrl))
        u = np.array([alpha, 0.])
        self.constant_step(self.dt, u)
        self.sim_step(self.dt)


if __name__ == '__main__':
    env = BoxEnv(AoAAbsolute())

    model = TD3('MlpPolicy', env, verbose=1, tensorboard_log="../../tmp/TD3_runs/")
    model.learn(total_timesteps=10000, callback=TensorboardCallback())
    model.save('../../tmp/TD3_runs/latest_model')

    run_network(nominal_start, env, model)
    choose_best(env.saved_agents).plot_state_history(style='segmented')
    plot_average_reward_curve(env.saved_agents, 100)
