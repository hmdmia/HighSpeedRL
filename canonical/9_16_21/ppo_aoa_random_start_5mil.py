import numpy as np
from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.aoa_base_class import AoABaseClass
from backend.utils.analysis import choose_best, plot_average_reward_curve, run_network
from backend.call_backs.tensorboard_callback import TensorboardCallback
from backend.utils.hyperparams import load_ppo_hyperparams


nominal_start = np.array([30000, 0, 0, 3000, 0*3.14159/180, 0])
variation = np.array([5000, 0, 0, 500, 2.5*3.14159/180, 0])

rand_gen = np.random.default_rng()


class AoARandomStart(AoABaseClass):
    def __init__(self, initial_state=nominal_start):
        AoABaseClass.__init__(self, initial_state)

        self._fpa_tol = 5 * np.pi / 180

        self.n_actions = 21
        self._aoa_options = np.linspace(-20, 20, self.n_actions)/180*np.pi
        self.dt = 1

        self._max_time = 100

        self._target_altitude_threshold = 15000

        # Define event to end integration to set done flag
        self.training_events = [self.generate_emergency_descent_event(trigger_alt=self._target_altitude_threshold)]

    def reset(self, initial_state=None):

        if initial_state is None:
            initial_state = np.random.normal(nominal_start, variation)

        self.__init__(initial_state=initial_state)

    def reward(self):
        return self._reward5()

    def _inner_step(self, action):
        u = np.array([self._aoa_options[action], 0.])
        self.constant_step(self.dt, u)
        self.sim_step(self.dt)


if __name__ == '__main__':
    env = DiscreteEnv(AoARandomStart())
    hyperparams = load_ppo_hyperparams("../../saved_hyperparams/ppo_absolute_params")

    model = PPO('MlpPolicy', env, verbose=1, **hyperparams, tensorboard_log="../../tmp/Canonical/")
    model.learn(total_timesteps=5000000, callback=TensorboardCallback())
    model.save('ppo_aoa_random_start')

    plot_average_reward_curve(env.saved_agents, 100)
    run_network(nominal_start, env, model)
