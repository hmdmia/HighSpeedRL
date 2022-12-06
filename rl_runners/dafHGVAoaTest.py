import logging
import numpy as np
import sys

from stable_baselines3 import PPO

from backend.rl_base_classes import AoABaseClass
from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.utils.analysis import plot_average_reward_curve
from backend.utils.daf_client import DafClient
from backend.utils.logger import start_log
from backend.call_backs.tensorboard_callback import TensorboardCallback
from backend.utils.hyperparams import load_ppo_hyperparams

# state = [altitude, theta, phi, velocity, gamma, psi]

nominal_start = np.array([30000, 0, 0, 3000, 0, 0])
PYTHON_PARAMS = {'initialState': nominal_start, 'isLogging': False}
MATLAB_RUNNER = 'a4h.runners.hgvAoaTest'
EARLY_EXIT_ACTION = 86


class DafTestStart(AoABaseClass):
    def __init__(self, initial_state):
        AoABaseClass.__init__(self, initial_state=nominal_start)

        self.resetc = 0
        self.daf_client = DafClient()
        self.MATLAB_RUNNER = MATLAB_RUNNER
        self.PYTHON_PARAMS = PYTHON_PARAMS
        self.PYTHON_PARAMS['initialState'] = initial_state.tolist()


    def reset(self, initial_state=None):
        # resets agent at end of episode
        if initial_state is None:
            initial_state = nominal_start

        AoABaseClass.__init__(self, initial_state)
        self.resetc += 1

        # Attempts to start another sim
        if self.resetc > 1:
            self.daf_client.receive()

        self.daf_client.run_sim(self.MATLAB_RUNNER, self._max_time, self.PYTHON_PARAMS)

    def reward(self):
        return self._reward5()

    def _inner_step(self, action):
        # Sends action to MATLAB (angle in radians)
        self.daf_client.send_action(float(self._aoa_options[action]))

    def observe(self):
        """
        Method to receive observation from DAF agent

        :return: observation vector
        """
        observables = self.daf_client.get_state()
        # h, theta, phi, v, gam, psi = self.state

        # Assign observable flags
        self.done = observables['done']
        self.success = observables['success']

        # Assign variables used to calculate reward
        self.state = observables['stateVector']
        self.time = observables['currentTime']

        return np.array([observables['currentTime'], observables['stateVector'][0],
                        observables['stateVector'][3], observables['stateVector'][4]])
        # returns t, h, v, gamma


if __name__ == '__main__':

    if len(sys.argv) > 1 and \
            (sys.argv[1].lower() == '-d' or sys.argv[1].lower() == '--debug'):
        start_log(logging.DEBUG)
    else:
        start_log(logging.INFO)

    log = logging.getLogger('a4h')
    full_run = False  # true for long runs with logging/saving model

    env = DiscreteEnv(DafTestStart(initial_state=nominal_start))
    hyperparams = load_ppo_hyperparams("saved_hyperparams/ppo_absolute_params")

    if full_run:
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='../tmp/DAF/dafHGVAoaTest')
        model.learn(total_timesteps=500_000, callback=TensorboardCallback())
        model.save('../tmp/DAF/dafHGVAoaTestAgent')
    else:  # Quick runs without logging/saving model
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=2_048)

    env.agent.daf_client.send_action(EARLY_EXIT_ACTION)
    env.agent.daf_client.receive()  # Get end msg
    env.agent.daf_client.exit_daf()

    plot_average_reward_curve(env.saved_agents, 100)
