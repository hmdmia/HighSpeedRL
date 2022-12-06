# daf_test_start.py - DAF connection test based on ppo_aoa_random_start.py

import logging
import numpy as np
from stable_baselines3 import PPO

from backend.rl_base_classes.aoa_base_class import AoABaseClass
from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.utils.analysis import plot_average_reward_curve, run_network
from backend.utils.daf_client import DafClient
from backend.utils.logger import start_log

start_log(logging.DEBUG)  # Change to ...INFO or ...DEBUG as needed
log = logging.getLogger('a4h')

nominal_start = np.array([30000, 0, 0, 3000, -10*3.14159/180, 0])
variation = np.array([5000, 0, 0, 500, 10*3.14159/180, 0])
rand_gen = np.random.default_rng()


class DafTestStart(AoABaseClass):
    def __init__(self, initial_state=nominal_start):
        log.info('__init__()')
        AoABaseClass.__init__(self, initial_state)

        self._fpa_tol = 5 * np.pi / 180

        self.n_actions = 21
        self._aoa_options = np.linspace(-20, 20, self.n_actions)/180*np.pi
        self.dt = 1

        self._max_time = 100

        self.resetc = 0
        self.instep = 0
        self.daf = DafClient()

    def reset(self, initial_state=None):
        self.resetc += 1
        log.info('reset(%d)' % self.resetc)

        if initial_state is None:
            initial_state = np.random.normal(nominal_start, variation)

        self.__init__(initial_state=initial_state)

        if self.resetc > 1:
            result_msg = client.receive()
            log.info('Sim result: '+result_msg)

        log.info('Requesting sim run #%d...' % self.resetc)
        runner_params = self.daf.run_sim('daf_sim.TestRunner', self._max_time)
        log.info('Received params %s...' % runner_params)

    def reward(self):
        return self._reward3()

    def _inner_step(self, action):
        self.instep += 1
        log.info('_inner_step(%d)' % self.instep)

        u = np.array([self._aoa_options[action], 0.])
        self.constant_step(self.dt, u)
        self.sim_step(self.dt)

        self.daf.send_action(0)  # Action 0: Continue sim


integrated_test = False


if integrated_test:  # SB3 and DAF
    env = DiscreteEnv(DafTestStart())
    log.info('Created env')

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10_000)  # was 250_000
    model.save('daf_test_start')
    log.info('Created model')

    log.info('Pre run')
    plot_average_reward_curve(env.saved_agents, 100)
    run_network(nominal_start, env, model)
    log.info('Post run')

    # TODO Better way than reaching into env.agent.daf?

    msg = env.agent.daf.receive()
    log.info('Sim result: '+msg)

    log.info('Requesting DAF exit...')
    env.agent.daf.exit_daf()

else:  # DAF only
    log.info('Creating DAF connection...')
    client = DafClient()
    num_runs = 3
    sim_secs = 10

    for run in range(num_runs):
        log.info('Requesting sim run #%d...' % (run+1,))
        params = client.run_sim('daf_sim.TestRunner', sim_secs)
        log.info('Received params %s...' % params)

        for sec in range(sim_secs):
            state = client.get_state()

            # Simple example of actions: terminate or continue
            if run == 1 and sec == 5:
                client.send_action(1)  # Action 1: Terminate sim early
                break
            else:
                client.send_action(0)  # Action 0: Continue

        msg = client.receive()
        log.info('Sim result: '+msg)

    log.info('Requesting DAF exit...')
    client.exit_daf()
