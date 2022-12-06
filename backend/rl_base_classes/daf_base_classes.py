import numpy as np

from backend.rl_base_classes.mp_base_classes import FPATrimsAndTurns
from backend.utils.daf_client import DafClient
from backend.utils.misc import wrap_ang


class DafBaseClass(FPATrimsAndTurns):
    def __init__(self, initial_state, target_state, matlab_runner, total_episodes=1, new_observation_space=False):
        FPATrimsAndTurns.__init__(self, initial_state, target_state[0:3])
        self.target_state = target_state
        self.dist = []

        if target_state[3] > 1e-3:
            self.isMoving = True
        else:
            self.isMoving = False

        self.resetc = 0
        self.daf_client = DafClient()
        self.daf_logging = False
        self.is_continuous_mp = False  # True for continuous actions space, False for discrete
        self.matlab_runner = matlab_runner
        self.total_episodes = total_episodes

        self.uniform_target_distribution = False  # set True if uniformly distributed target state is desired
        self.target_min_velocity = []
        self.target_max_velocity = []
        self.target_min_psi = []
        self.target_max_psi = []

        # Initialization of curriculum learning. To use curriculum, call generate_curriculum before learning.
        self.last_stage = 1  # tracks last stage so that at each new stage properties may be reset
        self.curriculum = self.generate_curriculum(stages=1)

        # Expand obserables to include target v & psi in moving case.
        if self.isMoving:
            self.low = np.append(self.low, [0, -np.deg2rad(180)])
            self.high = np.append(self.high, [self.max_vel, np.deg2rad(180)])

        self.new_observation_space = new_observation_space

    def python_params(self):
        python_params = {
            'initialState': self.initial_state.tolist(),
            'targetState':  self.target_state.tolist(),
            'captureDistance': self.target_tol,
            'altitudeTolerance': self.alt_tol,
            'isLogging': self.daf_logging,
            'mpList': self.mp_options,
            'isContinuousMP': self.is_continuous_mp}
        return python_params

    def generate_curriculum(self, stages):
        # Ensure in no-curriculum case that velocity is correct (numpy defaults to 1st value for linspace).
        if stages <= 1:
            start_vel = self.target_state[3]
            start_target_tol = self.target_tol
            start_alt_tol = self.alt_tol
        else:
            start_vel = 0
            start_target_tol = stages/2 * self.target_tol
            start_alt_tol = stages/2 * self.alt_tol

        curriculum = {
            'stages': stages,
            'target_velocities': np.linspace(start_vel, self.target_state[3], stages),
            'target_tols': np.linspace(start_target_tol, self.target_tol, stages).tolist(),
            'alt_tols': np.linspace(start_alt_tol, self.alt_tol, stages).tolist()}
        return curriculum

    def set_uniform(self, v_list, psi_list):
        self.uniform_target_distribution = True
        self.target_min_velocity = v_list[0]
        self.target_max_velocity = v_list[1]
        self.target_min_psi = psi_list[0]
        self.target_max_psi = psi_list[1]

    def reset(self, initial_state=None):
        # resets agent at end of episode
        FPATrimsAndTurns.reset(self, initial_state)
        self.time_history.append(np.array([]))  # initializes array to put times in
        self.state_history.append(np.empty(shape=(6, 0)))
        self.control_history.append(np.empty(shape=(2, 0)))

        self.resetc += 1

        # Attempts to start another sim
        if self.resetc > 1:
            self.daf_client.receive()

        # Implement Curriculum
        episodes_per_stage = self.total_episodes / self.curriculum['stages']
        stage = int(self.resetc / episodes_per_stage)

        if stage > self.last_stage:
            self.last_stage = stage
        if self.curriculum['stages'] == 1:
            stage = 0

        self.target_state[3] = self.curriculum['target_velocities'][stage]
        self.target_tol = self.curriculum['target_tols'][stage]
        self.alt_tol = self.curriculum['alt_tols'][stage]

        if self.uniform_target_distribution:
            self.target_state[3] = np.random.uniform(self.target_min_velocity, self.target_max_velocity)
            self.target_state[5] = np.random.uniform(self.target_min_psi, self.target_max_psi)

        # outputs MATLAB params, but unused in Python code
        self.daf_client.run_sim(self.matlab_runner, self.max_time, self.python_params())

    def _inner_step(self, action):
        # Sends action to MATLAB to integrate agent
        self.daf_client.send_action(action + 1)  # Send MP, (MATLAB 1-16, Python 0-15)

    def observe(self):
        """
        Method to receive observation from DAF agent
        [altitude, velocity, FPA, distance to target, difference btwn heading & bearing to target]
        In case of moving target, also observe target velocity and heading.
        :return: observation vector
        """
        observables = self.daf_client.get_state()

        # Assign observable flags
        self.success = observables['success']
        self.done = observables['done']

        # Assign variables used to calculate reward
        self.state = observables['stateVector']
        self.time = observables['currentTime']
        self.target_location = observables['targetStateVector'][0:3]

        # Assign history
        current_time = np.array(observables['currentTime'])
        current_state = np.array(observables['stateVector'])
        if observables['control']:
            current_control = np.array(observables['control'])
        else:
            current_control = np.array([np.NaN, np.NaN])

        self.add_history(current_time, current_state, current_control)

        # Assign variables for analysis
        self.dist = observables['dist']

        observation_array = np.array([observables['stateVector'][0], observables['stateVector'][3],
                                      observables['stateVector'][4], observables['surface_dist'],
                                      observables['relativeBearing']])

        if self.isMoving:
            if self.new_observation_space:
                observation_array = np.append(observation_array,
                                              (observables['targetStateVector'][3],
                                              observables['relativeTargetBearing']))
            else:
                observation_array = np.append(observation_array,
                                              (observables['targetStateVector'][3],
                                              observables['targetStateVector'][5]))
        observation_array = (observation_array - self.low) / (self.high - self.low)

        return observation_array
        # returns h, v, gamma, circular dist, relative bearing (non moving case)
        # appends v_tar & psi_tar (moving case)

    def add_history(self, time, state, control=np.array([np.NaN, np.NaN])):
        self.time_history[-1] = np.append(self.time_history[-1], time)
        self.state_history[-1] = np.hstack([self.state_history[-1], np.vstack(state)])
        self.control_history[-1] = np.hstack([self.control_history[-1], np.vstack(control)])


class DafContinuousClass(DafBaseClass):
    def __init__(self, initial_state, target_state, matlab_runner, total_episodes):
        DafBaseClass.__init__(self, initial_state, target_state, matlab_runner, total_episodes)

        self.is_continuous_mp = True  # True for continuous actions space, False for discrete

        self.max_pull = np.deg2rad(1)
        self.min_pull = np.deg2rad(-10)
        self.max_turn = np.deg2rad(10)
        self.min_heading = None
        self.max_heading = None
        self.update_turn_bounds()

        self.num_ctrl = 2

    def update_turn_bounds(self):
        current_heading = self.state[5]
        self.min_heading = current_heading - self.max_turn
        self.max_heading = current_heading + self.max_turn

    def _inner_step(self, action):
        # Convert normalized action [-1, 1] to radians [min_ctrl, max_ctrl]
        min_ctrl = np.array([self.min_pull, self.min_heading])
        max_ctrl = np.array([self.max_pull, self.max_heading])
        gampsi = 0.5 * (action * (max_ctrl - min_ctrl) + (min_ctrl + max_ctrl))
        gampsi[1] = wrap_ang(gampsi[1])

        # Sends action to MATLAB to integrate agent
        self.daf_client.send_action(gampsi)  # Send gam & psi val to MATLAB

    def observe(self):
        observation_array = DafBaseClass.observe(self)
        self.update_turn_bounds()
        return observation_array
