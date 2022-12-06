import gym
from gym import spaces
import copy


class MultiDiscreteEnv(gym.Env):
    """
    Training environment for problems with a multi-discrete action space (possible control options)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, agent):
        """
        Initialize environment

        :param agent: Simulation agent with which to train
        """
        super(MultiDiscreteEnv, self).__init__()

        self.agent = agent

        self.action_space = spaces.MultiDiscrete(self.agent.n_actions)
        self.observation_space = spaces.Box(self.agent.low, self.agent.high)

        self.saved_agents = []

        self.observation = None
        self.reward = None
        self.done = False

    def step(self, action):
        """
        Execute RL step with specified action
        Called by training algorithm

        :param action:
        :return: observation vector, reward for step, done flag, additional info
        """
        self.observation, self.reward, info = self.agent.rl_step(action)
        self.done = self.agent.done
        return self.observation, self.reward, self.done, info

    def reset(self, **kwargs):
        """
        Reset environment to state for next training episode

        :param kwargs: keyword arguments needed for reset method on agent
        :return: observation vector for first step
        """

        if self.agent.done:
            self.save_episode()

        self.agent.reset(**kwargs)
        observation = self.agent.observe()

        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        """
        Render episode (Not really used in our implementation yet
        :param mode:
        :return:
        """
        # self.agent.plot_state_history()
        pass

    def save_episode(self):
        """
        Save data from episode for later analysis
        :return:
        """

        # TODO Resolve below workaround for "TypeError("Cannot serialize socket object")" -wlevin 07/13/2021
        if hasattr(self.agent, 'daf_client'):
            daf_client = self.agent.daf_client  # Workaround
            self.agent.daf_client = []   # Workaround
            self.saved_agents.append(copy.deepcopy(self.agent))
            self.agent.daf_client = daf_client  # Workaround
        else:
            self.saved_agents.append(copy.deepcopy(self.agent))
