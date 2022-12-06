import numpy as np

import gym
from gym import spaces
import copy


class BoxEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, agent):
        super(BoxEnv, self).__init__()

        self.agent = agent

        # Since SB3 recommends all actions be normalized [-1, 1], agent specifies number of control actions
        # which are sent to Box in range [-1, 1].
        ctrl_shape = (self.agent.num_ctrl,)
        self.action_space = spaces.Box(low=-np.ones(ctrl_shape), high=np.ones(ctrl_shape), shape=ctrl_shape)
        self.observation_space = spaces.Box(self.agent.low, self.agent.high, shape=self.agent.low.shape)

        self.saved_agents = []

        self.observation = None
        self.reward = None
        self.done = False

    def step(self, action):
        self.observation, self.reward, info = self.agent.rl_step(action)
        self.done = self.agent.done
        return self.observation, self.reward, self.done, info

    def reset(self, **kwargs):

        if self.agent.done:
            self.save_episode()

        self.agent.reset(**kwargs)
        observation = self.agent.observe()

        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        self.agent.plot_state_history()

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
