from abc import ABC, abstractmethod


class RLBaseClass(ABC):
    """
    Abstract base class to define interfaces of agent to be used within RL environment
    """
    def __init__(self):

        self.reward_total = 0

        self.n_actions = None
        self.low = None
        self.high = None

        # Flags
        self.done = False
        self.success = False
        self.reached_alt = False

    @abstractmethod
    def observe(self):
        """
        Method to return observation
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Method to reset agent
        :return:
        """
        pass

    @abstractmethod
    def reward(self):
        """
        Method to return reward at step
        :return:
        """
        pass

    @abstractmethod
    def _inner_step(self, action):
        """
        Overwrittable method to execute problem specific code within self.rl_step() method
        :param action:
        :return:
        """
        pass

    def rl_step(self, action):
        """
        Executes a step in the RL context, maps action (at current time and state) to observation and reward
        :param action: action choosen by network/training
        :return:
        """

        self._inner_step(action)

        observation = self.observe()
        reward = self.reward()
        self.reward_total += reward

        info = {}

        return observation, reward, info

