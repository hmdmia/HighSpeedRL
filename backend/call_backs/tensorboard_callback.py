import torch
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.terminal_alt_dist = []

    def _on_step(self) -> bool:
        """
        Set values for Tensorboard to display on every step
        :return: True
        """
        # Execute if done flag from agent is True (Index is needed because list is returned for when in parallel)
        if self.training_env.get_attr('done')[0]:
            terminal_obs = self.training_env.get_attr('observation')[0]
            self.logger.record('terminal_stats/episode_duration', terminal_obs[0])
            self.logger.record('terminal_stats/terminal_alt', terminal_obs[1])
            self.logger.record('terminal_stats/terminal_velocity', terminal_obs[2])
            self.logger.record('terminal_stats/terminal_fpa', terminal_obs[3])
            self.terminal_alt_dist.append(terminal_obs[0])

        return True

    def _on_rollout_end(self) -> None:
        """
        This function is called every time the policy is updated.
        The resulting histogram is just used to provide an example
        """
        self.logger.record('terminal_alt', torch.from_numpy(np.array(self.terminal_alt_dist)))
