from typing import TypedDict, Union
import torch
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class MPTensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0,
                 reward_failure_threshold: Union[int, float] = -np.inf,
                 min_improvement: Union[int, float] = -np.inf,
                 stop_on_overtrain: bool = False):
        super(MPTensorboardCallback, self).__init__(verbose)
        self.reward_failure_threshold = reward_failure_threshold
        self.min_improvement = min_improvement
        self.stop_on_overtrain = stop_on_overtrain

        self.rollouts = 0
        self.continue_training = True
        self.max_ave_rew = -np.inf
        self.num_rollouts_between_checks = 30
        self.step = 0

    def _on_step(self) -> bool:
        """
        Set values for Tensorboard to display on every step
        :return: True
        """
        # Execute if done flag from agent is True (Index is needed because list is returned for when in parallel)
        if self.training_env.get_attr('done')[0]:
            terminal_obs = self.training_env.get_attr('observation')[0]
            self.logger.record('terminal_stats/dist_to_tar', terminal_obs[3])
            self.logger.record('terminal_stats/bear_to_target', terminal_obs[4])
            self.logger.record('terminal_stats/terminal_alt', terminal_obs[0])
            self.logger.record('terminal_stats/terminal_velocity', terminal_obs[1])
            self.logger.record('terminal_stats/terminal_fpa', terminal_obs[2])
        return self.continue_training

    def _on_rollout_end(self) -> None:
        """
        This function is called every time the policy is updated.
        The resulting histogram is just used to provide an example
        """
        self.rollouts += 1

        rewards = self.training_env.envs[0].get_episode_rewards()
        num_rewards = 100

        print(f'\n\nRollout: {self.rollouts}\n\n')

        if self.rollouts % self.num_rollouts_between_checks == 0 and len(rewards) > num_rewards:
            ave_reward = np.mean(rewards[-num_rewards:-1])

            # TODO remove this print:
            print(f'Rollout: {self.rollouts}, average reward: {ave_reward}\n\n')

            if ave_reward < self.reward_failure_threshold:
                self.continue_training = False
                print(f'Model failed to train to mean reward > {self.reward_failure_threshold} '
                      f'in last {self.num_rollouts_between_checks} rollouts.')

            if ave_reward - self.max_ave_rew < self.min_improvement and self.stop_on_overtrain:
                self.continue_training = False
                print(f'Model failed to improve by {self.min_improvement} '
                      f'in last {self.num_rollouts_between_checks} rollouts.')

            if ave_reward > self.max_ave_rew:
                self.max_ave_rew = ave_reward

