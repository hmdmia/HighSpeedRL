#f add super
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from rl_runners.aoa_runners.ppo_aoa_random_start import AoARandomStart
from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.aoa_base_class import AoABaseClass

'''


'''

angle = []
ctr = 0
su_ctr = 0

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
        angle.append(u[0]*180/np.pi)

def perturb_predict(obs):
    action = model.predict(obs,deterministic=True)[0]
    if action <= 5:
        action +=1
    elif 15 <= action <= 21:
        action -=1
    else:
        pass
    return action

def delete_fail(obs0, obs1, obs2, obs3, rewards, dones, angle, actions):
    j = -1
    while obs0[j] != 0:
        del obs0[j], obs1[j], obs2[j], obs3[j], rewards[j], dones[j], angle[j], actions[j]
    del obs0[j], obs1[j], obs2[j], obs3[j], dones[j]


if __name__ == '__main__':
    # Create environment
    env = DiscreteEnv(AoARandomStart())

    # Load the trained agent
    model = PPO.load('../trained_agents/ppo_aoa_random_start_35.zip', env=env)

    obs0 = []
    obs1 = []
    obs2 = []
    obs3 = []
    rewards = []
    dones = []
    actions = []

    while su_ctr < 1000:
        done = False
        obs = env.reset()
        reward = 0.
        done = False
        obs0.append(obs[0])
        obs1.append(obs[1])
        obs2.append(obs[2])
        obs3.append(obs[3])
        #rewards.append(reward)
        dones.append(0)

        while not done:
            #action = model.predict(obs,deterministic=True)[0]
            action = perturb_predict(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            obs0.append(obs[0])
            obs1.append(obs[1])
            obs2.append(obs[2])
            obs3.append(obs[3])
            dones.append(done)
            rewards.append(reward)
            su_ctr += 1
            if done == True:
                if obs1[-1] <= 3000:
                    su_ctr += 1
                else:
                    delete_fail(obs0,obs1,obs2,obs3,rewards,dones,angle,actions)
            else:
                continue

    d = {'time': obs0, 'altitude': obs1, 'velocity': obs2, 'FPA': obs3, 'done': dones}
    df = pd.DataFrame(data=d)
    Xy = df[df['done'] == False]
    Xy = Xy[['time', 'altitude', 'velocity', 'FPA','done']]

    Xy['action'] = actions
    Xy['aoa'] = angle
    Xy['reward'] = rewards

    Xy.to_csv('ppo_random_start_35.csv')