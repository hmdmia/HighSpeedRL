import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.aoa_base_class import AoABaseClass


nominal_start = np.array([30000, 0, 0, 3000, 0, 0])
stdvs = [0.00]
angle = []
su_ctr = 0

'''alter angle of attack to select a value around the selected action for creation of SHAP dataset'''

class AoAModification(AoABaseClass):
    def __init__(self, initial_state=nominal_start):
        AoABaseClass.__init__(self, initial_state)

        self._fpa_tol = 5 * np.pi / 180

        self.n_actions = 21
        self._aoa_options = np.linspace(-20, 20, self.n_actions) / 180 * np.pi
        self.dt = 1

    def reset(self, initial_state=None):
        if initial_state is None:
            initial_state = nominal_start

        self.__init__(initial_state)

    def reward(self):
        return self._reward5()

    def _inner_step(self, action):


        #uses a random normal distribution to select a different aoa command
        sampling = np.random.normal(self._aoa_options[action]* 180 / np.pi, 5, 10)
        value = random.choice(sampling)/180 * np.pi
        u = np.array([value, 0.])


        self.constant_step(self.dt, u)
        self.sim_step(self.dt)
        angle.append(u[0]*180/np.pi)

'''Remove failed runs from dataframe'''

def delete_fail(obs0,obs1,obs2,obs3,rewards,dones,angle,actions):
    j = -1
    while obs0[j] != 0:
        del obs0[j],obs1[j], obs2[j],obs3[j],rewards[j],dones[j],angle[j],actions[j]
    del obs0[j], obs1[j], obs2[j], obs3[j], dones[j]


if __name__ == '__main__':
    # Create environment
    env = DiscreteEnv(AoAModification())
    # Load the trained agent
    model = PPO.load('../../../../trained_agents/ppo_aoa_random_start_uni', env=env)

    obs0 = []
    obs1 = []
    obs2 = []
    obs3 = []
    rewards = []
    dones = []
    actions = []

    while su_ctr < 1000:
        done = False
        obs = env.reset(initial_state=nominal_start)

        obs = env.reset()
        reward = 0.
        done = False
        obs0.append(obs[0])
        obs1.append(obs[1])
        obs2.append(obs[2])
        obs3.append(obs[3])

        dones.append(0)

        while not done:
            action, _states = model.predict(obs,deterministic=True)
            actions.append(action)
            obs, reward, done, info = env.step(action)

            obs0.append(obs[0])
            obs1.append(obs[1])
            obs2.append(obs[2])
            obs3.append(obs[3])
            dones.append(done)
            rewards.append(reward)
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

    Xy.to_csv('shap_aoa_perturb.csv')