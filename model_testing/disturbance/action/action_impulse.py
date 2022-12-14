import random

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from backend.rl_environments.discrete_environment import DiscreteEnv
from rl_runners.aoa_runners.ppo_aoa_absolute import AoAAbsolute

def prediction(obs):
    """
    Overwrite model action

    :param obs: observation data
    :return: action
    """
    action, _states = model.predict(obs, deterministic=True)
    action = 10
    return action, _states

def delete_fail(obs0,obs1,obs2,obs3,rewards,dones,angle,actions):
    """
    Delete failed run observation data

    :param obs0, obs1, obs2, obs3: observation data
    :param rewards: reward sum
    :param angle: angle that the vehicle moves to
    :param actions: action taken by agent
    :return:
    """
    j = -1
    while obs0[j] != 0:
        del obs0[j],obs1[j], obs2[j],obs3[j],rewards[j],dones[j],angle[j],actions[j]
    del obs0[j], obs1[j], obs2[j], obs3[j], dones[j]

# Create environment
env = DiscreteEnv(AoAAbsolute())
# Load the trained agent
model = PPO.load('../../../trained_agents/ppo_aoa_random_start_uni.zip', env=env)

def action_impulse(env, model, perturb = True, size=100, file = 'aoa_rand_impulse.csv', kf = True):
    """
    Intoduce impulse at random time intervals to action space

    :param env: RL environment used for training
    :param model: RL model generated by training
    :param perturb: True or False to apply action disturbances
    :param rewards: reward sum
    :param size: number of runs to perform
    :param file: filename
    :param kf: True or False to keep failed runs
    :return:
    """
    obs0 = []
    obs1 = []
    obs2 = []
    obs3 = []
    rewards = []
    pert = []
    dones = []
    actions = []
    success = []

    for i in range(size):
        obs = env.reset()

        reward = 0.
        '''
        introduce perturbation at random time
        '''
        if perturb == True:
            perturb = (random.sample(range(30), 10))
            pert.append(perturb)

        done = False
        obs0.append(obs[0])
        obs1.append(obs[1])
        obs2.append(obs[2])
        obs3.append(obs[3])
        dones.append(done)
        rewards.append(reward)

        while not done:
            if obs[0] in perturb:
                action, _states = prediction(obs)
            else:
                action, _states = model.predict(obs,deterministic=True)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            # print(obs[0])
            # env.render()
            '''
            save observation data
            '''
            obs0.append(obs[0])
            obs1.append(obs[1])
            obs2.append(obs[2])
            obs3.append(obs[3])
            dones.append(done)
            rewards.append(reward)
            if kf == True:
                if done == True:
                    if obs1[-1] <= 3000:
                        success.append(1)
                        env.agent.save_run_data('action', save='act_impulse', dir='at')
                    else:

                        success.append(0)
                        env.agent.save_run_data('action', save='act_impulse', dir='at')
                else:
                    success.append('')
            else:
                if done == True:
                    if obs1[-1] <= 3000:
                        success.append(1)
                        env.agent.save_run_data('action', save='act_impulse', dir='at')
                    else:
                        delete_fail(obs0, obs1, obs2, obs3, rewards, dones, actions)
                else:
                    continue
    '''create dataframe and save to an excel file'''

    d = {'Time': obs0, 'Altitude': obs1, 'Velocity': obs2, 'FPA': obs3, 'reward': rewards, 'done': dones}
    df = pd.DataFrame(data=d)
    Xy = df[df['done'] == np.logical_and(True, False)]
    Xy = Xy[['Time', 'Altitude', 'Velocity', 'FPA']]
    Xy['Action'] = actions
    Xy['Success'] = success

    Xy.to_csv(file)