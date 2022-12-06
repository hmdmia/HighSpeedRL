import os
from stable_baselines3 import PPO
from trained_model_visual import save_lhs_array, save_alt_array, save_vel_array, save_fpa_array, save_all_array

from rl_runners.aoa_runners.ppo_aoa_random_start import AoARandomStart
from backend.rl_environments.discrete_environment import DiscreteEnv


import seaborn as sns
import pandas as pd
import numpy as np

i_alt = []
i_vel = []
i_fpa = []
alt = []
fpa = []
success = []

def endpoint_csv(dir, file):
    '''
    Create CSV file containing ICs and final alt and fpa

    :param dir: directory to access csv file
    :param file: txt file containing npy filenames

    :return: CSV file
    '''
    target_alt = 3000

    d1 = open(dir+file, 'r')
    for i in d1:
        file = dir + i.strip() + '_state.npy'
        rl = np.load(file, 'r')
        i = 0
        while rl[0,i] >= 3000:
            i+= 1
            if rl[0,i] <= 3000:
                alt.append(rl[0, i])
            elif rl[0,-1] >= 3000:
                alt.append(rl[0, -1])
                break
        i_alt.append(rl[0, 0])
        i_vel.append(rl[3,0])
        i_fpa.append(round((rl[4, 0])/(3.14159/180),1))
        fpa.append(rl[4, -1] / (3.14159 / 180))

        if rl[0, -1] >= target_alt:
            success.append(0)
        elif abs(rl[4, -1]) * (180 / np.pi) >= 0.25 * np.pi / 180:
            success.append(0)
        else:
            success.append(1)

#creates L-H sample by varying ICs centered around trained ICs
env = DiscreteEnv(AoARandomStart())

model = PPO.load('../trained_agents/ppo_aoa_random_start_25_1.zip', env=env)
save_all_array(env, model,  dir= "run_data/agent_library/25_")



var1 = 25
var2 = 35
l = [15, 20, 30]
f = 'all'
path = 'run_data\\agent_library\\'
path1 = 'csv\\agent_library\\'

if os.path.exists(path1) == False:
    os.mkdir(path1)

#creates L-H sample by varying ICs centered around trained ICs

# for i in range(len(l)):
endpoint_csv(path+'25_\\', file = f+'.txt')
d = {'outcome': success, 'start_altitude': i_alt, 'f_altitude': alt, 'initial_velocity' : i_vel, 'initial_fpa' : i_fpa, 'f_fpa': fpa}
df = pd.DataFrame(data=d)
df.to_csv('csv\\agent_library\\agent_'+ str(var1) +'__all.csv')
#
df = df[['start_altitude', 'outcome']]
df = df.rename(columns={'start_altitude': 'agent_25'})
df_filtered = df[df['outcome'] != 0]
df_filtered = df_filtered.drop(columns='outcome')

#plot kernal density
sns.kdeplot(data=df_filtered, fill = True, common_norm=False, palette='crest', alpha=0.5)
sns.displot(df_filtered, kde=True)
