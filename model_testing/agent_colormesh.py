import pandas as pd
import numpy as np
import math
from scipy.interpolate import griddata
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

dt = pd.read_csv('csv/agent_library/agent_25_35_15all.csv')
# df = pd.read_csv('csv/agent_library/agent_25_all.csv')
# df = pd.read_csv('csv/agent_library/agent_25_35__all.csv')
df = pd.read_csv('csv/agent_library/agent_25_35__all.csv')

dt = dt[['start_altitude','initial_velocity', 'initial_fpa', 'outcome']]
df = df[['start_altitude','initial_velocity', 'initial_fpa', 'outcome']]

df = df[df['outcome'] != 1]

x1 = np.linspace(df['start_altitude'].min(), df['start_altitude'].max(), len(df['start_altitude'].unique()))
y1 = np.linspace(df['initial_velocity'].min(), df['initial_velocity'].max(), len(df['initial_velocity'].unique()))

x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((df['start_altitude'], df['initial_velocity']), df['initial_fpa'], (x2, y2), method='cubic')

fig = plt.figure()
ax = fig.subplots(subplot_kw={"projection":"3d"})
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)


vae = df.values
sh_0, sh_1 = vae.shape

df = df.to_numpy()
#df = df[df[:,0].argsort()]
# df_1

ctr = 0
ctr1 = 0

x, y = np.ones((100,100)) * np.linspace(math.floor(min(df[:,0])), math.floor(max(df[:,0])), 100), np.ones((100,100)) * np.linspace(math.floor(min(df[:,1])), math.floor(max(df[:,1])), 100)

z = np.ones((100,100))

for j in range(sh_0):
    for i in range(len(x)):
        if (df[j, 0] <= x[ctr+1]) & (df[j, 0] >= x[ctr]):
            for k in range(len(y)):
                if (df[j, 1] <= y[ctr1+1]) & (df[j, 1] >= y[ctr1]):
                    z[ctr, ctr1] = df[j, 2]
                    z[ctr+1, ctr1] = df[j, 2] - 0.25
                    z[ctr - 1, ctr1] = df[j, 2] - 0.25
                    z[ctr, ctr1 +1] = df[j, 2] - 0.25
                    z[ctr, ctr1 - 1] = df[j, 2] - 0.25
                ctr1 += 1
                if ctr1 >= 99:
                    ctr1 = 0
        ctr += 1
        if ctr >= 99:
            ctr = 0

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(x, y, z)
