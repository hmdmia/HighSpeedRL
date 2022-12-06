import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import PPO
from backend.rl_environments.discrete_environment import DiscreteEnv
from rl_runners.aoa_runners.ppo_aoa_absolute import AoAAbsolute

def prediction(obs):
    action, _states = model.predict(obs, deterministic=True)
    action = 17
    return action, _states

# Create environment
env = DiscreteEnv(AoAAbsolute())
# Load the trained agent

#model = PPO.load('../rl_runners/aoa_runners/ppo_aoa_absolute_LHS_random', env=env)
model = PPO.load('../../../trained_agents/ppo_aoa_random_start_uni.zip', env=env)

reward = 0.
obs1 = []
timer = 0

perturb = [28, 32, 33, 34]

obs = env.reset()
done = False
obs1.append(obs[1])
while not done:
    if obs[0] in perturb:
        action, _states = prediction(obs)
    else:
        action, _states = model.predict(obs,deterministic=True)
    obs, reward, done, info = env.step(action)
    if done == True:
        if obs[1] <= 3000:
            pass
        else:
            env.agent.save_run_data('action', save='act_impulse', dir='at')

timer += 1