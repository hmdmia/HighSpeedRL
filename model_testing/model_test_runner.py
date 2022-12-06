from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from rl_runners.aoa_runners.ppo_aoa_absolute import AoAAbsolute
from backend.utils.analysis import run_network_stochastic

# TODO: account for different algorithms in an easy way
if __name__ == '__main__':
    env = DiscreteEnv(AoAAbsolute())
    model = PPO.load("../tmp/ppo_aoa_absolute")  # model to be tested
    num_eps = 1000

    run_network_stochastic(model, env, num_eps)

    # deterministic run
    done = False
    obs = env.reset()
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, _, done, __ = env.step(action)

    env.agent.plot_state_history(style='segmented')
