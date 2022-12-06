import numpy as np
from stable_baselines3 import DQN

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.mp_base_classes import RandomStartFPATrimsAndTurns
from backend.utils.analysis import plot_average_reward_curve
from backend.call_backs.mp_tensorboard_callback import MPTensorboardCallback
from backend.utils.hyperparams import load_ppo_hyperparams

# initial_state = np.array([40000, np.deg2rad(35.1), np.deg2rad(-106.6), 6000, 0, np.pi/4])
# target_location = np.array([20000, np.deg2rad(40.4), np.deg2rad(-86.9)])

initial_state = np.array([40000, np.deg2rad(0), np.deg2rad(0), 6000, -np.deg2rad(0.5), np.pi/2])
# initial_variance = np.array([5000, np.deg2rad(2), np.deg2rad(5), 250, np.deg2rad(0.5), np.deg2rad(15)])
initial_variance = np.array([0, 0, 0, 0, 0, 0])
target_location = np.array([20000, np.deg2rad(0), np.deg2rad(20)])
target_variance = np.array([1000, np.deg2rad(5), np.deg2rad(10)])


if __name__ == '__main__':

    def run_network(_initial_state, target, _env, _model):
        done = False
        obs = _env.reset(initial_state=_initial_state, target_location=target)
        while not done:
            action, _state = _model.predict(obs, deterministic=True)
            obs, _, done, __ = _env.step(action)

        _env.agent.plot_state_history(style='3d')

    env = DiscreteEnv(RandomStartFPATrimsAndTurns(initial_state, target_location, initial_variance, target_variance))

    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="../../tmp/mp/DQN/")
    model.learn(total_timesteps=50000, callback=MPTensorboardCallback())
    model.save('dqn_random_mp_fpa_trim_and_turns')

    plot_average_reward_curve(env.saved_agents, 100)
    # run_network(initial_state, target_location, env, model)
    run_network(initial_state, target_location + target_variance, env, model)
