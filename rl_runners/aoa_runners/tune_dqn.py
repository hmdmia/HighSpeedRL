import optuna

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from backend.rl_environments.discrete_environment import DiscreteEnv
from dqn_aoa_relative import AoARelative
from backend.utils.hyperparams import save_hyperparams
from backend.utils.hyperparams import dqn_net_arch_options
from backend.utils.hyperparams import activation_options
from backend.call_backs.tensorboard_callback import TensorboardCallback


def dqn_param_set(trial):
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    net_arch = dqn_net_arch_options(net_arch)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = activation_options(activation_fn)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn),
    }

    return hyperparams


def optimize_rl_agent(trial):
    model_params = dqn_param_set(trial)
    env = DiscreteEnv(AoARelative())
    model = DQN('MlpPolicy', env, verbose=0, **model_params, tensorboard_log="../../tmp/dqn_aoa_relative_study")
    model.learn(total_timesteps=500000, callback=TensorboardCallback())
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    return mean_reward


study = optuna.create_study(direction='maximize')
study.optimize(optimize_rl_agent, n_trials=100)

best_params = study.best_params
print(best_params.items())
save_hyperparams(best_params, "../../saved_hyperparams/dqn_relative_params")
