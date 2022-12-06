import optuna
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from backend.rl_environments.discrete_environment import DiscreteEnv
from ppo_aoa_random_start import AoARandomStart
from backend.utils.hyperparams import save_hyperparams
from backend.utils.hyperparams import activation_options
from backend.utils.hyperparams import ppo_net_arch_options


def ppo_param_set(trial: optuna.trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    gamma = trial.suggest_uniform("gamma", 0.9, 0.99)
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    # lr_schedule = "constant"
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00001, 0.1)
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.4)
    n_epochs = trial.suggest_int("n_epochs", 1, 20, log=True)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)
    max_grad_norm = trial.suggest_loguniform("max_grad_norm", 0.3, 5)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    ortho_init = False
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    net_arch = ppo_net_arch_options(net_arch)

    activation_fn = activation_options(activation_fn)

    hyperparams = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

    return hyperparams


def optimize_ppo_agent(trial):
    model_params = ppo_param_set(trial)
    env = DiscreteEnv(AoARandomStart())
    model = PPO('MlpPolicy', env, verbose=0, **model_params, tensorboard_log="../../tmp/ppo_random_absolute_study")
    model.learn(total_timesteps=200000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

    return mean_reward


study = optuna.create_study(direction='maximize')
study.optimize(optimize_ppo_agent, n_trials=64)

best_params = study.best_params
print(best_params.items())
save_hyperparams(best_params, "../../saved_hyperparams/ppo_random_start_params")
