import optuna
from torch import nn

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from backend.rl_environments.discrete_environment import DiscreteEnv
from a2c_aoa_absolute import AoAAbsolute
from backend.utils.hyperparams import save_hyperparams


def a2c_param_set(trial):
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    n_steps = trial.suggest_categorical('n_steps', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform('vf_coef', 0, 1)

    return {
        'n_steps': n_steps,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef
    }


def optimize_a2c_agent(trial):
    model_params = a2c_param_set(trial)
    env = DiscreteEnv(AoAAbsolute())
    model = A2C('MlpPolicy', env, verbose=0, **model_params, tensorboard_log="../../tmp/a2c_absolute_study")
    model.learn(total_timesteps=100000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)

    return mean_reward


study = optuna.create_study(direction='maximize')
study.optimize(optimize_a2c_agent, n_trials=30)

best_params = study.best_params
print(best_params.items())
save_hyperparams(best_params, "../../saved_hyperparams/a2c_absolute_params")
