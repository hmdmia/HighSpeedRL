import os

import optuna
import joblib
import numpy as np

from copy import deepcopy

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold,\
    StopTrainingOnNoModelImprovement

from backend.rl_environments.multi_discrete_environment import MultiDiscreteEnv
from backend.rl_base_classes.non_mp_base_classes import MovingTargetNonMP
from backend.base_aircraft_classes.target_classes import RandomMovingTarget
from rl_runners.mp_runners.moving_target_training import initial_state, target_location
from backend.utils.hyperparams import save_hyperparams
from backend.utils.hyperparams import activation_options
from backend.utils.hyperparams import on_policy_net_arch_size

target = RandomMovingTarget(target_location, v_range=(0, 0), psi_range=(0, 0), distribution="uniform")
env = Monitor(MultiDiscreteEnv(MovingTargetNonMP(initial_state, target)))
dt = env.unwrapped.agent.dt

reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=0)
stop_no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=int(30), verbose=0)

continue_threshold = 150
mean_velocity = 250
# velocity_curriculum = np.linspace(0, 250, 5)
# psi_curriculum = np.deg2rad(np.linspace(0, 180, 5))
velocity_curriculum = [250]
psi_curriculum = np.deg2rad([180])

def param_set(trial):
    batch_size = trial.suggest_int(name="batch_size", low=256, high=2048, step=32)
    step_multiplier = trial.suggest_int(name="step_multiplier", low=2, high=6)
    gamma = trial.suggest_float(name="gamma", low=0.9, high=0.99999)
    learning_rate = trial.suggest_float(name="lr", low=1e-6, high=0.9, log=True)
    arch_size = trial.suggest_int(name="arch_size", low=256, high=1024, step=32)
    ent_coef = trial.suggest_float(name="ent_coef", low=1e-5, high=0.1, log=True)
    clip_range = trial.suggest_float(name="clip_range", low=0.01, high=0.3)
    n_epochs = trial.suggest_int(name="n_epochs", low=20, high=40, step=2)
    gae_lambda = trial.suggest_float(name="gae_lambda", low=0.75, high=0.99)
    max_grad_norm = trial.suggest_float(name="max_grad_norm", low=0.01, high=5)
    vf_coef = trial.suggest_float(name="vf_coef", low=0.01, high=0.99)
    ortho_init = trial.suggest_categorical(name="ortho_init", choices=[True, False])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    net_arch = on_policy_net_arch_size(arch_size)
    activation_fn = activation_options(activation_fn)

    hyperparams = {
        "n_steps": batch_size * step_multiplier,
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


def optimize_agent(trial):
    model_params = param_set(trial)

    eval_callback_freq = model_params['n_steps']
    while eval_callback_freq < 500:
        eval_callback_freq += eval_callback_freq

    log_path = 'tmp/tuning/non_mp/PPO' + str(trial.number + 1)
    model_path = log_path + '/best_model'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    save_hyperparams(trial.params, log_path + '/hyperparams')

    model = PPO('MlpPolicy', env, verbose=0, **model_params, tensorboard_log=log_path)
    model.save(model_path)

    model_goodness = 0
    successful_trials = 0

    print(f'Trial #{trial.number}: {trial.params.items()}')
    for delta_velocity, max_psi in zip(velocity_curriculum, psi_curriculum):
        v_range = (mean_velocity - delta_velocity, mean_velocity + delta_velocity)
        psi_range = (-max_psi, max_psi)

        env.unwrapped.agent.target = RandomMovingTarget(target_location, v_range, psi_range, distribution="uniform")
        model = PPO.load(model_path, env)
        eval_callback = EvalCallback(deepcopy(env),
                                     best_model_save_path=log_path,
                                     eval_freq=eval_callback_freq, n_eval_episodes=25, verbose=0,
                                     callback_on_new_best=reward_threshold_callback,
                                     callback_after_eval=stop_no_improvement_callback)
        model.learn(total_timesteps=int(10e6), callback=eval_callback)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        model_goodness += (mean_reward - std_reward) * (successful_trials + 1)
        if mean_reward < continue_threshold:
            print(f'Mean Reward = {mean_reward} < {continue_threshold}, '
                  f'breaking after v_max = {delta_velocity}, psi_max = {max_psi}...')
            break
        else:
            print(f'Mean Reward = {mean_reward} > {continue_threshold}, '
                  f'continuing after v_max = {delta_velocity}, psi_max = {max_psi}...')
            successful_trials += 1

    print(f'Model Goodness: {model_goodness}')

    return model_goodness


study = optuna.create_study(direction='maximize')
# study = joblib.load("tmp/tuning/non_mp/study.pkl")
study.optimize(optimize_agent, n_trials=100)

best_params = study.best_params
print(best_params.items())
save_hyperparams(best_params, "saved_hyperparams/non_mp")

joblib.dump(study, "tmp/tuning/non_mp/study.pkl")
