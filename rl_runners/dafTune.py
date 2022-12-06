import optuna
import joblib
import sys
import logging
import numpy as np

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from backend.utils.logger import start_log
from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_environments.box_environment import BoxEnv
from backend.rl_base_classes.daf_base_classes import DafBaseClass, DafContinuousClass
from rl_runners.dafHGVmpTest import initial_state, target_location, TARGET_VELOCITY, MATLAB_RUNNER, EARLY_EXIT_ACTION
from backend.utils.hyperparams import save_hyperparams
from backend.utils.hyperparams import activation_options
from backend.utils.hyperparams import on_policy_net_arch_size, off_policy_net_arch_size, train_freq_tuple

# Logging
if len(sys.argv) > 1 and \
        (sys.argv[1].lower() == '-d' or sys.argv[1].lower() == '--debug'):
    start_log(logging.DEBUG)
else:
    start_log(logging.INFO)
log = logging.getLogger('a4h')

# Flags
continuous_env = True
policy = "TD3"  # supported options: PPO, SAC, TD3


def ppo_param_set(trial):
    batch_size = trial.suggest_int(name="batch_size", low=32, high=1024)
    gamma = trial.suggest_float(name="gamma", low=0.9, high=0.99999)
    learning_rate = trial.suggest_float(name="lr", low=1e-6, high=1, log=True)
    arch_size = trial.suggest_int(name="arch_size", low=64, high=1024)
    activation_fn = trial.suggest_categorical(name="activation_fn", choices=["tanh", "relu"])

    activation_fn = activation_options(activation_fn)

    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
    }

    if continuous_env and not policy == "TD3":
        use_sde = trial.suggest_categorical(name="ortho_init", choices=[True, False])
        hyperparams["use_sde"] = use_sde

    if policy == "PPO" or policy == "SAC":
        ent_coef = trial.suggest_float(name="ent_coef", low=1e-5, high=0.1, log=True)
        hyperparams["ent_coef"] = ent_coef

    if policy == "PPO":
        step_multiplier = trial.suggest_int(name="step_multiplier", low=1, high=5)
        hyperparams["n_steps"] = batch_size * step_multiplier
        clip_range = trial.suggest_float(name="clip_range", low=0, high=0.3)
        hyperparams["clip_range"] = clip_range
        n_epochs = trial.suggest_int(name="n_epochs", low=1, high=30)
        hyperparams["n_epochs"] = n_epochs
        gae_lambda = trial.suggest_float(name="gae_lambda", low=0.8, high=1)
        hyperparams["gae_lambda"] = gae_lambda
        max_grad_norm = trial.suggest_float(name="max_grad_norm", low=0.1, high=5)
        hyperparams["max_grad_norm"] = max_grad_norm
        vf_coef = trial.suggest_float(name="vf_coef", low=0, high=1)
        hyperparams["vf_coef"] = vf_coef

        ortho_init = trial.suggest_categorical(name="ortho_init", choices=[True, False])
        net_arch = on_policy_net_arch_size(arch_size)
        hyperparams["policy_kwargs"] = dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init)

    if policy == "SAC" or policy == "TD3":
        buffer_size = trial.suggest_int(name="buffer_size", low=1e3, high=1e6)
        hyperparams["buffer_size"] = buffer_size
        learning_starts = trial.suggest_int(name="learning_starts", low=0, high=1e3)
        hyperparams["learning_starts"] = learning_starts
        tau = trial.suggest_float(name="tau", low=0, high=1)
        hyperparams["tau"] = tau
        train_freq_episode = trial.suggest_int(name="train_freq_episode", low=1, high=1e5)
        hyperparams["train_freq"] = train_freq_tuple(train_freq_episode)
        gradient_steps = trial.suggest_int(name="gradient_steps", low=1, high=batch_size)
        hyperparams["gradient_steps"] = gradient_steps

        net_arch = off_policy_net_arch_size(arch_size)
        hyperparams["policy_kwargs"] = dict(
            net_arch=net_arch,
            activation_fn=activation_fn)

    return hyperparams


target_state = np.append(target_location, TARGET_VELOCITY)
v_range = np.array((1, 1500))
psi_range = np.array((-180, 180))

if continuous_env:
    env = BoxEnv(DafContinuousClass(initial_state, target_state, MATLAB_RUNNER, total_episodes=1))
else:
    env = DiscreteEnv(DafBaseClass(initial_state, target_state, MATLAB_RUNNER, total_episodes=1))
env.agent.curriculum = env.agent.generate_curriculum(1)
env.agent.set_uniform(v_range, psi_range)
env = Monitor(env, allow_early_resets=True)


def optimize_ppo_agent(trial):
    model_params = ppo_param_set(trial)
    if policy == "PPO":
        model = PPO('MlpPolicy', env, verbose=0, **model_params, tensorboard_log='tmp/tuning/PPO')
    elif policy == "SAC":
        model = SAC('MlpPolicy', env, verbose=0, **model_params, tensorboard_log='tmp/tuning/SAC')
    elif policy == "TD3":
        model = TD3('MlpPolicy', env, verbose=0, **model_params, tensorboard_log='tmp/tuning/TD3')
    else:
        raise ValueError("Policy not yet implemented!")

    model.learn(total_timesteps=1)

    model.env.envs[0].unwrapped.agent.daf_client.send_action(EARLY_EXIT_ACTION)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)

    return mean_reward


study = optuna.create_study(direction='maximize')
study.optimize(optimize_ppo_agent, n_trials=50)

best_params = study.best_params
print(best_params.items())

if continuous_env:
    hyperparams_name = "saved_hyperparams/continuousDAF"
else:
    hyperparams_name = "saved_hyperparams/discreteDAF"

save_hyperparams(best_params, hyperparams_name)
joblib.dump(study, "tmp/study.pkl")

# Receive out message from MATLAB, close socket
env.agent.daf_client.exit_daf()
