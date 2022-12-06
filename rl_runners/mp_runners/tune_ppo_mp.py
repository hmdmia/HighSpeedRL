import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.mp_base_classes import FPATrimsAndTurns
from rl_runners.mp_runners.ppo_mp_fpa_trim_and_turn import initial_state, target_location
from backend.utils.hyperparams import save_hyperparams
from backend.utils.hyperparams import activation_options
from backend.utils.hyperparams import on_policy_net_arch_size


def ppo_param_set(trial):
    batch_size = trial.suggest_int(name="batch_size", low=32, high=1024)
    step_multiplier = trial.suggest_int(name="step_multiplier", low=1, high=5)
    gamma = trial.suggest_float(name="gamma", low=0.9, high=0.99999)
    learning_rate = trial.suggest_float(name="lr", low=1e-6, high=1, log=True)
    arch_size = trial.suggest_int(name="arch_size", low=64, high=1024)
    ent_coef = trial.suggest_float(name="ent_coef", low=1e-5, high=0.1, log=True)
    clip_range = trial.suggest_float(name="clip_range", low=0, high=0.3)
    n_epochs = trial.suggest_int(name="n_epochs", low=1, high=30)
    gae_lambda = trial.suggest_float(name="gae_lambda", low=0.8, high=1)
    max_grad_norm = trial.suggest_float(name="max_grad_norm", low=0.1, high=5)
    vf_coef = trial.suggest_float(name="vf_coef", low=0, high=1)
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


def optimize_ppo_agent(trial):
    model_params = ppo_param_set(trial)
    env = DiscreteEnv(FPATrimsAndTurns(initial_state, target_location))
    model = PPO('MlpPolicy', env, verbose=0, **model_params, tensorboard_log="tmp/tuning/mp/PPO")
    model.learn(total_timesteps=int(1e6))
    model.save("tmp/tuning/mp/saved_models/PPO" + str(trial.number + 1))
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)

    return mean_reward


study = optuna.create_study(direction='maximize')
study.optimize(optimize_ppo_agent, n_trials=10)

best_params = study.best_params
print(best_params.items())
save_hyperparams(best_params, "saved_hyperparams/ppo_random_mp")
