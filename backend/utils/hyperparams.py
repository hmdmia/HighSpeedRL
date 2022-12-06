import ast
from torch import nn


def save_hyperparams(params, filename):
    f = open(filename + '.txt', 'w')
    f.write(str(params))
    f.close()

    return


def _load_hyperparams(file):
    with open(file + '.txt', "r") as f:
        contents = f.read()
    hyperparams = ast.literal_eval(contents)

    hyperparams["learning_rate"] = hyperparams["lr"]
    del hyperparams["lr"]

    if "step_multiplier" in hyperparams:
        hyperparams["n_steps"] = hyperparams["step_multiplier"] * hyperparams["batch_size"]
        del hyperparams["step_multiplier"]

    _activation_fn = activation_options(hyperparams["activation_fn"])
    del hyperparams["activation_fn"]

    hyperparams["policy_kwargs"] = dict(net_arch=None, activation_fn=_activation_fn)
    return hyperparams


def _load_off_policy_hyperparams(hyperparams):
    if "arch_size" in hyperparams:
        _net_arch = off_policy_net_arch_size(hyperparams["arch_size"])
        del hyperparams["arch_size"]
    elif "net_arch" in hyperparams:
        _net_arch = off_policy_net_arch_options(hyperparams["net_arch"])
        del hyperparams["net_arch"]
    else:
        _net_arch = off_policy_net_arch_size(256)

    return hyperparams


def load_sac_hyperparams(file):
    hyperparams = _load_hyperparams(file)
    hyperparams = _load_off_policy_hyperparams(hyperparams)

    hyperparams["train_freq"] = train_freq_tuple(hyperparams["train_freq_episode"], type="episode")
    del hyperparams["train_freq_episode"]

    return hyperparams


def load_dqn_hyperparams(file):
    hyperparams = _load_hyperparams(file)
    hyperparams = _load_off_policy_hyperparams(hyperparams)
    del hyperparams["subsample_steps"]

    return hyperparams


def load_ppo_hyperparams(file):
    hyperparams = _load_hyperparams(file)

    if "arch_size" in hyperparams:
        _net_arch = on_policy_net_arch_size(hyperparams["arch_size"])
        del hyperparams["arch_size"]
    elif "net_arch" in hyperparams:
        _net_arch = on_policy_net_arch_options(hyperparams["net_arch"])
        del hyperparams["net_arch"]
    else:
        _net_arch = on_policy_net_arch_size(256)

    if "ortho_init" in hyperparams:
        _ortho_init = hyperparams["ortho_init"]
        del hyperparams["ortho_init"]
    else:
        _ortho_init = False

    hyperparams["policy_kwargs"]["net_arch"] = _net_arch
    hyperparams["policy_kwargs"]["ortho_init"] = _ortho_init

    return hyperparams


def off_policy_net_arch_options(net_arch):
    _net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    return _net_arch


def activation_options(activation_fn):
    _activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return _activation_fn


def on_policy_net_arch_options(net_arch):
    _net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    return _net_arch


def on_policy_net_arch_size(arch_size):
    _net_arch = [dict(pi=[arch_size, arch_size], vf=[arch_size, arch_size])]
    return _net_arch


def off_policy_net_arch_size(arch_size):
    _net_arch = [arch_size, arch_size]
    return _net_arch


def train_freq_tuple(train_freq_int, train_freq_type="episode"):
    assert(train_freq_type == "episode" or train_freq_type == "step")
    _train_freq = (train_freq_int, train_freq_type)
    return _train_freq
