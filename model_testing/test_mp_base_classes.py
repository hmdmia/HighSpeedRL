import numpy as np
from random import randrange

from backend.rl_base_classes.mp_base_classes import FPATrimsAndTurns, MovingTargetFPATrimsAndTurns
from backend.rl_environments import DiscreteEnv
from backend.base_aircraft_classes.target_classes import MovingTarget


def run_actions(_initial_state, _env, _actions, plot=False):
    done = False
    i = 0
    max_i = len(_actions)
    _ = _env.reset(initial_state=_initial_state)
    while not done and i < max_i:
        action = _actions[i]
        _, __, done, ___ = _env.step(action)
        i += 1

    if plot:
        _env.agent.plot_state_history(style='3d')


agent = 'MovingTargetFPATrimsAndTurns'
initial_state = np.array((19000, np.deg2rad(35.1), np.deg2rad(-179.6), 6000, 0, -np.pi/2))
target_state = np.array((00000, np.deg2rad(40.4), np.deg2rad(-86.9), 100, 0, 0))

if agent == 'FPATrimsAndTurns':
    env = DiscreteEnv(FPATrimsAndTurns(initial_state, target_state[:3]))
elif agent == 'MovingTargetFPATrimsAndTurns':
    target = MovingTarget(target_state)
    env = DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, target))

for trial in range(500):
    actions = [randrange(0, env.agent.n_actions) for __ in range(100)]
    run_actions(initial_state, env, actions, plot=False)
    print(f'Finished trial {trial}\n')
