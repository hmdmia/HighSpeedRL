import numpy as np

from stable_baselines3 import PPO

from backend.rl_environments.multi_discrete_environment import MultiDiscreteEnv
from backend.rl_base_classes.non_mp_base_classes import MovingTargetNonMP
from backend.utils.coordinate_transform import lla_dist
from backend.utils import circle_ang_dist, calc_bearing, wrap_ang
from backend.base_aircraft_classes.hgv_class import re
from rl_runners.non_mp_runners.non_mp_movingtarget import initial_state, target


def run_network(_initial_state, _env, _model):
    done = False
    obs = _env.reset(initial_state=_initial_state)
    while not done:
        action, _state = _model.predict(obs, deterministic=True)
        obs, _, done, __ = _env.step(action)

    _state = _env.unwrapped.agent.state
    _target_location = _env.unwrapped.agent.target_location
    final_distance = lla_dist(_state[1], _state[2], _state[0],
                              _target_location[1], _target_location[2], _target_location[0])
    final_circ_distance = circle_ang_dist(_state[1], _state[2], _target_location[1], _target_location[2]) * re
    final_bear_to_tar = wrap_ang(calc_bearing(_state[1], _state[2], _target_location[1], _target_location[2])
                                 - _state[5])

    print(f'HGV final location: h = {_state[0]/1000} [km],'
          f'theta = {np.rad2deg(_state[1])} [deg], phi = {np.rad2deg(_state[2])}')
    print(f'Target final location: h = {_target_location[0]/1000} [km],'
          f' theta = {np.rad2deg(_target_location[1])} [deg], phi = {np.rad2deg(_target_location[2])}')
    print(f'Final distance: {final_distance/1000} [km]')
    print(f'Final circ. distance: {final_circ_distance/1000} [km]')
    print(f'Final bear. to tar.: {np.rad2deg(final_bear_to_tar)} [deg]\n')

    _env.agent.plot_state_history(style='3d')


if __name__ == '__main__':
    env = MultiDiscreteEnv(MovingTargetNonMP(initial_state, target))
    model = PPO.load('canonical/05_17_22_non_mp/canonical_0to500v_pm180psi', env)

    run_network(initial_state, env, model)
