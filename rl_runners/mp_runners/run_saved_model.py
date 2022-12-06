import numpy as np

from stable_baselines3 import PPO

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_base_classes.mp_base_classes import FPATrimsAndTurns, MovingTargetFPATrimsAndTurns
from backend.utils.coordinate_transform import lla_dist
from backend.utils import circle_ang_dist, calc_bearing, wrap_ang
from backend.base_aircraft_classes.hgv_class import re


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

    # _env.agent.plot_state_history(style='3d')
    return _env.agent


if __name__ == '__main__':
    moving_target = True

    if moving_target:
        from rl_runners.mp_runners.moving_target_training import initial_state, target_location  # , target
        from backend.base_aircraft_classes.target_classes import MovingTarget

        target = MovingTarget(np.concatenate((target_location, (450, 0, np.deg2rad(0)))))
        env = DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, target))
        model = PPO.load('canonical/05_16_22_moving_target/canonical_0to500v_pm180psi', env)
    else:
        from rl_runners.mp_runners.ppo_mp_fpa_trim_and_turn import initial_state, target_location
        env = DiscreteEnv(FPATrimsAndTurns(initial_state, target_location))
        model = PPO.load('tmp/saved_models/ppo_mp_fpa_trim_and_turns', env)

    agent_mp = run_network(initial_state, env, model)

    from backend.rl_environments.multi_discrete_environment import MultiDiscreteEnv
    from backend.rl_base_classes.non_mp_base_classes import MovingTargetNonMP
    env = MultiDiscreteEnv(MovingTargetNonMP(initial_state, target))
    model = PPO.load('canonical/05_17_22_non_mp/canonical_0to500v_pm180psi', env)

    agent_non_mp = run_network(initial_state, env, model)

    from matplotlib import pyplot as plt
    # figsize = (7.5, 4)
    figsize = (7, 3)
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    t_mp = np.concatenate(agent_mp.time_history)
    y_mp = np.hstack(agent_mp.state_history)
    d_mp = np.concatenate(agent_mp.d_history)
    b_mp = np.concatenate(agent_mp.bear_history)

    t_non = np.concatenate(agent_non_mp.time_history)
    y_non = np.hstack(agent_non_mp.state_history)
    d_non = np.concatenate(agent_non_mp.d_history)
    b_non = np.concatenate(agent_non_mp.bear_history)


    ax1.plot(t_mp, d_mp / 1000, color='#ff7f0e')
    ax1.plot(t_non, d_non / 1000, color='#1f77b4')

    ax2.plot(t_non, b_non * 180 / np.pi, label='non-MP')
    ax2.plot(t_mp, b_mp * 180 / np.pi, label='MP')

    ax3.plot(t_non, y_non[4, :] * 180 / np.pi, label='non-MP')
    ax3.plot(t_mp, y_mp[4, :] * 180 / np.pi, label='MP')

    ax4.plot(t_non, y_non[5, :] * 180 / np.pi)
    ax4.plot(t_mp, y_mp[5, :] * 180 / np.pi)


    # for t_mp, y_mp, d_mp, b_mp, t_non, y_non, d_non, b_non in zip(agent_mp.time_history,
    #                                                               agent_mp.state_history,
    #                                                               agent_mp.d_history,
    #                                                               agent_mp.bear_history,
    #                                                               agent_non_mp.time_history,
    #                                                               agent_non_mp.state_history,
    #                                                               agent_non_mp.d_history,
    #                                                               agent_non_mp.bear_history):
    #     ax1.plot(t_mp, d_mp / 1000)
    #     ax1.plot(t_non, d_non / 1000)
    #
    #     ax2.plot(t_mp, b_mp * 180 / np.pi)
    #     ax2.plot(t_non, b_non * 180 / np.pi)
    #
    #     ax3.plot(t_mp, y_mp[4, :] * 180 / np.pi)
    #     ax3.plot(t_non, y_non[4, :] * 180 / np.pi)
    #
    #     ax4.plot(t_mp, y_mp[5, :] * 180 / np.pi)
    #     ax4.plot(t_non, y_non[5, :] * 180 / np.pi)

    ax1.set_xlabel('Time [s]')
    ax2.set_xlabel('Time [s]')
    ax3.set_xlabel('Time [s]')
    ax4.set_xlabel('Time [s]')

    ax1.set_ylabel('d [km]')
    ax2.set_ylabel('β - ψ [deg]')
    ax3.set_ylabel('γ [deg]')
    ax4.set_ylabel('ψ [deg]')

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    ax3.legend(loc='center')

    plt.tight_layout()
    fig.savefig(fname='tmp/trajectory_250v0psi.eps', format='eps', bbox_inches='tight')

    plt.show()
