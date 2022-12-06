import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from backend.rl_environments.discrete_environment import DiscreteEnv
from backend.rl_environments.multi_discrete_environment import MultiDiscreteEnv
from backend.rl_base_classes.mp_base_classes import MovingTargetFPATrimsAndTurns
from backend.rl_base_classes.non_mp_base_classes import MovingTargetNonMP
from backend.base_aircraft_classes.target_classes import MovingTarget
from rl_runners.mp_runners.moving_target_training import initial_state, target
from backend.utils.coordinate_transform import lla_dist
from backend.utils import circle_ang_dist, calc_bearing, wrap_ang
from backend.base_aircraft_classes.hgv_class import re


def run_network(_initial_state: np.array, _env: Monitor, _model: PPO):
    done = False
    obs = _env.reset(initial_state=_initial_state)
    while not done:
        action, _state = _model.predict(obs, deterministic=True)
        obs, _, done, __ = _env.step(action)

    _state = _env.unwrapped.agent.state
    _target_location = _env.unwrapped.agent.target_location
    final_time = _env.unwrapped.agent.time
    final_distance = lla_dist(_state[1], _state[2], _state[0],
                              _target_location[1], _target_location[2], _target_location[0])
    final_circ_distance = circle_ang_dist(_state[1], _state[2], _target_location[1], _target_location[2]) * re
    final_bear_to_tar = wrap_ang(calc_bearing(_state[1], _state[2], _target_location[1], _target_location[2])
                                 - _state[5])
    v_tar = _env.unwrapped.agent.target.target_speed
    psi_tar = _env.unwrapped.agent.target.target_heading

    return v_tar, psi_tar, final_distance, final_circ_distance, final_bear_to_tar, final_time


mp = True
generate_data = False

if mp:
    env = Monitor(DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, target)))
    model_path = 'canonical/05_16_22_moving_target/canonical_0to500v_pm180psi'
    csv_load_path = 'canonical/05_16_22_moving_target/varyingICs.csv'
    csv_save_path = 'tmp/moving_target/varyingICs.csv'
    fig_path = 'canonical/05_16_22_moving_target'
else:
    env = Monitor(MultiDiscreteEnv(MovingTargetNonMP(initial_state, target)))
    model_path = 'canonical/05_17_22_non_mp/canonical_0to500v_pm180psi'
    csv_load_path = 'canonical/05_17_22_non_mp/varyingICs.csv'
    csv_save_path = 'tmp/non_mp/varyingICs.csv'
    fig_path = 'canonical/05_17_22_non_mp'

model = PPO.load(model_path, env)

data = []
num_data = 10_000

# # Load data (comment out "Generate data" section to use)
# data = np.loadtxt(csv_path, delimiter=",")

if generate_data:
    # Generate Corner Data
    tar_1 = MovingTarget(np.concatenate((target.target_location, (target.v_range[0], 0, target.psi_range[0]))))
    tar_2 = MovingTarget(np.concatenate((target.target_location, (target.v_range[-1], 0, target.psi_range[0]))))
    tar_3 = MovingTarget(np.concatenate((target.target_location, (target.v_range[0], 0, target.psi_range[-1]))))
    tar_4 = MovingTarget(np.concatenate((target.target_location, (target.v_range[-1], 0, target.psi_range[-1]))))

    if mp:
        envs_corner = (
            Monitor(DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, tar_1))),
            Monitor(DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, tar_2))),
            Monitor(DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, tar_3))),
           Monitor(DiscreteEnv(MovingTargetFPATrimsAndTurns(initial_state, tar_4)))
       )
    else:
        envs_corner = (
            Monitor(MultiDiscreteEnv(MovingTargetNonMP(initial_state, tar_1))),
            Monitor(MultiDiscreteEnv(MovingTargetNonMP(initial_state, tar_2))),
            Monitor(MultiDiscreteEnv(MovingTargetNonMP(initial_state, tar_3))),
            Monitor(MultiDiscreteEnv(MovingTargetNonMP(initial_state, tar_4)))
        )
    print('Trials to generate corner data...')
    for _env in envs_corner:
        _model = PPO.load(model_path, _env)
        v_tar, psi_tar, final_distance, final_circ_distance, final_bear_to_tar, final_time = \
            run_network(initial_state, _env, _model)
        data.append([v_tar, psi_tar, final_distance, final_circ_distance, final_bear_to_tar, final_time])

    # Generate data (interior)
    for i in range(num_data):
        print(f'Trial {i}...')
        v_tar, psi_tar, final_distance, final_circ_distance, final_bear_to_tar, final_time = \
            run_network(initial_state, env, model)
        data.append([v_tar, psi_tar, final_distance, final_circ_distance, final_bear_to_tar, final_time])

    # Save data as CSV
    data = np.asarray(data)
    np.savetxt(csv_save_path, data, delimiter=",")
else:
    # Load data
    data = np.loadtxt(csv_load_path, delimiter=",")

# Plot results
v_tars = data[:, 0]
psi_tars = data[:, 1]
psi_tars_deg = np.rad2deg(psi_tars)
final_distances = data[:, 2]
final_distances_km = final_distances / 1000
final_circ_distances = data[:, 3]
final_circ_distances_km = final_circ_distances / 1000
final_bear_to_tars = data[:, 4]
final_bear_to_tars_deg = np.rad2deg(final_bear_to_tars)
final_times = data[:, 5]

# Final Distance Surface Plot
max_distance = np.Inf
inds = np.where(final_distances < max_distance)
_final_distances_km = final_distances_km[inds]
_v_tars = v_tars[inds]
_psi_tars_deg = psi_tars_deg[inds]

num_total = len(final_distances)
num_5 = len(final_distances[np.where(final_distances_km < 5)])
num_50 = len(final_distances[np.where(final_distances_km < 50)])
num_500 = len(final_distances[np.where(final_distances_km < 500)])
num_5000 = len(final_distances[np.where(final_distances_km < 5000)])
num_50000 = len(final_distances[np.where(final_distances_km < 50000)])

print(f'Num < 5: {num_5 / num_total}')
print(f'Num < 50: {num_50 / num_total}')
print(f'Num < 500: {num_500 / num_total}')
print(f'Num < 5,000: {num_5000 / num_total}')
print(f'Num < 50,000: {num_50000 / num_total}')

levels = (0, 5, 50, 500, 5e3, 5e4)
colors = ('b', 'c', 'g', 'y', 'r')
fig1, ax1 = plt.subplots()
# ax1.plot(_v_tars, _psi_tars_deg, 'o', markersize=2, color='grey')
cs1 = ax1.tricontourf(_v_tars, _psi_tars_deg, _final_distances_km, levels=levels, colors=colors)
ax1.set_xlabel("Target Velocity [m/s]")
ax1.set_ylabel("Target Heading [deg]")
ax1.set_title("Final Distance to Target Surf. Plot [km]")
fig1.colorbar(cs1)

# max_circ_distance = 100e3
# inds = np.where(final_circ_distances < max_circ_distance)
# _final_circ_distances_km = final_distances_km[inds]
# _v_tars = v_tars[inds]
# _psi_tars_deg = psi_tars_deg[inds]
#
# levels = np.linspace(0, max_circ_distance / 1000, 20)
# fig2, ax2 = plt.subplots()
# ax2.plot(_v_tars, _psi_tars_deg, 'o', markersize=2, color='grey')
# cs2 = ax2.tricontourf(_v_tars, _psi_tars_deg, _final_circ_distances_km, levels=levels)
# ax2.set_xlabel("Target Velocity [m/s]")
# ax2.set_ylabel("Target Heading [deg]")
# ax2.set_title("Final Circular Distance to Target Surf. Plot [km]")
# fig2.colorbar(cs2)
#
# max_bearing_to_tar = np.pi / 2
# inds = np.where(abs(final_bear_to_tars) < max_bearing_to_tar)
# _final_bear_to_tars_deg = final_bear_to_tars_deg[inds]
# _v_tars = v_tars[inds]
# _psi_tars_deg = psi_tars_deg[inds]
#
# levels = np.linspace(-90, 90, 20)
# fig3, ax3 = plt.subplots()
# ax3.plot(_v_tars, _psi_tars_deg, 'o', markersize=2, color='grey')
# cs3 = ax3.tricontourf(_v_tars, _psi_tars_deg, _final_bear_to_tars_deg, levels=levels)
# ax3.set_xlabel("Target Velocity [m/s]")
# ax3.set_ylabel("Target Heading [deg]")
# ax3.set_title("Final Bearing to Target Surf. Plot [km]")
# fig3.colorbar(cs3)

levels = np.linspace(0, 1_000, 21)
fig4, ax4 = plt.subplots()
# ax4.plot(_v_tars, _psi_tars_deg, 'o', markersize=2, color='grey')
cs4 = ax4.tricontourf(_v_tars, _psi_tars_deg, final_times)
ax4.set_xlabel("Target Velocity [m/s]")
ax4.set_ylabel("Target Heading [deg]")
ax4.set_title("Final Time of Episode [s]")
fig4.colorbar(cs4)

fig1.savefig(fname=fig_path + '/distance_surf_plot.eps', format='eps')
fig4.savefig(fname=fig_path + '/time_surf_plot.eps', format='eps')

# Display plots
plt.show()
