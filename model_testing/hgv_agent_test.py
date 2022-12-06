import numpy as np

from backend.base_aircraft_classes.hgv_class import HGV

if __name__ == '__main__':
    hgv = HGV(np.array([30000, 0, 0, 3000, -1/180*np.pi, 0]))

    # test_routine = 'direct'
    # test_routine = 'step'
    # test_routine = 'delta_step'
    # test_routine = 'linear'
    # test_routine = 'trim_mp'
    test_routine = 'trim_and_turn_mp'

    if test_routine == 'direct':
        # hgv.load_control_profile(t_ref, u_ref)
        # for _ in range(11):
        #     hgv.sim_step(5)
        #
        # hgv.plot_state_history('segmented')
        pass

    elif test_routine == 'step':
        step_length = 10
        for aoa_step in np.array([7.5, 10, 9.5, 7.5, 6, 5, 9.5]) * np.pi/180:
            hgv.constant_step(step_length, np.array([aoa_step, 0]))
            hgv.sim_step(step_length)

        hgv.plot_state_history('segmented', title='Step Primitives')

    elif test_routine == 'delta_step':
        step_length = 10
        for delta_aoa in np.array([7.5, -10, 9.5, -7.5, -6, 5, 9.5]) * np.pi / 180:
            hgv.delta_step(step_length, np.array([delta_aoa, 0]))
            hgv.sim_step(step_length)

        hgv.plot_state_history('segmented', title='Delta Step Primitives')

    elif test_routine == 'linear':
        step_length = 10
        for aoa_change in np.array([7.5, 10, 9.5, 7.5, 6, 5, 9.5]) * np.pi/180:
            hgv.linear_change(step_length, np.array([aoa_change, 0]))
            hgv.sim_step(step_length)

        hgv.plot_state_history('segmented', title='Linear Primitives')

    elif test_routine == 'trim_mp':
        step_length = 25

        for fpa_i in np.array([-0.5, -1, -2, -1.5]) * np.pi/180:
            hgv.pull_up(fpa_i)
            hgv.sim_step(step_length)
            hgv.fpa_trim(fpa_i)
            hgv.sim_step(step_length)

        hgv.plot_state_history('segmented', title='Linear Primitives')

    elif test_routine == 'trim_and_turn_mp':
        step_length = 25

        fpa_list = np.deg2rad(np.array([-1.5, -0, -0.25, -1]))
        head_list = np.deg2rad(np.array([10, -5, 5, 15]))

        for fpa_i, head_i in zip(fpa_list, head_list):
            _ti = (hgv.time + step_length)
            hgv.pull_up(fpa_i)
            hgv.sim_step(step_length)
            hgv.fpa_trim(fpa_i)
            hgv.sim_step(step_length)

            _ti = (hgv.time + step_length)
            hgv.turn(head_i)
            hgv.sim_step(step_length)
            hgv.fpa_trim(fpa_i)
            hgv.sim_step(step_length)

        hgv.plot_state_history('3d', size='ppt')
