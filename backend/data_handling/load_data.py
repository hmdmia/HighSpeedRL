import numpy as np
import scipy.interpolate


def load_reference_set(filename):
    unpacked_data = []
    raw_data = []
    with open(filename + '.npy', 'rb') as f:
        while True:
            try:
                raw_data.append(np.load(f))
            except ValueError:
                break

    for traj_data in raw_data:
        unpacked_data.append({
            'time': traj_data[:, 0],
            'alt': traj_data[:, 1],
            'down': traj_data[:, 2],
            'vel': traj_data[:, 3],
            'fpa': traj_data[:, 4],
            'aoa': traj_data[:, 5],
            'states': traj_data[:, 1:5],
            'controls': traj_data[:, 5]
        })

    return unpacked_data


def sample_data(in_t, in_data, dt):
    t_sam = np.arange(in_t[0], in_t[-1], dt)
    return t_sam, scipy.interpolate.PchipInterpolator(in_t, in_data)(t_sam)


def dicretize_control_delta(t, u, dt, du_opt):
    t_sam, u_sam = sample_data(t, u, dt)
    t_ave, u_ave = (t_sam[1:] + t_sam[:-1])/2, (u_sam[1:] + u_sam[:-1])/2
    u_0 = u_ave[0]
    du = np.concatenate((np.array([0]), np.diff(u_ave))) / dt
    du_bins = np.concatenate((np.array([-np.inf]), (du_opt[1:] + du_opt[:-1])/2, np.array([np.inf])))
    du_dis = du_opt[np.digitize(du, du_bins) - 1]

    return u_0, du, t_ave, u_ave, t_sam, u_sam, du_dis


def create_bins(u_min, u_max, step):
    center = (u_max + u_min)/2
    step_array_left = -np.flip(np.arange(center, center - u_min, step)[1:])
    step_array_right = np.arange(center, u_max, step)
    return np.concatenate((np.array([u_min]), step_array_left, step_array_right, np.array([u_max])))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = load_reference_set('emergency_descent_v_series_small')[-1]

    plt.figure(1)
    plt.subplot(221)
    plt.plot(data['down'] * 180 / np.pi, data['alt'] / 1000)
    plt.xlabel('Downrange [deg]')
    plt.ylabel('Altitude [km]')
    plt.title('Cav-H Trajectories')

    d_time, d_d_aoa = 1, 0.1

    d_aoa_opt = create_bins(-20 * np.pi / 180, 20 * np.pi / 180, d_d_aoa)
    aoa_0, d_aoa, t_a, aoa_a, t_s, aoa_s, d_aoa_dis =\
        dicretize_control_delta(data['time'], data['aoa'], d_time, d_aoa_opt)

    aoa_cur = aoa_0
    aoa_dis = []
    for d_aoa_i in d_aoa_dis:
        aoa_cur += d_aoa_i * d_time
        aoa_dis.append(aoa_cur)
    aoa_dis = np.array(aoa_dis)

    plt.subplot(222)
    plt.plot(data['time'], data['aoa'] * 180 / np.pi, label='Optimal Angle of Attack')
    plt.plot(t_s, aoa_s * 180 / np.pi, '*', label='Sample Points')
    plt.step(t_a, aoa_a * 180 / np.pi, where='mid', label='Average Angle of Attack')
    plt.step(t_s[:-1], aoa_dis * 180 / np.pi, where='post', label='Resulting Discre Angle of Attack')

    plt.xlabel('Time [s]')
    plt.ylabel('Angle of Attack [deg]')
    plt.title('Angle of Attack History')

    plt.subplot(223)
    plt.plot(data['vel'] / 1000, data['alt'] / 1000)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Altitude [km]')
    plt.title('h-v Diagram')

    plt.subplot(224)
    plt.step(t_s[:-1], d_aoa * 180 / np.pi, where='pre', label='Average Control Differences')
    plt.step(t_s[:-1], d_aoa_dis * 180 / np.pi, where='pre', label='Discretized Control Differences')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Differences [deg]')
    plt.title('Control Differences History')

    plt.suptitle('Emergency Decent Problem')

    plt.show()

