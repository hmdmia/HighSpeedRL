import numpy as np
from matplotlib import pyplot as plt, tri, ticker, cm, scale, gridspec, colors, patheffects

mp = True

csv_load_path_mp = '../canonical/05_16_22_moving_target/varyingICs.csv'
fig_path_mp = '../canonical/05_16_22_moving_target'

csv_load_path_non_mp = '../canonical/05_17_22_non_mp/varyingICs.csv'
fig_path_non_mp = '../canonical/05_17_22_non_mp'

# # Load data (comment out "Generate data" section to use)
# data = np.loadtxt(csv_path, delimiter=",")

cmap = 'viridis'
# cmap = 'viridis_r'
# cmap = 'magma_r'
# cmap = 'hot_r'
# cmap = 'inferno_r'

# Load data
data_mp = np.loadtxt(csv_load_path_mp, delimiter=",")
data_non_mp = np.loadtxt(csv_load_path_non_mp, delimiter=",")

v_tars_mp = data_mp[:, 0]
psi_tars_mp = data_mp[:, 1]
psi_tars_deg_mp = np.rad2deg(psi_tars_mp)
final_distances_mp = data_mp[:, 2]
final_distances_km_mp = final_distances_mp / 1000

v_tars_non_mp = data_non_mp[:, 0]
psi_tars_non_mp = data_non_mp[:, 1]
psi_tars_deg_non_mp = np.rad2deg(psi_tars_non_mp)
final_distances_non_mp = data_non_mp[:, 2]
final_distances_km_non_mp = final_distances_non_mp / 1000

success_threshold = 5
linthresh = success_threshold * 2
max_dist = max(max(final_distances_km_mp), max(final_distances_km_non_mp))
# levels = 50
# levels = (0, 5, 50, 500, 5e3, 5e4)
# levels = np.concatenate((np.array([0]), np.geomspace(success_threshold, 1.5*max_dist, 200)))
levels_non_mp = np.linspace(min(final_distances_km_non_mp), max(final_distances_km_non_mp), 20)
levels_mp = np.concatenate((np.linspace(0, linthresh, 1, endpoint=False),
                            np.geomspace(linthresh, max(final_distances_km_mp), 20)))

normalizer = colors.SymLogNorm(linthresh, vmin=0, vmax=max_dist)

fig1 = plt.figure(figsize=(7.5, 6))
axs = fig1.subplots(1, 2, subplot_kw=dict(polar=True))

triangulation = tri.Triangulation(psi_tars_mp/(2*np.pi), v_tars_mp/500)
mask = tri.TriAnalyzer(triangulation).get_flat_tri_mask()
triangulation.set_mask(mask)
triangles = triangulation.get_masked_triangles()
cs0 = axs[0].tricontourf(psi_tars_non_mp, v_tars_non_mp, triangles, final_distances_km_non_mp,
                         norm=normalizer, levels=levels_non_mp, cmap=cmap)

triangulation = tri.Triangulation(psi_tars_mp/(2*np.pi), v_tars_mp/500)
mask = tri.TriAnalyzer(triangulation).get_flat_tri_mask()
triangulation.set_mask(mask)
triangles = triangulation.get_masked_triangles()
cs1 = axs[1].tricontourf(psi_tars_mp, v_tars_mp, triangles, final_distances_km_mp,
                         norm=normalizer, levels=levels_mp, cmap=cmap)
# cs2 = axs[1].tricontour(psi_tars_mp, v_tars_mp, triangles, final_distances_km_mp,
#                         levels=(success_threshold,), colors='orangered')
# plt.setp(cs2.collections, path_effects=[patheffects.withTickedStroke(angle=60, length=-0.5)])

fig1.suptitle("Final Distance to Target as Function of Target Heading and Velocity")
axs[0].set_title("Non-MP Results")
axs[1].set_title("MP Results")


ticks = ticker.SymmetricalLogLocator(linthresh=linthresh, base=10)
fig1.colorbar(cm.ScalarMappable(norm=normalizer, cmap=cmap), ticks=ticks, ax=axs, orientation='horizontal',
              spacing='proportional', format='%.0f', label='Final Distance to Target [km]')

fig1.savefig(fname=fig_path_mp + '/distance_surf_plot_combined.eps', format='eps')
# fig4.savefig(fname=fig_path + '/time_surf_plot.eps', format='eps')

# Display plots
plt.show()
