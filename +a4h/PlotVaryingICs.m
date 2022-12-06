data = table2struct(readtable('tmp/DAF/varying_ICs.csv'));

% Parse Data
speed_mps = [data.Var1]';
heading_rad = [data.Var2]';
dist_m = [data.Var3]';

% Convert Units for Plotting
speed   = speed_mps;
heading = rad2deg(heading_rad);
dist    = dist_m / 1000;

% Remove extreme points
distMax = 250;
inds = dist < distMax;
speed = speed(inds);
heading = heading(inds);
dist = dist(inds);

dt = delaunayTriangulation(speed, heading);
tri = dt.ConnectivityList;
xi = dt.Points(:,1);
yi = dt.Points(:,2);

F = scatteredInterpolant(speed,heading,dist);
zi = F(xi, yi);

figure()
trisurf(tri, xi, yi, zi)

grid on
zlim([0, distMax])
shading interp
colorbar

xlabel('V_{target} [m/s]')
ylabel('\psi [deg]')
zlabel('Final Distance [km]')