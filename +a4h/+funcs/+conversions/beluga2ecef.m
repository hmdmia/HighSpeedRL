function [ecefPosition,ecefVelocity] = beluga2ecef(altitude,downrange,velocityNorm,gamma,psi,reference)
    % 2-D position and velocity from 2D beluga input

    % gamma and psi must be in radians

    [startTheta,startPhi,startRadius] = cart2sph(reference(1),reference(2),reference(3));

    world = base.world.Earth;
    world.setModel('elliptical');
    earthRadius = world.getRadius;

    % Ideally, startEarthRadius will be identical to earthRadius.
    % Currently, this variable is calculated for debugging and

    % testing

    radius = altitude + earthRadius;

    theta = downrange.*cos(psi) + startTheta;

    % Adds startTheta to the downrange values. cos(psi) makes sure
    % that the downrange is the correct sign relative to the
    % startTheta (positive for eastward, negative for westward)

    [x,y,z] = sph2cart(theta,0,radius);

    ecefPosition = [x,y,z];

    enuVel = [cos(gamma).*velocityNorm.*cos(psi),zeros(length(velocityNorm),1),sin(gamma).*velocityNorm];

    ecefVelocity = a4h.funcs.conversions.enuVec2ecefVec(enuVel,ecefPosition);
end