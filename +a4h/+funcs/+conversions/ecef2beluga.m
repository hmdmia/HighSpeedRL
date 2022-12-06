function [altitude,downrange,velocityNorm,gamma,psi] = ecef2beluga(ecefPosition,ecefVelocity,reference)

            [theta,phi,radius] = cart2sph(ecefPosition(:,1),ecefPosition(:,2),ecefPosition(:,3));

            [refTheta,refPhi,refRadius] = cart2sph(reference(1),reference(2),reference(3));

            enuVelocity = zeros(length(ecefVelocity(:,1)),3);
            gamma = zeros(length(ecefVelocity(:,1)),1);
            psi = zeros(length(ecefVelocity(:,1)),1);
            east = zeros(length(ecefVelocity(:,1)),1);
            north = zeros(length(ecefVelocity(:,1)),1);
            up = zeros(length(ecefVelocity(:,1)),1);

            velocityNorm = sqrt(sum(ecefVelocity.^2, 2));

            for i = 1:length(ecefVelocity(:,1))
               enuVelocity(i,:) = a4h.funcs.conversions.ecefVec2enuVec(ecefPosition(i,:),ecefVelocity(i,:));

               east(i) = enuVelocity(i,1);
               north(i) = enuVelocity(i,2);
               up(i) = enuVelocity(i,3);

               gamma(i) = atan(up(i)/sqrt(east(i)^2 + north(i)^2)); % 2-quadrant inverse tangent

               psi(i) = atan2(north(i),east(i)); % 4-quadrant inverse tangent
               % For 2D case, psi should be either 0 or pi
            end

            downrange = abs(theta - refTheta);

            world = base.world.Earth;
            world.setModel('elliptical')
            earthRadius = world.getRadius;

            altitude = radius - earthRadius;

        end