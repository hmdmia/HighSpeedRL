classdef Spatial < handle
    % spatial fed into HGV Movable's movement manager.
    % Includes all fields and methods to be used by other classes.
    
    properties
        % Properties required by Movable (Cartesian values, ECEF)
        position
        velocity
        acceleration
        
        % Properties used for EOM
        vehicleType
        world
        
        % Controller for spatial
        controller
    end
    
    methods
        % varargin{1} = control
        function obj = Spatial(position, velocity, acceleration, vehicleType, world, varargin)
            %SPATIAL Construct an instance of this class
            %   Detailed explanation goes here
            obj.position = position;
            obj.velocity = velocity;
            obj.acceleration = acceleration;
            obj.vehicleType = vehicleType;
            obj.world = world;
            
            % Sets the optional controller
            if ~isempty(varargin)
                obj.controller = varargin{1};
            end % if
        end % CONSTRUCTOR
        
        % Used to update the spatial when integrating the movement
        % This must take place during integration so acceleration may
        % be recorded
        function updateSpatial(obj, stateVector, vDot, gamDot, psiDot)
            lla = [rad2deg(stateVector(2)), rad2deg(stateVector(3)), stateVector(1)];
            vgampsi = [stateVector(4), rad2deg(stateVector(5)), rad2deg(stateVector(6))];
            vgampsiDot = [vDot, rad2deg(gamDot), rad2deg(psiDot)];
            [obj.position, obj.velocity, obj.acceleration] = obj.world.getrvaECEF(lla, vgampsi, vgampsiDot);
        end % updateSpatial
        
        % outputs a new spatial with the same properties as the old spatial
        % to prevent mishap when passing by address
        function spatial = newSpatial(obj)
            spatial = a4h.util.Spatial(obj.position, obj.velocity, obj.acceleration, obj.vehicleType, obj.world, obj.controller);
        end % newSpatial
        
        %% getters for transformed state information
        function out = stateVector(obj)
            stateStruct = obj.stateStruct();
            out = [stateStruct.altitude; ...
                stateStruct.theta; ...
                stateStruct.phi; ...
                stateStruct.velocity; ...
                stateStruct.gamma; ...
                stateStruct.psi];
        end % stateVector
        
        function out = stateStruct(obj)
            [lla, vgampsi] = obj.world.ecef2spherical(obj.position, obj.velocity);

            out.altitude = lla(3);
            out.theta = deg2rad(lla(1));
            out.phi = deg2rad(lla(2));
            out.velocity = vgampsi(1);
            out.gamma = deg2rad(vgampsi(2));
            out.psi = deg2rad(vgampsi(3));
        end
        
        % outputs struct with position in spherical coordinates
        % (radius [m], theta [rad], phi [rad])
        % NOTE: cart2sph gives longitude (azimuth), then latitude
        % (elevation)
        function out = spherical(obj) % vector
            stateStruct = obj.stateStruct();
            out.radius = stateStruct.altitude + obj.world.getRadius();
            out.theta  = stateStruct.theta;
            out.phi    = stateStruct.phi;
        end % spherical
        
        % outputs scalar norm of velocity vector [m/s]
        function out = velocityNorm(obj)
            out = obj.stateStruct().velocity;
        end % velocityNorm
        
        % outputs scalar flight path angle (gamma) [rad]
        function out = gamma(obj)
            out = obj.stateStruct().gamma;
        end % gamma
        
        % outputs scalar heading angle (psi) [rad]
        function out = psi(obj)
            out = obj.stateStruct().psi;
        end % psi
        
        function out = str(obj)
            x = obj.stateVector();
            out = sprintf('[%.2f, %.2f%c, %.2f%c, %.2f, %.2f%c, %.2f%c]', ...
                x(1), rad2deg(x(2)), char(176), rad2deg(x(3)), char(176), ...
                x(4), rad2deg(x(5)), char(176), rad2deg(x(6)), char(176));
        end % str
    end % methods
    
    methods (Static)
        function spatial = stateVector2spatial(stateVector, vehicleType, world, varargin)
            [position, velocity] = a4h.util.Spatial.stateVector2ecef(stateVector, world);
            acceleration = [];
            spatial = a4h.util.Spatial(position, velocity, acceleration, vehicleType, world, varargin{:});
        end % stateVector2spatial

        function [position, velocity] = stateVector2ecef(stateVector, world)
            lla = [rad2deg(stateVector(2)), rad2deg(stateVector(3)), stateVector(1)];
            vgampsi  = [stateVector(4), rad2deg(stateVector(5)), rad2deg(stateVector(6))];

            [position, velocity] = world.getrvECEF(lla, vgampsi);
        end % stateVector2ecef
        
        % Can take position & velcity as vectors or nx3 matrices
        % For vector pos./vel., world must be a scalar of a world object
        % For vector pos./vel., stateVector is 6x1 vector
        % For matrix pos./vel., world can be a scalar or n-vector of world
        % objects
        % For matrix pos./vel., stateVector is nx6 matrix
        function stateVector = ecef2stateVector(position, velocity, world)
            if isvector(position)
                stateVector = a4h.util.Spatial(position ,velocity, [], ...
                    [], world).stateVector();
            else
                vecLength = length(position(:,1));
                stateVector = zeros(vecLength, 6);
                
                if length(world) == vecLength
                    for i = 1:vecLength
                        stateVector(i,:) = ...
                          a4h.util.Spatial(position(i,:), velocity(i,:), ...
                              [], [], world(i)).stateVector()';
                    end % for
                else
                    for i = 1:vecLength
                        stateVector(i,:) = ...
                          a4h.util.Spatial(position(i,:), velocity(i,:), ...
                              [], [], world(1)).stateVector()';
                    end % for
                end % if

            end % if
        end % ecef2stateVector
        
        % Converts 3-vec position to scalars theta & phi (theta in I, IV)
        function [theta, phi] = position2thetaPhi(position)
            theta = atan(position(2)/position(1));
            phi = acos(dot(position, [cos(theta) sin(theta) 0])/norm(position))*sign(position(3));
        end % position2thetaPhi
        
        function bearing = calcBearing(theta1, phi1, theta2, phi2)
            dlong = phi2 - phi1;
            x = cos(theta2) * sin(dlong);
            y = cos(theta1) * sin(theta2) - sin(theta1) * cos(theta2) * cos(dlong);
            bearing = atan2(x, y);
        end % calcBearing
        
        function out = calcCircleAngDist(theta1, phi1, theta2, phi2)
            % Haversine of angle, i.e. great-circle distance
            hav = @(angle) (1 - cos(angle))/2;
            
            % Inverse haversine of angle
            ahav = @(angle) 2*asin(sqrt(angle));
            
            hav_theta = hav(theta2 - theta1) ...
                + cos(theta1)*cos(theta2)*hav(phi2 - phi1);
            
            out = ahav(hav_theta);
        end % calcCircleAngDist
        
        % puts input in range [-180, +180) deg
        function out = wrapAngle(ang)
            out = mod((ang + pi), (2 * pi)) - pi;
        end % wrapAngle
    end % static methods
    
    %%% TEST METHODS %%%
    methods (Static)
        function testWrapAngle
            values = 1000;
            angles = deg2rad(linspace(-720, 720, values));
            wrappedAngles = a4h.util.Spatial.wrapAngle(angles);

            figure()
            colors = get(gca,'colororder');
            plot(rad2deg(angles), rad2deg(wrappedAngles), 'Color', colors(1,:))
            hold on
            plot(rad2deg(angles), ones(values)*180, '--', 'Color', colors(2,:))
            plot(rad2deg(angles), ones(values)*-180, '--', 'Color', colors(2,:))
            hold off
            xlabel('Angle [deg]')
            ylabel('Wrapped Angled [deg]')
        end % testWrapAngle

        function testSpatialConversion
            % This method tests whether the conversion from
            % Cartesian (ECEF) to spherical (lla, [v, gamma, psi])
            % properly takes place.
            world = a4h.util.AtmosphericWorld();

            cases = 10;
            stateVector = [1e5*rand(cases,1), pi*rand(cases,1) - pi/2, 2*pi*rand(cases,1) - pi, 1e4*rand(cases,1)+1, pi*rand(cases,1) - pi/2, 2*pi*rand(cases,1) - pi]';
            
            for i = 1:cases
                a4h.util.Spatial.stateVectorError(world, stateVector(:,i), i)
            end

            thirdCases = 5;
            position = [2000*rand(thirdCases,1) + world.getRadius(), 2000*rand(thirdCases,1), 2000*rand(thirdCases,1); ...
                2000*rand(thirdCases,1), 2000*rand(thirdCases,1) + world.getRadius(), 2000*rand(thirdCases,1); ...
                2000*rand(thirdCases,1), 2000*rand(thirdCases,1), 2000*rand(thirdCases,1) + world.getRadius()];
            velocity = [200*rand(thirdCases*3,1), 200*rand(thirdCases*3,1), 200*rand(thirdCases*3,1)];

            for i = 1:thirdCases*3
                a4h.util.Spatial.ecefError(world, position(i,:), velocity(i,:), i+cases);
            end % for

        end % testSpatialConversion

        % Helper methods
        function stateVectorError(world, stateVector, i)
            [position, velocity] = a4h.util.Spatial.stateVector2ecef(...
                stateVector, world);
            newStateVector = a4h.util.Spatial.ecef2stateVector(...
                position, velocity, world);
            error = norm(stateVector - newStateVector);

            if error > 1e-3
                fprintf('\n   ---TEST CASE %d FAILED---\n      (error = %.2f)\n', i, error)
                fprintf(['stateVec: ',mat2str(stateVector),'\n -->    ',mat2str(newStateVector),'\n'])
            else
                fprintf('\n   ---TEST CASE %d PASSED---\n', i)
            end % if
        end % stateVectorError

        function ecefError(world, position, velocity, i)
            stateVector = a4h.util.Spatial.ecef2stateVector(...
                position, velocity, world);
            [newPosition, newVelocity] = a4h.util.Spatial.stateVector2ecef(...
                stateVector, world);

            error = norm([position, velocity] - [newPosition, newVelocity]);

            if error > 1e-3
                fprintf('\n   ---TEST CASE %d FAILED---\n      (error = %.2f)\n', i, error)
                fprintf(['Position: ',mat2str(position),' --> ',mat2str(newPosition),'\n'])
                fprintf(['Velocity: ',mat2str(velocity),' --> ',mat2str(newVelocity),'\n'])
            else
                fprintf('\n   ---TEST CASE %d PASSED---\n', i)
            end % if
        end % ecefError
    end % static tester methods
end % CLASS DEF

