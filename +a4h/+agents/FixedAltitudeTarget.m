classdef FixedAltitudeTarget < base.agents.Movable & base.agents.Detectable & base.agents.physical.Destroyable & base.agents.physical.Worldly
    % Object that moves with newton updates but is fixed to the earth.
    properties
        altitude % Constant altitude obj is held to
        agentUpdateRate = 1
    end

    properties (Hidden,Access=protected)
        maxTime = 1 % Length of integration steps
    end

    methods
        function obj = FixedAltitudeTarget()
            world = base.world.Earth();
            world.setModel(world.SPHERICAL);
            obj.setWorld(world);
        end % Altitude Movable

        function init(obj)
            lla = obj.world.convert_ecef2lla(obj.spatial.position); % Pulling directly from spatial because altitude not yet set
            obj.altitude = max(lla(3), 1); % For analysis, target should > 0 m
            obj.movementManager = base.funcs.movement.NewtonMotion;
            obj.setHeading(obj.getVelocity());
            
            % For radar detectability
            targetWidth = 1; %m
            targetLength= 3; %m
            obj.setDimensions(targetLength,targetWidth);
            
            obj.scheduleAtTime(0);
        end % init

        function runAtTime(obj,time)
            obj.getPosition();
            
            if obj.isDestroyed
                obj.instance.RemoveCallee(obj);
                fprintf('Removing %s from call list \n',obj.commonName)
                obj.instance.endSim();
            end % if
            obj.scheduleAtTime(time+obj.agentUpdateRate);
        end % runAtTime
        
        function position = getPosition(obj)
            position = getPosition@base.agents.Movable(obj);
            obj.setHeading(obj.getVelocity());
        end % getPosition
        
        function fixToAltitude(obj)
            % We are going to snoop the spatial position and
            % adjust it on the fly
            position = obj.spatial.position; % This is in ECEF
            lla = obj.world.convert_ecef2lla(position);
            obj.spatial.position = obj.world.convert_lla2ecef([lla(1:2), obj.altitude]);

            % Now we will also fix the acceleration and velocity to be tangent at the new position.
            velocity = obj.spatial.velocity;
            speed = norm(velocity);

            % Subtract the component of velocity in the "Position" direction
            % because this is vertical w.r.t. the earth.
            newVelocityHat = velocity - dot(velocity,position)*position /(norm(position)^2);
            if speed <= 0
                newVelocityHat = ones(size(velocity));
            end % if
            newVelocityHat = newVelocityHat/norm(newVelocityHat);
            
            newVelocity = newVelocityHat*speed;
            obj.spatial.velocity = newVelocity;

            acceleration = obj.spatial.acceleration;
            accelerationMag = norm(acceleration);

            % Subtract the component of acceleration in the "Position" direction
            % because this is vertical w.r.t. the earth.
            newAccelerationHat = acceleration - dot(acceleration,position)*position /(norm(position)^2);
            newAccelerationHat = newAccelerationHat/norm(newAccelerationHat);
            if accelerationMag > 0
                newAccelerationHat = newAccelerationHat/norm(newAccelerationHat);
            else
                newAccelerationHat = ones(size(acceleration));
            end % if
            newAcceleration = newAccelerationHat*accelerationMag;
            obj.spatial.acceleration = newAcceleration;
        end % fixToAltitude

        function updateMovement(obj,targetTime)

            % Updates movement using the movement manager
            timeOffset = targetTime - obj.movementLastUpdateTime;

            if timeOffset > obj.maxTime
                % Then let's make timeOffset a vector and loop over it
                tVect = 0:obj.maxTime:timeOffset;
                if tVect(end) ~= timeOffset
                    tVect(end+1)=timeOffset;
                end % if

                targetTime = tVect+obj.movementLastUpdateTime;
            end % if

            for dt = targetTime
                % Fix the movable to the earth, then move it.
                obj.fixToAltitude();
                updateMovement@base.agents.Movable(obj,dt);
            end % for
        end % updateMovement

        function updateIrradiance(obj)
            % Update irradiance using the setIrradiance function
            obj.setIrradiance(1e20); % was 1e9
        end
    end % object methods

    methods (Static, Access = {?base.test.UniversalTester})
        function tests = test()
            % Run all tests
            tests = {};
        end
    end % Universal Tester methods

end % CLASS DEFINITION