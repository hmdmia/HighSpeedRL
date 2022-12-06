classdef HGVmp < a4h.agents.concrete.BaseHGV

    properties (Constant)
        MINFPA = deg2rad(-45)
        MAXFPA = deg2rad(10)
        MINALT = 20000 % [m]
        MAXALT = 75000 % [m]
        MINVEL = 1500 % [m/s]
        MAX_BEARING = pi/2 % [rad]
        FARUPDATERATE = 25; % [s]
        CLOSEUPDATERATE = 0.5 % [s]
        CLOSE_SCALE_FACTOR = 2
    end
    
    properties (Hidden)
        isContinuousMP = false % True -> use Continuous MPs
        mpList % input taken in runner, but assigned to moveMan in init.
        captureDistance = 5000 % circular distance tolerance for
                                 % pred to capture prey [m]
        altitudeTolerance = 1000 % altitude tolerance for
                                  % pred to capture prey [m]
        closeUpdateRate = 0.5 % [s]
    end % properties
    
    methods % Public methods
        % DESCRIPTION:
        %   constructor for hgvmp object
        % OUTPUT:
        %   obj (an HGVmp object)
        % INPUT:
        %   state   (struct) position, velocity, acceleration vectors
        function obj = HGVmp()
            obj@a4h.agents.concrete.BaseHGV();
            obj.movementUpdateRate = obj.FARUPDATERATE; % sec
        end % HGVmp (constructor)
        
        % Overrides BaseHGV to incorporate MP control
        function setInitialState(obj, initialTime, initialState)
            setInitialState@a4h.agents.concrete.BaseHGV(obj, initialTime, initialState);
            
            obj.spatial = a4h.util.SpatialMP(initialState.position, ...
                initialState.velocity, initialState.acceleration, ...
                a4h.util.HGVType(), obj.world);
        end
        
        % Obtains data to be sent to RL for observation as well as
        % any state info stored in HGV agent.
        % "time" is the time to which we are scheduled to integrate,
        % whereas "currentTime" is the time to which the agent has
        % already been integrated.
        function [data4rl,currentTime] = getStateInfo(obj,time)
            obj.disp_DEBUG(sprintf('State (t = %d, MP = %d):\n', ...
                time, obj.spatial.getPrimitive()));
            obj.disp_DEBUG(sprintf('%s\n', obj.spatial.str()));
            obj.disp_DEBUG(sprintf('Control: [%.2f, %.2f]\n', ...
                rad2deg(obj.spatial.getControl())));
            currentTime = obj.movementLastUpdateTime();
            predPos = obj.spatial.position;
            
            % Prey values
            observables = obj.observableObjectManager.getObservables(time);
            
            % Prey position/velocity is the average of observables'
            sumPos = [0,0,0];
            sumVel = [0,0,0];
            for i = 1:numel(observables)
                % Seems a little overkill if the observable manager
                % updates the positions... but sure
                % TODO ^ is there a better way to implement this? -wlevin
                % 06/30/2021
                sumPos = observables{i}.getPosition() + sumPos;
                sumVel = observables{i}.getVelocity() + sumVel;
                posStr = sprintf('\t%s position: %s time %f',observables{i}.commonName,mat2str(predPos),time);
                obj.disp_DEBUG(posStr);
            end

            preyPos = sumPos / i;
            preyVel = sumVel / i;
            
            % Logical values
            success = false;
            endSim = false;
            done = false;
           
            stateVector = obj.spatial.stateVector();
            control = obj.spatial.getControl();
            
            targetStateVector = obj.spatial.ecef2stateVector(preyPos, preyVel, obj.spatial.world);
            
            % Distance along earth's surface to target
            surface_dist = obj.spatial.calcCircleAngDist(stateVector(2), ...
                stateVector(3), targetStateVector(2), targetStateVector(3)) ...
                * obj.world.getRadius();
            
            % Altitude difference
            relativeAltitude = abs(stateVector(1) - targetStateVector(1));
            
            estimatedFarDistance = stateVector(4) * obj.FARUPDATERATE;
            obj.closeUpdateRate = max(obj.CLOSEUPDATERATE, estimatedFarDistance / (stateVector(4) * obj.CLOSE_SCALE_FACTOR));  % Set close update rate to time taken to reach target
            
            % Update 
            dist = norm(obj.spatial.position - preyPos);
            close = dist < estimatedFarDistance * obj.CLOSE_SCALE_FACTOR;

            % bearing(theta, phi, thetaPrey, phiPrey)
            bearing = obj.spatial.calcBearing(stateVector(2), ...
                stateVector(3), targetStateVector(2), targetStateVector(3));
            
            % wrap angle(bearing - psi)
            relativeBearing = obj.spatial.wrapAngle(bearing - stateVector(6));

            % relative bearing from target to HGV
            tarBearing = obj.spatial.calcBearing(targetStateVector(2), ...
                targetStateVector(3), stateVector(2), stateVector(3));
            relativeTargetBearing = obj.spatial.wrapAngle(tarBearing - targetStateVector(6));
            
            if surface_dist < obj.captureDistance && ...
                    relativeAltitude < obj.altitudeTolerance
                success = true;
                done = true;
                obj.disp_INFO(sprintf('Reached capture distance. Distance to target: %d m\n', ...
                    round(dist)));
            elseif time >= obj.simSecs
                obj.disp_INFO(sprintf('Simulation timed out. Distance to target: %d m\n', ...
                    round(dist)));
                endSim = true;
                done = true;
            elseif stateVector(1) < obj.MINALT % check h
                obj.disp_INFO(sprintf('Altitude < %d km: h = %d m. Distance to target: %d m\n', ...
                    obj.MINALT/1000, stateVector(1), round(dist)));
                done = true;
            elseif stateVector(1) > obj.MAXALT % check h
                obj.disp_INFO(sprintf('Altitude > %d km: h = %d. Distance to target: %d m\n', ...
                    obj.MAXALT/1000, stateVector(1), round(dist)));
                done = true;
            elseif (stateVector(5) < obj.MINFPA || stateVector(5) > obj.MAXFPA) % check FPA
                done = true;
                obj.disp_INFO(sprintf('FPA out of bounds. Distance to target: %d m, FPA: %.2f deg\n', ...
                    round(dist), rad2deg(stateVector(5))));
            elseif stateVector(4) <= obj.MINVEL % check v
                done = true;
                obj.disp_INFO(sprintf('Velocity out of bounds.  Distance to target: %d m, V: %d mps\n', ...
                    round(dist), round(stateVector(4))));
            elseif relativeBearing > obj.MAX_BEARING
                done = true;
                obj.disp_INFO(sprintf('Bearing too far from target. Distance to target: %d m, bearing: %.2f deg\n', ...
                    round(dist), rad2deg(relativeBearing)));
            end % if
            
            % Assign everything to data struct
            data4rl.currentTime = currentTime; % passed to Python
            data4rl.control = control;
            data4rl.success = success;
            data4rl.done = done;
            data4rl.stateVector = stateVector;
            data4rl.targetStateVector = targetStateVector;
            data4rl.surface_dist = surface_dist;
            data4rl.dist = dist;
            data4rl.relativeBearing = relativeBearing;
            data4rl.relativeTargetBearing = relativeTargetBearing;
            
            data4rl.endSim = endSim; % used in HGV MATLAB code
            data4rl.close = close;
        end % getStateInfo
        
        function action = getActionFromRL(obj)
            % Override getActionFromRL to allow for continuous actions
            if ~isempty(obj.server)
                action = obj.server.getAction();
                obj.disp_DEBUG(sprintf('Received action %d', action));
            else
                if obj.isContinuousMP
                    pullRange = 1;
                    pullMin = rad2deg(obj.spatial.gamma) - 0.5*pullRange;
                    turnRange = 2;
                    turnMin = rad2deg(obj.spatial.psi) - 0.5*turnRange;
                    action = deg2rad([pullRange*rand(1) + pullMin; ...
                        turnRange*rand(1) + turnMin]);
%                     action = deg2rad([11*rand(1) - 10; turnRange*rand(1) + turnMin]);
                    obj.disp_DEBUG(sprintf('No DAF server in orchestrator: random action %s', mat2str(action)));
                else
                    action = randi([1,obj.numActions],1);
                    obj.disp_DEBUG(sprintf('No DAF server in orchestrator: random action %d', action));
                end % if/else
            end % if
        end % getActionFromRL

        function updateMovementUpdateRate(obj, data4rl)
            % Sets base update rate
            if data4rl.close
                obj.movementUpdateRate = obj.closeUpdateRate; 
            else
                obj.movementUpdateRate = obj.FARUPDATERATE;
            end % if
            
            % Checks if event will trigger before end of update rate
            if ~isempty(obj.spatial.getPrimitive())
                timeToEvent = ...
                    obj.movementManager.predictEventTime( ...
                    obj.spatial.newSpatial(), obj.movementUpdateRate);

                % Only update if event triggers in middle of trajectory
                if timeToEvent > obj.movementManager.upsilon
                    obj.movementUpdateRate = timeToEvent;
                end % if
            end % if
        end % updateMovementUpdateRate

        % DESCRIPTION: (extended from BaseHGV)
        %   initializes var in hgvmp object with default fields/values
        function init(obj)
            if obj.isContinuousMP
                movement = a4h.funcs.movement.HGVmpContinuousMovementManager();
            else
                movement = a4h.funcs.movement.HGVmpMovementManager();
                if ~isempty(obj.mpList)
                    movement.setMPList(obj.mpList);
                end % if
            end % if/else
            
            movement.setWorld(obj.world);
            movement.minAltitude = obj.MINALT - obj.altitudeTolerance;
            obj.setMovementManager(movement);
            obj.numActions = length(movement.mpList);
            
            obj.scheduleAtTime(obj.movementLastUpdateTime);
            obj.scheduleAtTime(...
                obj.movementLastUpdateTime + obj.movementUpdateRate, ...
                @obj.moveHGV);
        end % init

        function fini(~) % obj
            % Actions for HGV agent at end of sim
        end % fini

        % Overrides updateControl from BaseHGV
        function updateControl(obj, action)
            obj.spatial.setPrimitive(action);
            obj.movementManager.setPrimitive(action, obj.spatial.stateVector());
        end % updateControl
    end % public methods
end % classdef HGVmp