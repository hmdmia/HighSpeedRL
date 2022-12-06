classdef HGVAoa < a4h.agents.concrete.BaseHGV

    properties (Constant)
        MINFPA = deg2rad(-45) 
        MAXFPA = deg2rad(10)
        MINVEL = 1500 % m/s
        TARGETALT = 3000
        
    end
       
    methods % Public methods
        % DESCRIPTION:
        %   constructor for hgvAoa object
        % OUTPUT:
        %   obj (an HGVAoa object)
        % INPUT:
        %   state   (struct) position, velocity, acceleration vectors
        function obj = HGVAoa(state)
            obj@a4h.agents.concrete.BaseHGV(state);
            obj.agentRunRate = 1; % sec
        end % HGVAoa (constructor)
              
        % DESCRIPTION: (overriden from BaseHGV)
        %   checks and reports info in MSG_TEMPLATE
        % OUTPUT:
        %   data (struct) fields in MSG_TEMPLATE
        %   currentTime (float) time sim. is at due to violation or end of time step [s]
        % INPUT:
        %   obj (HGVAoa object)
        %   time (float) end of period being checked for info [s]
        function [data,currentTime] = getStateInfo(obj,time)
            obj.disp_DEBUG(sprintf('State (t = %d):\n', ...
                time));
            obj.disp_DEBUG(sprintf('%s\n', obj.spatial.str()));
            
            currentTime = time;                
            
            % Logical values
            tooFar = false;
            endSim = false;
            done = false;
            isCaptured = false; % not a useful value for Aoa sim, kept for python
            success = false;
                      
            stateVector = obj.spatial.stateVector();             
             % TODO: implement logic for succes, AoaBaseClass ln 43          
            if time >= obj.simSecs
                fprintf('Simulation timed out. h = %d\n', ...
                    round(stateVector(1)));
                endSim = true;
                done = true;
            elseif (stateVector(5) < obj.MINFPA || stateVector(5) > obj.MAXFPA) % check FPA
                tooFar = true;
                done = true;
                fprintf('FPA out of bounds. h = %d FPA: %.2f\n', ...
                    stateVector(1), rad2deg(stateVector(5)));
            elseif stateVector(4) <= obj.MINVEL % check v
                tooFar = true;
                done = true;
                fprintf('Velocity out of bounds.  h = %d V: %.2f\n', ...
                    stateVector(1), rad2deg(stateVector(4)));
            elseif (stateVector(1) <= obj.TARGETALT) && ...
                    (stateVector(5) >= obj.MINFPA && stateVector(5) <= obj.MAXFPA)% check altitude/fpa
                success = true;
                done = true;
                fprintf('Successful episode.  h = %d FPA: %.2f\n', ...
                    stateVector(1), rad2deg(stateVector(5)));
            else
            end % if
            
            % Assign everything to data struct
            data.done = done;
            data.tooFar = tooFar;
            data.endSim = endSim;
            data.success = success;
            data.isCaptured = isCaptured;
            data.currentTime = currentTime;           
            data.stateVector = stateVector;                
        end % getStateInfo

        % DESCRIPTION: (extended from BaseHGV)
        %   initializes var in hgvAoa object with default fields/values
        function init(obj)
            obj.setInitialState();
            
            movement = a4h.funcs.movement.HGVaoaMovementManager();
            movement.setWorld(obj.world);
            movement.timestepMax = obj.MOVEMENT_INTEGRATION_STEP;
            obj.setMovementManager(movement);
                        
            obj.scheduleAtTime(0);
            obj.scheduleAtTime(obj.movementUpdateRate,@obj.moveHGV);
        end % init

        function updateControl(obj, action)
            obj.spatial.controller = action;
        end % updateControl
         
        function action = getActionFromRL(obj)
            if ~isempty(obj.server)
                action = obj.server.getAction();
                obj.disp_DEBUG(sprintf('Recieved RL action %d', action));
            else
                % default action is Aoa between -20 and 20 degrees
                action = rand(1,1)*range([deg2rad(-20), deg2rad(20)]) + deg2rad(-20);
                obj.disp_DEBUG(sprintf('No DAF server: random action %d', action));
            end % if
        end % getActionFromRL
        
        % rl action comes in as AOA value in radians
        function action = decideAction(obj, rlAction, time) 
            action = rlAction;
        end % decideAction
        
    end % public methods
end % classdef HGVAoa