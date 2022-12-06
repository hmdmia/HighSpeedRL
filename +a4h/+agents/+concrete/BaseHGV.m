classdef BaseHGV < a4h.agents.concrete.HGVAbstract & ...
                    cisa.Sensing

    methods
        function init(obj)
            init@a4h.agents.concrete.HGVAbstract(obj);

            movement = a4h.funcs.movement.HGVMovementManager();
            movement.setWorld(obj.world);  % World should be set in orchestrator
            obj.setMovementManager(movement);

            obj.scheduleAtTime(obj.movementLastUpdateTime);
            obj.scheduleAtTime(...
                obj.movementLastUpdateTime + obj.movementUpdateRate, ...
                @obj.moveHGV);
        end % init

        function data4rl = runAtTime(obj,time)
            % Update position to time if action given
            if obj.initialStateCalculated
                obj.updateMovement(time);
                obj.addDefaultLogEntry(obj.HGV_LOGGING_KEY,obj.spatial.newSpatial());
            end % if
            
            [data4rl, currentTime] = obj.getStateInfo(time);

            % Send an observation to SB3 agent's observe func.
            obj.sendState2RL(currentTime,data4rl);

            % Does state indicate that we're done?
            if data4rl.success
                obj.endThisRun(obj.SUCCESS, currentTime);
            elseif data4rl.done
                obj.endThisRun(obj.FAIL, currentTime);
            elseif data4rl.endSim
                obj.endThisRun(obj.END_SIM, currentTime);
            else % Continue receiving actions if run not done

                % Receive an action from SB3 agent's inner_step func.
                data4rl.action = obj.getActionFromRL();

                % Transform action into desired formate
                action = obj.decideAction(data4rl.action, currentTime);

                if isempty(action)  % Nothing we can do, we're done
                    obj.endThisRun(obj.EMPTY_ACTION, currentTime);
                elseif action == obj.ACTION_EARLY_EXIT
                    obj.endThisRun(obj.END_SIM, currentTime);
                else
                    % Applies control to spatial
                    obj.updateControl(action);

                    % Agent schedules next run
                    % nextUpdateTime ensures movement is updated if sim
                    % times out at end of sim.
                    % NOTE: Python can't handle out of sync messages,
                    % so if sim ended without messaging Python error will
                    % occur
                    obj.updateMovementUpdateRate(data4rl);

                    nextUpdateTime = min(obj.movementUpdateRate, obj.simSecs - currentTime);
                    obj.scheduleAtTime(time+nextUpdateTime);

                    obj.updateMovement(time);
                    obj.addDefaultLogEntry(obj.HGV_LOGGING_KEY,obj.spatial.newSpatial());
                end % if
            end % if
        end % runAtTime

        function [data4rl, currentTime] = getStateInfo(obj, time)
            currentTime = time;
            position = obj.spatial.position;
            velocity = obj.spatial.velocity;
            acceleration = obj.spatial.acceleration;

            data4rl.position = position;
            data4rl.velocity = velocity;
            data4rl.acceleration = acceleration;

            data4rl.success = false;
            data4rl.done = true;
            data4rl.endSim = false;
        end % getStateInfo

        function updateMovementUpdateRate(~, ~) % obj, data4rl
            % Permits data of current state to determine when the agent
            % Should next be updated. By default, no change.
        end % updateMovementUpdateRate
        
        function action = decideAction(~, rlAction, ~)  % obj, rlAction, time
            % take rlAction in format given from Python and transform it to
            % format used in controller. By default, do no transformation.
            action = rlAction;
        end % decideAction

        function updateControl(~, ~)  % obj, action
            % The base HGV's movement does not admit a control.
        end % updateControl
    end % object methods

    %%%% TEST METHODS %%%%

    methods (Static, Access = {?base.test.UniversalTester})
        function tests = test()
            tests = {};
        end
    end
end
