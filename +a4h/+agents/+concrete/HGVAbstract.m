classdef HGVAbstract < base.agents.physical.Impactable

    properties (Constant)
        HGV_LOGGING_KEY = 'HGV'

        % Fini messages
        SUCCESS = 'fini-success'
        FAIL = 'fini-fail'
        END_SIM = 'fini-endSim'
        EMPTY_ACTION = 'fini-emptyAction'

        ACTION_EARLY_EXIT = 86
    end

    properties
        server  % Object for communicating w/AI, empty ([]) if in test mode
        simSecs  % Number of seconds to run
        movementUpdateRate % sec
        numActions % integer number of possible actions
    end

    methods
        function obj = HGVAbstract()
            world = a4h.util.AtmosphericWorld();
            obj.world = world;
            
            obj.movementUpdateRate = 1; % sec
            obj.numActions = 1;
        end

        function setInitialState(obj, initialTime, initialState)
            if iscell(initialState)
                initialState = obj.cellState2structState(initialState);
            end % if
            
            setInitialState@base.agents.Movable(obj, ...
                initialTime, initialState);

            obj.spatial = a4h.util.Spatial(initialState.position, ...
                initialState.velocity, initialState.acceleration, ...
                a4h.util.HGVType(), obj.world);
        end % setInitialState

        function configure(obj, server, simSecs)  % Test mode: server = []
            assert(isempty(server) || base.util.code.isa(class(server),'daf_sim.DafServer'),...
                'Server input must be of type DafServer or empty!');
            obj.server = server;
            obj.simSecs = simSecs;
        end

        function moveHGV(obj,time)
            obj.updateMovement(time);
            obj.scheduleAtTime(time+obj.movementUpdateRate,@obj.moveHGV);
        end

        function endThisRun(obj, finiStr, time)
            obj.instance.endSim(sprintf('%s: fini=%s, time=%f', ...
                                obj.commonName, finiStr, time));
        end

        function sendState2RL(obj, time, stateStruct)
            stateStruct.fini = '';  % Empty fini string indicates sim continues
            obj.disp_DEBUG(sprintf('Sending state, time=%d', time));

            if ~isempty(obj.server)
                obj.server.sendState(stateStruct); % 'sendStruct' changed to 'stateStruct' not sure if typo or not -wlevin 7/2/2021
            end
        end

        function action = getActionFromRL(obj)
            if ~isempty(obj.server)
                action = obj.server.getAction();
                obj.disp_DEBUG(sprintf('Received action %d', action));
            else
                action = randi([1,obj.numActions],1);
                obj.disp_DEBUG(sprintf('No DAF server in orchestrator: random action %d', action));
            end % if
        end % getActionFromRL
    end % object methods

    %%%% TEST METHODS %%%%

    methods (Static, Access = {?base.test.UniversalTester})
        function tests = test()
            tests = {};
        end
    end

    methods (Abstract)
        [data4rl, currentTime] = getStateInfo(obj, time)
        action = decideAction(obj, rlAction, time)
        updateControl(obj,action)
    end
end
