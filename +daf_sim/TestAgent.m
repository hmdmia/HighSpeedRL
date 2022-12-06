classdef TestAgent < base.sim.Callee

   properties
        server % Interprocess communications object (conn. to Python code)
        state % Example of data held by agent
    end

    methods
        function obj = TestAgent(server,state)
            obj.server = server;
            obj.state = state;
        end

        function init(obj)
            obj.setLogLevel(base.sim.Logger.log_DEBUG);
            obj.disp_DEBUG('agent init()');
            obj.scheduleAtTime(1);
        end

        function runAtTime(obj,time)
            obj.disp_DEBUG(sprintf('agent runAtTime(%d)',time));
            action = obj.server.sendStateAndGetAction(time);
            obj.disp_DEBUG(sprintf('Action: %d',action));

            if action == 1
                obj.instance.endSim(sprintf('%s: endSim: Early termination!',obj.commonName))
            else
                obj.scheduleAtTime(time+1);
            end
        end

        function fini(obj)
            obj.disp_DEBUG('fini');
        end
    end

    %%%% TEST METHODS %%%%

    methods (Static, Access = {?base.test.UniversalTester})
        function tests = test()
            tests = {};
        end
    end
end
