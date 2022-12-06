classdef TestRunner < base.template.RunnerTemplate

    properties  (Constant)
        LOGPATH  = 'tmp/TestRunner'
    end

    methods (Static)
        function info(msg)
            dbs = dbstack;
            fprintf('INFO: %s (%s:%d)\n',msg,dbs(2).file,dbs(2).line);
        end

        function test()
            daf_sim.TestRunner.info('Testing...');
            daf_sim.TestRunner.runner(999);
        end

        function params = getParams()
            params = struct();
            params.test1 = 1;
            params.test2 = 2;
        end

        % TODO Isn't logpath always required when calling runner()?
        function out = runner(sim_secs, server)
            daf_sim.TestRunner.info(sprintf("Running sim for %d secs",sim_secs));
            out = [];

            orch = daf_sim.Orchestrator();
            orch.init(daf_sim.TestRunner.LOGPATH);
            orch.simInst.internalLogger.setLogLevel(0);
            orch.setEndTime(sim_secs);

            state = struct();
            state.testValue = 42;

            testAgent = daf_sim.TestAgent(server,state);
            orch.addAgent(testAgent,'testAgent');

            orch.fini();
            orch.run();
        end
    end
end
