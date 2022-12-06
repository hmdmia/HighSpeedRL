classdef hgvAoaTest < daf_sim.PythonRunnerTemplate

    properties (Constant)
        LOG_PATH = 'tmp/hgvAoaTest' 
        TEST_DURATION = 100  % secs

        NAME = 'hgv'
        AGENT_CLASS = 'a4h.agents.concrete.HGVAoa'
        % STATE = [altitude, theta, phi, velocity, gamma, psi]
        STATE = [3e4, deg2rad(0), deg2rad(0), 3e3, 0, deg2rad(0)]
        %STATE = [4e4, deg2rad(35.1), deg2rad(-106.6), 6e3, 0, deg2rad(45)]
        ACCELERATION = [0, 0, 0]
        
    end

    properties
        logPath = a4h.runners.hgvAoaTest.LOG_PATH  % Exact prop req'd by RunnerTemplate
    end

    methods
        function out = orchestrateAndRun(obj, simDurationSecs, dafServer, params)
            % Build orchestrator
            orch = a4h.Orchestrator();
            orch.init(obj.LOG_PATH);
            orch.setEndTime(simDurationSecs);
            
            world = a4h.util.AtmosphericWorld();

            % Create agent
            [state.position, state.velocity] = ...
                a4h.util.Spatial.stateVector2ecef(params.initialState, ...
                world.getRadius());
            state.acceleration = obj.ACCELERATION;
            agent = eval([obj.AGENT_CLASS, '(state);']);
            orch.addAgent(agent, obj.NAME);

            % Configure agent
            simSecs = simDurationSecs - 1;
            agent.configure(dafServer, simSecs);
            contested = a4h.agents.contested.contestedManager;
            contested.init();
            agent.setContestedManager(contested);
            
            % Finish up and run sim
            orch.fini();
            
%             % Turns off logging for Loggable agents during RL, on/off for
%             analyseRun
%             callees = orch.simInst.CalleeMap;
%             keys = cell2mat([callees.keys]);
%             
%             for key = keys
%                 callee = callees(key);
%                 if isa(callee, 'base.sim.Loggable')
%                     callee.setAddDefaultLogs(false);
%                 end % if
%             end % for
            
            % Runs simulation
            orch.run();
            out.inputData = obj.getInputStruct();  
        end % orchestrateAndRun
    end % object methods

    methods (Static)
        function test()
            runnerObj = a4h.runners.hgvAoaTest();
            params.initialState = runnerObj.STATE;
            runnerObj.runner(runnerObj.TEST_DURATION, [], params);
        end % test
        
        function analyzeRun()
            clear
            close all
            runnerObj = a4h.runners.hgvAoaTest();
            logger = base.sim.Logger(runnerObj.logPath);
            logger.restore(); 
            coordinator = stdlib.analysis.Coordinator();
            coordinator.setLogger(logger);
            
            % Movement analyzer
            movement = coordinator.requestAnalyzer('stdlib.analysis.basic.Movement');
            earth = base.world.Earth();
            earth.setModel('spherical');
            movement.plotOnEarth(earth);
                       
            % Custom analyzer
            analyzer = coordinator.requestAnalyzer('a4h.analysis.SpatialAoa');
            analyzer.plotState();
            % analyzer.plotControl();
        end % analyzeRun
        
        function out = runner(simDurationSecs, dafServer, params)
            runner = a4h.runners.hgvAoaTest();
            out = runner.orchestrateAndRun(simDurationSecs, dafServer, params);
        end

        function params = getParams()
            params = struct();           
            params.dt = 1; 
        end
    end
end