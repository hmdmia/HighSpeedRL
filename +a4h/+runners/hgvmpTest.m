classdef hgvmpTest < daf_sim.PythonRunnerTemplate

    properties (Constant)
        LOG_PATH = 'tmp/hgvmpTest'
        TEST_DURATION = 1000  % secs

        PRED_NAME = 'Predator'
        PRED_AGENT_CLASS = 'a4h.agents.concrete.HGVmp'
        PRED_STATE = [4e4, deg2rad(35.1), deg2rad(-106.6), 6e3, 0, deg2rad(45)]
        PRED_ACCELERATION = [0, 0, 0]
        
        PREY_NAME = 'Prey'
        PREY_STATE = [2e4, deg2rad(40.4), deg2rad(-86.9), 0, 0, 0]
        PREY_SPEED = 200 % m/s
        PREY_ACCELERATION = [0, 0, 0]

        INITIAL_SIM_TIME = 0
    end

    properties
        logPath = a4h.runners.hgvmpTest.LOG_PATH  % Exact prop req'd by RunnerTemplate
        isLogging = false;
    end

    methods
        function out = orchestrateAndRun(obj, simDurationSecs, dafServer, params)
            % Build orchestrator
            orch = a4h.Orchestrator();
            orch.init(obj.LOG_PATH);
            orch.setEndTime(simDurationSecs);
            
            world = a4h.util.AtmosphericWorld();

            % Create predator
            [state.position, state.velocity] = ...
                a4h.util.Spatial.stateVector2ecef(params.initialState, ...
                world);
            state.acceleration = obj.PRED_ACCELERATION;
            pred = eval([obj.PRED_AGENT_CLASS, '();']);
            pred.setWorld(world);
            pred.setInitialState(obj.INITIAL_SIM_TIME, state);
            orch.addAgent(pred, obj.PRED_NAME);

            % Configure predator
            simSecs = simDurationSecs - 1;
            pred.configure(dafServer, simSecs);
            pred.setObservableManager(orch.simObservableManager);
            
            % Assign optional params to agent
            if isfield(params, 'mpList')
                pred.mpList = [params.mpList{:}]';
            end % if
            if isfield(params, 'captureDistance')
                pred.captureDistance = params.captureDistance;
            end % if
            if isfield(params, 'altitudeTolerance')
                pred.altitudeTolerance = params.altitudeTolerance;
            end % if
            if isfield(params, 'isContinuousMP')
                pred.isContinuousMP = params.isContinuousMP;
            end % if
            
            % Create prey
            [state.position, state.velocity] = ...
                a4h.util.Spatial.stateVector2ecef(params.targetState, ...
                world);
            state.acceleration = obj.PREY_ACCELERATION;
            prey = a4h.agents.FixedAltitudeTarget();
            prey.setWorld(world);
            prey.setInitialState(obj.INITIAL_SIM_TIME, state);
            
            orch.addAgent(prey, obj.PREY_NAME);
            orch.simObservableManager.addObservable(prey);  % Add to pred observables (seems unfair)

            % Finish up and run sim
            orch.fini();
            
            % Turns off logging for Loggable agents during RL
            callees = orch.simInst.CalleeMap;
            keys = cell2mat([callees.keys]);
            
            for key = keys
                callee = callees(key);
                if isa(callee, 'base.sim.Loggable')
                    callee.setAddDefaultLogs(obj.isLogging);
                end % if
            end % for

            if ~obj.isLogging
                orch.simInst.internalLogger.setLogLevel(orch.simInst.Logger.log_WARN)
            end % if
            
            % Runs simulation
            orch.run();
            out.inputData = obj.getInputStruct();  % TODO Guess required by something?
            
            % Analyzes run when logged
            if obj.isLogging
                obj.analyzeRun();
            end % if
        end % orchestrateAndRun
        
    end % object methods

    methods (Static)
        function test()
            % Set parameters (runnerObj not used in orchestrateAndRun
            runnerObj = a4h.runners.hgvmpTest();
            params.isLogging = true;
            params.initialState = runnerObj.PRED_STATE;
            params.targetState = runnerObj.PREY_STATE;
            
            % Run
            runnerObj.runner(runnerObj.TEST_DURATION, [], params);
        end % test
        
        function testMovingTarget()
            % Set parameters (runnerObj not used in orchestrateAndRun
            runnerObj = a4h.runners.hgvmpTest();
            params.isLogging = true;
            params.initialState = runnerObj.PRED_STATE;
            params.targetState = runnerObj.PREY_STATE;
            params.targetState(4) = runnerObj.PREY_SPEED;
            
            % Run
            runnerObj.runner(runnerObj.TEST_DURATION, [], params);
        end

        function testContinuous()
            % Set parameters (runnerObj not used in orchestrateAndRun
            runnerObj = a4h.runners.hgvmpTest();
            params.isLogging = true;
            params.isContinuousMP = true;
            params.initialState = runnerObj.PRED_STATE;
            params.targetState = runnerObj.PREY_STATE;
            params.targetState(4) = runnerObj.PREY_SPEED;
            
            % Run
            runnerObj.runner(runnerObj.TEST_DURATION, [], params);
        end
        
        function analyzeRun()
            clear
            close all
            runnerObj = a4h.runners.hgvmpTest();
            logger = base.sim.Logger(runnerObj.logPath);
            logger.restore(); % reads logs, i.e. restores to memory
            coordinator = stdlib.analysis.Coordinator();
            coordinator.setLogger(logger);
            
            % Movement analyzer
            movement = coordinator.requestAnalyzer('stdlib.analysis.basic.Movement');
            earth = base.world.Earth();
            earth.setModel('spherical');
            movement.plotOnEarth(earth);
            movement.plotPositions();
            
            % Topic analyzer
%             topics = coordinator.requestAnalyzer('stdlib.analysis.comms.Topic');
%             topics.listTopicsWithAgents();
            
            % Custom analyzer
            analyzer = coordinator.requestAnalyzer('a4h.analysis.Spatial');
            analyzer.plotState();
            analyzer.plotPrimitives();
        end % analyzeRun

        function out = runner(simDurationSecs, dafServer, params)
            runner = a4h.runners.hgvmpTest();
            runner.isLogging = params.isLogging;
            
            out = runner.orchestrateAndRun(simDurationSecs, dafServer, params);
        end

        function params = getParams()
            % Called by DAF server to send params to Python
            params = struct();
        end
    end
end
