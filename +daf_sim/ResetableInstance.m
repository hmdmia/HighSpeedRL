classdef ResetableInstance < base.sim.Instance
    %RESETABLEINSTANCE Extension of Instance
    % Contains logic for executing multiple episodes where state is
    % reset upon certain conditions
    
    properties
        simTimeLogRate = 10 % TODO move to Instance -wlevin 3/8/2022
        reset
    end
    
    methods
        function obj = ResetableInstance(logpath)
%             obj@base.sim.Instance(logpath)
%             scheduler = daf_sim.ResetableScheduler(obj);
            obj@base.sim.Instance(logpath, 'SchedulerClass', 'daf_sim.ResetableScheduler')
        end

        function initializeRun(obj, startTime, endTime)
            initializeRun@base.sim.Instance(obj, startTime, endTime);
            obj.Scheduler.setInitialSchedule();
        end % initializeRun

%         function AddCallee(obj, callee)
%             AddCallee@base.sim.Instance(obj, callee)
%             assert(isa(callee, 'daf_sim.Resetable'), ['Callee ' callee.commonName ' must be Resetable!'])
%         end % AddCalee
        
        function runUntil(obj, startTime, endTime)
            %Run the simulation from the start to the end time [s]
            obj.initializeRun(startTime, endTime);
            obj.reset = false;
            nextDispTime=-Inf;
            time = obj.startTime;
            totalTimeBeforeReset = 0;

            while ~obj.killSwitch
                if obj.reset
                    obj.resetCallees();
                    obj.Scheduler.resetSchedule();
                    totalTimeBeforeReset = totalTimeBeforeReset + time;

                    obj.reset = false;
                    nextDispTime = -Inf;
                end % if

                [callee,time,calleeFunction]=obj.Scheduler.getNextCallee();
                if isempty(callee) || isempty(time) || (totalTimeBeforeReset + time)>endTime
                    break;
                end
                calleeFunction(time);
                if time>=nextDispTime
                    obj.internalLogger.disp_INFO([' Sim Time ' num2str(round(time)) ' s']);
                    nextDispTime=floor(time/obj.simTimeLogRate)*obj.simTimeLogRate+obj.simTimeLogRate;
                end
            end
            obj.finalizeRun(time);
        end % runUntil

        function resetSim(obj,message)
            if exist('message','var')
                assert(ischar(message),'End message must be a string!');
                obj.internalLogger.disp_INFO(sprintf('Reseting Simulation with Message:\n\t%s',message));
            else
                obj.internalLogger.disp_INFO('Ending Simulation by call to base.sim.Instance.endSim...');
            end

            obj.reset = true;
        end % resetSim

        function resetCallees(obj)
            keyList = obj.CalleeMap.keys;
            for i = 1:length(keyList)
                callee = obj.CalleeMap(keyList{i});  % TODO only call reset is callee is resetable
                if isa(callee, "daf_sim.Resetable")
                    callee.reset();
                end % if

                % TODO: after reset, events are still queued. How do events
                % get appropriately reset? -wlevin 3/8/2022
            end % for


        end % resetCallees
    end % Methods
end % ResetableInstance

