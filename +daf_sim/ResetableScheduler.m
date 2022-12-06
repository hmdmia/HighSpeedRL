classdef ResetableScheduler < base.sim.Scheduler
    %RESETABLESCHEDULER Schedule used with ResetableInstance
    %   Detailed explanation goes here
    
    properties
        initialEventQueue
        initialEventMap
        initialLastMapIdx
        initialCurrentTime
    end
    
    methods        
        function setInitialSchedule(obj)
            %METHOD1 Once inits have been run, save schedule which will be
            %reloaded when resetSchedule is run
            obj.initialEventQueue = obj.eventQueue.clone();
            obj.initialEventMap = containers.Map(obj.eventMap.keys, obj.eventMap.values);
            obj.initialLastMapIdx = obj.lastMapIdx;
            obj.initialCurrentTime = obj.currentTime;
        end % setInitialSchedule

        function resetSchedule(obj)
            obj.eventQueue = obj.initialEventQueue.clone();
            obj.eventMap = containers.Map(obj.initialEventMap.keys, obj.initialEventMap.values);
            obj.lastMapIdx = obj.initialLastMapIdx;
            obj.currentTime = obj.initialCurrentTime;
        end % resetSchedule
    end
end

