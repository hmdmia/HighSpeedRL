classdef Resetable < base.agents.Movable
    %RESETABLE Agent type to be used for a ResetableInstance simulation
    %   Must include logic to reset agent back to initial state
    
    properties
        initialTime
        initialState
    end % properties

    methods
        function obj = Resetable()

        end % CONSTRUCTOR Resetable

        function setInitialState(obj, time, state)
            setInitialState@base.agents.Movable(obj, time, state)
            obj.initialTime = time;
            obj.initialState = state;
        end % setInitialState
        
        function reset(obj)
            reset@base.agents.Movable(obj)
            obj.setInitialState(obj.initialTime, obj.initialState)
        end % reset
    end % Methods
end % CLASS DEF Resetable

