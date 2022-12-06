classdef Target < base.agents.Periodic & base.agents.Detectable & base.agents.physical.Destroyable

    methods
        function obj = Target(state)
             obj.setInitialState(0,state);
        end

        function init(obj)
            % Do the initialization here.  Note that all destroyables are moveable
            obj.movementManager = base.funcs.movement.NewtonMotion;
            obj.setHeading(obj.getVelocity())
            obj.setDimensions(15,10);
        end

        function v=getPosition(obj)
            v = getPosition@base.agents.Moveable(obj);
            obj.setHeading(obj.spatial.velocity);
        end

        function runAtTime(obj,~)
            obj.getPosition()
            if obj.isDestroyed
                obj.instance.RemoveCallee(obj);
                fprintf('Removing %s from call list \n',obj.commonName)
                obj.instance.endSim();
            end
        end

    end

    %%%% TEST METHODS %%%%

    methods (Static, Access = {?base.test.UniversalTester})
        function tests = test()
            tests = {};
        end
    end

end

