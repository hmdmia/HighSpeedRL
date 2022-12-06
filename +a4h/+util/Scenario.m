classdef Scenario < base.build.codeBased.Scenario
    methods
        % override createEarth to make an atmospheric earth
        function createEarth(obj, ~)
            obj.earth = a4h.util.AtmosphericWorld();
        end % createEarth
    end % methods
end % CLASS DEF