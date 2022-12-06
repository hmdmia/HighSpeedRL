classdef PythonRunnerTemplate < base.template.RunnerTemplate

    methods (Static,Abstract)
        out = runner(simSecs, dafServer, params) % This is the runner that will be called by the study.
    end % Abstract, static methods
end % CLASS DEF
