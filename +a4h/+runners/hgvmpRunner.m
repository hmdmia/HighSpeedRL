log_path = 'tmp/DAF/scenarioBuilder';
mt = a4h.runners.scenarios.MovingTarget(log_path);
mt.build(); % TODO setInitialState called with position, velocity, accel.
            % not accepted by HGVmp.initialState. Change? -wlevin
            % 12/14/2021