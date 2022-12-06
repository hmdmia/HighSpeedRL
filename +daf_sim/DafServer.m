classdef DafServer < handle
    % DAFSERVER  Service network requests to perform DAF simulation tasks

    properties (Constant)
        % Connection parameters
        SOCKET_TIMEOUT = 5  % secs
        CONNECTION_TIMEOUT = 60 * 5  % secs
        LARGE_BUFSIZE = 5120  % bytes

        % Create single definition file common between MATLAB and Python

        % Messages:
        %    Format: "[{<header>}, {<payload>}]
        %    Header: "'type':<msg_type>"
        %    Payload: "<name1>:<value2>, <name2>:<value2>, ...'

        % Message types with payload descriptions
        RUNSIM = "RUNSIM"  % <runner_name>,<sim_time><incomming_parameters>
        PARAMS = "PARAMS"  % <outgoing_parameters>
        STATE = "STATE"    % <var_name1>:<var_value1>...
        ACTION = "ACTION"  % <action_number>
        EXIT = "EXIT"      % (none)
        OK = "OK"          % (none)
        ERROR = "ERROR"    % <error_info>

        % Reserved header field names
        MTYPE = 'type'

        % Reserved payload field names
        RUNNER = 'runner'
        SIM_TIME = 'sim_time'
        ACTION_NUM = 'action_num'
    end

    properties
        port  % network port number
        socket  % communication object
        debugging % true=output debug info
    end

    methods
        function obj = DafServer(port, debug)
            obj.port = port;
            obj.debugging = debug;
            obj.socket = [];
        end

        function info(obj,msg) %#ok<INUSL>
            dbs = dbstack;
            fprintf('INFO: %s (%s:%d)\n',msg,dbs(2).file,dbs(2).line);
        end

        function debug(obj,msg)
            if obj.debugging
                dbs = dbstack;
                fprintf('DEBUG: %s (%s:%d)\n',msg,dbs(2).file,dbs(2).line);
            end
        end

        function serve(obj)
            obj.info(sprintf('DAF listening on port %d',obj.port));
            serving = true;
            tic();

            % Outer comm loop
            while serving

                if isempty(obj.socket)
                    try
                        % TODO tcpip() deprecated, change to tcpserver() in 2020b
                        obj.socket = tcpip('localhost',obj.port,'NetworkRole','server', ...
                                           'Timeout',obj.SOCKET_TIMEOUT); %#ok<TNMLP>
                        obj.socket.InputBufferSize = obj.LARGE_BUFSIZE;
                        obj.socket.OutputBufferSize = obj.LARGE_BUFSIZE;  % TODO Remove w/sane state sizes?
                        fopen(obj.socket);
                    catch
                        % NOTE Exception often due port-in-use from pending client requets(s).
                        %      Retries should pick them up. When moving to tcpserver(), specify
                        %      port sharing enabled or equivalent to avoid errors.
                        continue;
                    end
                end

                [header, payload] = obj.receive();

                if header.type == obj.RUNSIM
                    % Reply with runner's params, exec runner (sim)), clear comm buffer
                    % NOTE: The sim  itself is an inner comm loop
                    obj.send(obj.PARAMS,obj.getParams(payload.runner));
                    status = obj.runSim(payload.runner, payload.sim_time, payload);
                    obj.send(status);

                elseif header.type == obj.EXIT || toc() > obj.CONNECTION_TIMEOUT
                    fclose(obj.socket);
                    obj.debug('Socket closed');
                    serving = false;
                else
                    obj.info(sprintf('Discarded unrecognized msg: header=%s, payload=%s', ...
                                        jsonencode(header), jsonencode(payload)));
                end % if
            end % while
        end % serve

        function send(obj, mtype, payload)
            if ~exist('payload','var')
                payload = struct();
            end

            outMsg = jsonencode({struct(obj.MTYPE,mtype),payload});
            obj.debug(sprintf('COMM|MATLAB->|%s', outMsg));

            try
                fwrite(obj.socket,outMsg);
            catch exception
                obj.info(['ERROR: Exception during send(): "' getReport(exception) '"']);
            end
        end

        function [header, payload] = receive(obj)
            tic();

            while obj.socket.BytesAvailable < 1
                pause(0.01); % TODO Verify appropriate wait time

                if toc() > obj.CONNECTION_TIMEOUT && ~usejava('desktop')
                    obj.info('ERROR: Timeout during receive()');
                    exit();
                end
            end

            inMsg = fread(obj.socket,obj.socket.BytesAvailable);
            inMsg = char(inMsg);
            inMsg = transpose(inMsg);
            obj.debug(sprintf('COMM|MATLAB<-|%s',inMsg));
            inMsg = jsondecode(inMsg);

            if numel(inMsg) ~= 2
                obj.info('ERROR: Received incorrectly formatted message!');
                exit();
            end

            header = inMsg{1};
            payload = inMsg{2};
        end

        function params = getParams(~, runner)
            % Pass params from runner to calling process

            % Exec runner's getParams() method
            % TODO Why warning? Figure out, end it, and remove suppression
            warning('off','MATLAB:Java:classLoad');
            params = eval(strcat(runner,'.getParams();'));
            warning('on','MATLAB:Java:classLoad');
        end

        function status = runSim(obj,runner,sim_secs,params)
            % Exec runner's "runner()" method passing total sim run time
            % Also pass along this server object so agent can use it for comm
            obj.debug(sprintf('Running: %s for %d secs',runner,sim_secs));
            obj.debug(sprintf('Incomming params: "%s"',jsonencode(params)));

            % TODO Why warning? Figure out, end it, and remove suppression
            warning('off','MATLAB:Java:classLoad');
            eval(strcat(runner,'.runner(sim_secs,obj,params);'));  % NOTE return value currently ignored
            warning('on','MATLAB:Java:classLoad');
            status = obj.OK;
        end

        function sendState(obj, state)
            obj.send(obj.STATE, state);
        end

        function actionNum = getAction(obj)
            actionNum = [];
            error = true;
            [header, payload] = obj.receive();

            if header.type == obj.ACTION
                actionNum = payload.action_num;
                error = false;
            end

            if error || isempty(actionNum)
                obj.info(['ERROR: Bad action message: "' jsonencode([header,payload]) '"']);
            end
        end

        function actionNum = sendStateAndGetAction(obj,state)
            obj.sendState(state);
            actionNum = obj.getAction();
        end
    end

    methods (Static)
        function run(port,debug)
            if ~exist('debug','var')
                debug = false;
            end

            server = daf_sim.DafServer(port,debug);
            server.serve();
            fprintf('Exiting....\n');
        end % run
    end % static methods
end % CLASS DEF DafServer
