classdef HGVmpContinuousMovementManager < a4h.funcs.movement.HGVmpMovementManager
    % Movement manager for HGV vehicle with continuous MP control
    % In this case, the "primitive" is a vector containing a target value
    % for FPA ("primitive(1)") and heading angle ("primitive(2)")
    properties (Constant)
        CONTROLLENGTH = 2
    end % Constant properties

    methods
        function obj = HGVmpContinuousMovementManager()
            obj=obj@a4h.funcs.movement.HGVmpMovementManager();

            obj.mpList = [];
            obj.trimMP = [];
        end % CONSTRUCTOR

        % Overrides updateLocation to allow the pull/turn/trim approach
        function [new_spatial, start_spatial, event, t, y, u] = updateLocation(obj, spatial, time_offset) % TODO separate control from spatial? -kezra 06/17/2021
            [spatial2, start_spatial, event, t, y, u] = obj.integrateThroughEvents(spatial, time_offset);
            % Returns current state if update is smaller than tolerance
            % or if initial state activates terminal events
            terminalEventValues = obj.integrationEvents(0, start_spatial.stateVector());
            if time_offset < obj.upsilon || ...
                    min(abs(terminalEventValues) < obj.upsilon)
                new_spatial = spatial2;
                return
            end % if
            
            time_offset = time_offset - t(end);

            % Integrate remainder of time trimming MP.
            if ~obj.hasTriggeredIntegrationEvent(spatial2) && time_offset > obj.upsilon
                obj.setPrimitive(obj.trimMP, spatial2.stateVector())
                [new_spatial, ~, event, t_new, y_new, u_new] = integrateUntilEvent(obj, spatial2, time_offset);
                t_new = t_new + t(end);
            else
                new_spatial = spatial2.newSpatial();
                t_new = [];
                y_new = [];
                u_new = [];
            end % if

            t = [t; t_new];
            y = [y; y_new];
            u = [u; u_new];
        end % updateLocation
        
        function [new_spatial, start_spatial, event, t, y, u] = integrateThroughEvents(obj, spatial, time_offset)
            start_spatial = spatial();
            stateVector = start_spatial.stateVector();
            t = [];
            y = [];
            u = [];
            event = [];
            new_spatial = start_spatial.newSpatial();
            
            % Returns current state if update is smaller than tolerance
            % or if initial state activates terminal events
            terminalEventValues = obj.integrationEvents(0, stateVector);
            if time_offset < obj.upsilon || ...
                    min(abs(terminalEventValues) < obj.upsilon)
                return
            end % if
            
            % Set 1st primitive event
            obj.setPrimitive(start_spatial.getPrimitive(), start_spatial.stateVector());
            
            % First integrate 1 or 2 events to achieve MP
            if ~obj.hasTriggeredEvent(start_spatial)
                % If both pull and turn, req. two MP integrations
                % Otherwise, only 1 event required.
                if strcmp(obj.controlType, 'pullturn')
                    [spatial1, start_spatial, event, t_new, y_new, u_new] = integrateUntilEvent(obj, start_spatial, time_offset);
                    t1 = t_new(end);

                    % Set 2nd primitive event
                    obj.setPrimitive(start_spatial.getPrimitive(), spatial1.stateVector());
                else
                    spatial1 = start_spatial.newSpatial();
                    t1 = 0;
                    t_new = [];
                    y_new = [];
                    u_new = [];
                end % if/else

                t = [t; t_new];
                y = [y; y_new];
                u = [u; u_new];
                time_offset = time_offset - t1;
                
                % Only integrate if time remains and other event has not
                % been triggered.
                if ~obj.hasTriggeredEvent(spatial1) && time_offset > obj.upsilon
                    [new_spatial, ~, event, t_new, y_new, u_new] = integrateUntilEvent(obj, spatial1, time_offset);
                    t_new = t_new + t1;
                else
                    new_spatial = spatial1.newSpatial();
                    t_new = [];
                    y_new = [];
                    u_new = [];
                end % if/else

                t = [t; t_new];
                y = [y; y_new];
                u = [u; u_new];
            else
                t = 0;
                y = stateVector.';
                u = obj.calcControl(stateVector, obj.start_spatial.vehicleType);
                start_spatial.setControl(u);
                event = [];
                new_spatial = start_spatial.newSpatial();
            end % if
        end % integrateThroughEvents

        function t = predictEventTime(obj, spatial, time_offset)
            [~, ~, ~, t, ~, ~] = obj.integrateThroughEvents(spatial, time_offset);
            t = t(end);
        end % predictEventTime

        function [alpha, sigma] = calcControl(obj, stateVector, vehicleType)
            isPull = contains(obj.controlType, 'pull');
            isTurn = contains(obj.controlType, 'turn');

            a_ref = vehicleType.reference_area;
            mass = vehicleType.mass;
            
            if isvector(stateVector)
                h = stateVector(1);
                v = stateVector(4);
                gam = stateVector(5);
                psi = stateVector(6);
            else
                h = stateVector(:,1);
                v = stateVector(:,4);
                gam = stateVector(:,5);
                psi = stateVector(:,6);
            end
                
            cl0 = obj.cl_vec(1);
            cl1 = obj.cl_vec(2);
            cl2 = obj.cl_vec(3);
            cl3 = obj.cl_vec(4);
            
            [temp, ~, rho] = obj.world.calcAtmosphere(h);
            mach = obj.world.calcMach(temp, v);
            r = obj.world.getRadius() + h;
            mu = obj.world.getGravParam();
            q = 0.5*rho.*v.^2;
            

            sigma = (isTurn & isPull) * obj.direction(2) * deg2rad(20) * ones(length(h),1) ...
                + (isTurn & ~isPull) * obj.direction(2) * deg2rad(60) * ones(length(h),1) ...
                + ~isTurn * obj.gainTrimGamma .* a4h.util.Spatial.wrapAngle(obj.targetPsi - psi);

            alpha_trim = (-a_ref*q.*r.^2.*(cl0 + cl2*exp(cl3*mach)).*cos(sigma) ...
                + mass*mu*cos(obj.targetGamma) - mass*r.*v.^2*cos(obj.targetGamma)) ...
                ./ (a_ref*cl1*q.*r.^2.*cos(sigma));
            
            alpha = alpha_trim...
                + isPull * obj.gainPullGamma * obj.direction(1)...
                + ~isPull .* obj.gainTrimGamma * (obj.targetGamma - gam);
            
            [alpha, sigma] = obj.saturateControl(alpha, sigma);
        end % calcControl

        function setPrimitive(obj, primitive, stateVector)
            
            if ~isempty(primitive) 
                reqPull = abs(primitive(1) - stateVector(5)) > obj.upsilon;
                reqTurn = abs(primitive(2) - stateVector(6)) > obj.upsilon;

                if reqPull && reqTurn
                    obj.controlType = 'pullturn';
                elseif reqPull
                    obj.controlType = 'pull';
                elseif reqTurn
                    obj.controlType = 'turn';
                else
                    obj.controlType = 'trim';
                end % if/else
            else
                obj.controlType = 'trim';
            end % if/else
            
            switch obj.controlType
                case 'pullturn'
                    obj.targetGamma = primitive(1);
                    obj.targetPsi = primitive(2);
                    obj.direction = [sign(primitive(1) - stateVector(5)); ...
                        sign(a4h.util.Spatial.wrapAngle(primitive(2) - stateVector(6)))];
                    obj.odeOptions.Events = @(t, x) obj.generatePullTurnEvents(t, x);
                case 'pull'
                    obj.targetGamma = primitive(1);
                    obj.targetPsi = stateVector(6);
                    obj.direction =  [sign(primitive(1) - stateVector(5)); 0];
                    obj.odeOptions.Events = @(t, x) obj.generatePullEvent(t, x);
                case 'turn'
                    obj.targetGamma = stateVector(5);
                    obj.targetPsi = primitive(2);
                    obj.direction = [0; sign(a4h.util.Spatial.wrapAngle(primitive(2) - stateVector(6)))];
                    obj.odeOptions.Events = @(t, x) obj.generateTurnEvent(t, x);
                otherwise
                    obj.targetGamma = stateVector(5);
                    obj.targetPsi = stateVector(6);
                    obj.direction = [0; 0];
                    obj.odeOptions.Events = @(t, x) obj.integrationEvents(t, x);
            end % switch
        end % setPrimitive

        % Adds event to the base events in the HGVMovementManager
        % value(end) terminates integration when gamma/psi are reached
        function [value, isterminal, direction] = generatePullTurnEvents(obj, t, x)
            [value, isterminal, direction] = obj.integrationEvents(t, x);
            value = [value; x(5) - obj.targetGamma; x(6) - obj.targetPsi];
            isterminal = [isterminal; 1; 1];
            direction = [direction; 0; 0];
        end % generatePullEvent
    end % object methods

        %%% TEST METHODS %%%
    methods (Static)
        function testPrimitiveProgression
            clear
            close all

            world = a4h.util.AtmosphericWorld();
            startState = [4e4,0,0, 6e3,0,0];
            targetGampsi = deg2rad([1, 5]);

            spatial = a4h.util.SpatialMP.stateVector2spatial(startState, a4h.util.HGVType, world);
            moveMan = a4h.funcs.movement.HGVmpContinuousMovementManager();
            moveMan.setWorld(world);

            % How long MP will run
            time_offset = 150; % s

            spatial.setPrimitive(targetGampsi);
            moveMan.setPrimitive(targetGampsi, startState);
            [~, ~, event,t, y, u] = moveMan.updateLocation(spatial, time_offset);

            fprintf('Flag: %d\n', event.flag)
            fprintf('Event time: %f\n', event.time)
            
            deg = char(176);
            title = sprintf('FPA %.0f --> %.0f%c, Head. %.0f --> %.0f%c', ...
                rad2deg(startState(5)), rad2deg(targetGampsi(1)), deg, ...
                rad2deg(startState(6)), rad2deg(targetGampsi(2)), deg);
            a4h.funcs.movement.HGVmpMovementManager.graphAngles(t, y, u, title);
            a4h.funcs.movement.HGVmpMovementManager.graphStates(t, y, title);
        end % testPrimitiveProgression
    end % Static tester methods
end % CLASS DEF
