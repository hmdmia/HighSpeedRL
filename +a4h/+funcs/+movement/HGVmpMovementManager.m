classdef HGVmpMovementManager < a4h.funcs.movement.HGVMovementManager
    % Movement manager for HGV vehicle with motion primitive control
    
    properties (Constant)
        ALPHAMAX = deg2rad(20) % angle of attack bounds
        ALPHAMIN = deg2rad(-20)
        SIGMAMAX = deg2rad(89) % bank angle bounds
        SIGMAMIN = deg2rad(-89)
        STATELENGTH = 6 % number of states in eom
        
        % MP list. 'turn' & 'pull' are types of MP's. float corresponds to
        % targetGamma for pull (absolute) & change in targetPsi for turn
        DEFAULT_MP = {...
            'pull', deg2rad(0.5); ... % MP 1
            'pull', deg2rad(0.); ... % MP 2
            'pull', deg2rad(-0.5); ... % MP 3
            'pull', deg2rad(-1); ... % MP 4
            'pull', deg2rad(-1.5); ... % MP 5
            'pull', deg2rad(-2); ... % MP 6
            'turn', deg2rad(-5); ... % MP 7
            'turn', deg2rad(-2.5); ... % MP 8
            'turn', deg2rad(-1); ... % MP 9
            'turn', deg2rad(-0.5); ... % MP 10
            'turn', deg2rad(-0.1); ... % MP 11
            'turn', deg2rad(0.1); ... % MP 12
            'turn', deg2rad(0.5); ... % MP 13
            'turn', deg2rad(1); ... % MP 14
            'turn', deg2rad(2.5); ... % MP 15
            'turn', deg2rad(5)}  % MP 16
        CTRL_TEMPLATE = struct('primitive', [], 'alpha', [], 'sigma', [])
    end % constant properties
    
    % properties used to set the control
    properties
        mpList % list interpreting integer MP as pull/turn/trim command
        targetGamma % gamma value targeted
        targetPsi % psi value targeted
        direction % positive or negative change in angle
        controlType % string with controller type
        trimMP
        
        % controller gains
        gainTrimGamma
        gainPullGamma
        gainTurnGamma
    end % control properties
    
    methods
        function obj = HGVmpMovementManager()
            obj=obj@a4h.funcs.movement.HGVMovementManager();
            
            % changes to parameters in HGVMovementManager
            obj.timestepMax = 50;
            
            % assigning values to new parameters
            obj.mpList = a4h.funcs.movement.HGVmpMovementManager.DEFAULT_MP;
            obj.trimMP = 0;
            
            obj.gainTrimGamma = 25;
            obj.gainPullGamma = deg2rad(2.5);
            obj.gainTurnGamma = 5;
        end % CONSTRUCTOR
        
        % Overrides updateLocation to set controller info
        function [new_spatial, start_spatial, event, t, y, u] = updateLocation(obj, spatial, time_offset) % TODO separate control from spatial? -kezra 06/17/2021
            start_spatial = spatial.newSpatial();
            stateVector = start_spatial.stateVector();
            
            % Returns current state if update is smaller than tolerance
            % or if initial state activates terminal events
            % else integrates for length of time_offset
            terminalEventValues = obj.integrationEvents(0, stateVector);
            if time_offset < obj.upsilon || ...
                    min(abs(terminalEventValues) < obj.upsilon)
                event = [];
                y = [];
                t = [];
                new_spatial = spatial.newSpatial();
                return
            end % if
            
            if ~obj.hasReachedMP(start_spatial)
                [new_spatial, start_spatial, event, t, y, u] = integrateUntilEvent(obj, start_spatial, time_offset);
                time_offset = time_offset - t(end);
            else
                event.flag = obj.triggerEvent();
                t = 0;
                y = stateVector';
                [u(1,1), u(1,2)] = obj.calcControl(y, start_spatial.vehicleType);
                start_spatial.setControl(u(1,:));
                new_spatial = start_spatial.newSpatial();
            end % if
            
            % If integration events (hit ground) not triggered,
            % Integrate for remainder of time, trimming FPA
            if ~obj.hasTriggeredIntegrationEvent(new_spatial) && time_offset > 0
                obj.setPrimitive(obj.trimMP, y(end,:));
                [new_spatial, ~, event, t_new, y_new, u_new] = integrateUntilEvent(obj, new_spatial.newSpatial(), time_offset);
                t_new = t_new + t(end);

                t = [t; t_new];
                y = [y; y_new];
                u = [u; u_new];
            end % if
        end % updateLocation
        
        function [new_spatial, start_spatial, event, t, y, u] = integrateUntilEvent(obj, start_spatial, time_offset)
            stateVector = start_spatial.stateVector();

            % Integrates until MP event reached
            [~, event, t, y] = obj.rk45step(time_offset, stateVector, start_spatial.vehicleType);
            
            dyfdt = obj.eom(t(end), y(end,:), start_spatial.vehicleType);
            new_spatial = start_spatial.newSpatial();
            new_spatial.updateSpatial(y(end,:), dyfdt(4), dyfdt(5), dyfdt(6));
            
            [u(:,1), u(:,2)] = obj.calcControl(y, start_spatial.vehicleType);
            
            start_spatial.setControl(u(1,:));
            new_spatial.setControl(u(end,:));
        end % integrateUntilEvent
        
        function t = predictEventTime(obj, start_spatial, time_offset)
            if obj.hasTriggeredEvent(start_spatial)
                t = 0;
            else
                [~, ~, ~, t, ~, ~] = ...
                    obj.integrateUntilEvent(start_spatial, time_offset);
                t = t(end);
            end
        end % predictEventTime
                
        function bool = hasReachedMP(obj, spatial)
            bool = true;
            
            if contains(obj.controlType, 'pull')
                bool = bool && abs(obj.targetGamma - spatial.gamma) < obj.upsilon;
            end % if

            if contains(obj.controlType, 'turn')
                bool = bool && abs(obj.targetPsi - spatial.psi) < obj.upsilon;
            end % if
        end % hasReachedMP
        
        % Setters/Getters
        function setMPList(obj, mpList)
            obj.mpList = mpList;
        end % setMPList
        
        % Returns which event number corresponds to hitting target FPA or
        % target heading angle, depending on MP.
        function out = triggerEvent(obj)
            out = length(obj.integrationEvents(0, zeros(obj.STATELENGTH,1))) + 1;
        end % triggerEvent
    end % object methods
    
    % Helper methods
    methods
        function setPrimitive(obj, primitive, stateVector)
            % If primitive is not set to one withing the MP list,
            % defaults to trimming FPA at current conditions
            if isempty(primitive) || primitive == 0 || primitive > length(obj.mpList)
                obj.controlType = 'trimFPA';
            else
                obj.controlType = obj.mpList{primitive, 1};
            end % if
            
            switch obj.controlType
                case 'pull'
                    obj.targetGamma = obj.mpList{primitive, 2};
                    obj.targetPsi = stateVector(6);
                    obj.direction =  sign(obj.targetGamma - stateVector(5));
                    obj.odeOptions.Events = @(t, x) obj.generatePullEvent(t, x);
                case 'turn'
                    obj.targetGamma = stateVector(5);
                    obj.targetPsi = obj.mpList{primitive, 2} + stateVector(6);
                    obj.direction = sign(obj.mpList{primitive, 2});
                    obj.odeOptions.Events = @(t, x) obj.generateTurnEvent(t, x);
                otherwise
                    obj.targetGamma = stateVector(5);
                    obj.targetPsi = stateVector(6);
                    obj.direction = 0;
                    obj.odeOptions.Events = @(t, x) obj.integrationEvents(t, x);
            end % switch
        end % setPrimitive
        
        function [alpha, sigma] = calcControl(obj, stateVector, vehicleType)
            isTurn = strcmp(obj.controlType,'turn');
            isPull = strcmp(obj.controlType,'pull');
            
            a_ref = vehicleType.reference_area;
            mass = vehicleType.mass;
            
            if isvector(stateVector)
                h = stateVector(1);
                v = stateVector(4);
                gam = stateVector(5);
            else
                h = stateVector(:,1);
                v = stateVector(:,4);
                gam = stateVector(:,5);
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
            
            sigma = isTurn * obj.direction * deg2rad(75) * ones(length(h),1);
            
            alpha_trim = (-a_ref*q.*r.^2.*(cl0 + cl2*exp(cl3*mach)).*cos(sigma) ...
                + mass*mu*cos(obj.targetGamma) - mass*r.*v.^2*cos(obj.targetGamma)) ...
                ./ (a_ref*cl1*q.*r.^2.*cos(sigma));
            
            alpha = alpha_trim...
                + isPull * obj.gainPullGamma * obj.direction...
                + ~(isTurn || isPull) .* obj.gainTrimGamma * (obj.targetGamma - gam);
            
            [alpha, sigma] = obj.saturateControl(alpha, sigma);
        end % calcControl
        
        % Adds a turn event to the base events in the HGVMovementManager
        % value(end) terminates integration when psi is reached
        % Triggers integration to switch to trimFPA controller
        function [value, isterminal, direction] = generateTurnEvent(obj, t, x)
            [value, isterminal, direction] = obj.integrationEvents(t, x);
            value = [value; x(6) - obj.targetPsi];
            isterminal = [isterminal; 1];
            direction = [direction; 0];
        end % generateTurnEvent
        
        % Adds a turn event to the base events in the HGVMovementManager
        % value(end) terminates integration when gamma is reached
        % Triggers integration to switch to trimFPA controller
        function [value, isterminal, direction] = generatePullEvent(obj, t, x)
            [value, isterminal, direction] = obj.integrationEvents(t, x);
            value = [value; x(5) - obj.targetGamma];
            isterminal = [isterminal; 1];
            direction = [direction; 0];
        end % generatePullEvent
    end % public helper methods
    
    methods (Static)
        % outputs the input saturated by the min/max values for control
        function [alpha, sigma] = saturateControl(alpha, sigma)
            ALPHAMIN = a4h.funcs.movement.HGVmpMovementManager.ALPHAMIN;
            ALPHAMAX = a4h.funcs.movement.HGVmpMovementManager.ALPHAMAX;
            SIGMAMIN = a4h.funcs.movement.HGVmpMovementManager.SIGMAMIN;
            SIGMAMAX = a4h.funcs.movement.HGVmpMovementManager.SIGMAMAX;
            
            alpha = min(max(alpha, ALPHAMIN), ALPHAMAX);
            sigma = min(max(sigma, SIGMAMIN), SIGMAMAX);
        end
    end % static methods
    
    %%% TEST METHODS %%%
    methods (Static)
        function testRK45()
            clear
            close all
            
            position = [6415637,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2

            moveMan = a4h.funcs.movement.HGVmpMovementManager();
            moveMan.setWorld(a4h.util.AtmosphericWorld());
            spatial = a4h.util.SpatialMP(position, velocity, acceleration, a4h.util.HGVType(), moveMan.world);
            spatial.setPrimitive(4);
            moveMan.setPrimitive(4, spatial.stateVector())
            
            [~, ~, t, y] = moveMan.rk45step(300, spatial.stateVector(), spatial.vehicleType);
            
            figure()
            sgtitle('Hit ground event, FPA Trim Control')
            subplot(4,1,1)
            plot(rad2deg(y(:,2)), rad2deg(y(:,3)), 'LineWidth', 2)
            xlabel('Downrange [deg]')
            ylabel('Crossrange [deg]')
            grid on
            
            subplot(4,1,2)
            plot(t, y(:,1)/1000, 'LineWidth', 2)
            xlabel('Time [s]')
            ylabel('Altitude [km]')
            grid on
            
            subplot(4,1,3)
            plot(t, rad2deg(y(:,5)), 'LineWidth', 2)
            xlabel('Time [s]')
            ylabel('FPA [deg]')
            grid on
            
            subplot(4,1,4)
            plot(t, y(:,4), 'LineWidth', 2)
            xlabel('Time [s]')
            ylabel('Velocity [m/s]')
            grid on
        end % testRK45
        
        % Tests the default MP case (MP == 0), i.e. trim to initial
        % conditions, outputting plots
        function testTrimFPA()
            clear
            close all
            
            world = a4h.util.AtmosphericWorld();
            position = [world.getRadius() + 40000,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            spatial = a4h.util.SpatialMP(position, velocity, acceleration, a4h.util.HGVType(), world);
            
            moveMan = a4h.funcs.movement.HGVmpMovementManager();
            moveMan.setWorld(world);

            % How long MP will run
            time_offset = 100; % s
            
            spatial.setPrimitive(0);
            [~, ~, event,t, y, u] = moveMan.updateLocation(spatial, time_offset);
            
            fprintf('Flag: %d\n', event.flag)
            fprintf('Event time: %f\n', event.time)
            
            a4h.funcs.movement.HGVmpMovementManager.graphAngles(t, y, u, sprintf('Trim FPA to %.2f%c', rad2deg(y(1,5)), char(176)));
            a4h.funcs.movement.HGVmpMovementManager.graphStates(t, y, sprintf('Trim FPA to %.2f%c', rad2deg(y(1,5)), char(176)));
        end % testTrimFPA
        
        % Tests default MP's 5-10, outputting plots
        function testPull()
            clear
            close all
            
            world = a4h.util.AtmosphericWorld();
            position = [world.getRadius() + 40000,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            spatial = a4h.util.SpatialMP(position, velocity, acceleration, a4h.util.HGVType(), world);
            
            moveMan = a4h.funcs.movement.HGVmpMovementManager();
            moveMan.setWorld(world);

            % How long MP will run
            time_offset = 100; % s
            
            primList = a4h.funcs.movement.HGVmpMovementManager.DEFAULT_MP;
            
            for i = 1:length(primList)
                if strcmp(primList{i,1}, 'pull')
                    spatial.setPrimitive(i);
                    [~, ~, event,t, y, u] = moveMan.updateLocation(spatial, time_offset);
                    fprintf('MP %d Flag: %d\n', i, event.flag)
                    fprintf('MP %d Event time: %f\n', i, event.time)
                    
                    a4h.funcs.movement.HGVmpMovementManager.graphAngles(t, y, u, sprintf('Pull to %.2f%c [MP %d]', rad2deg(primList{i,2}), char(176), i));
                end % if
            end % for
        end % testPull
        
        function testTurn()
            clear
            close all
            
            world = a4h.util.AtmosphericWorld();
            position = [world.getRadius() + 40000,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            spatial = a4h.util.SpatialMP(position, velocity, acceleration, a4h.util.HGVType(), world);
            
            moveMan = a4h.funcs.movement.HGVmpMovementManager();
            moveMan.setWorld(world);

            % How long MP will run
            time_offset = 50; % s
            
            primList = a4h.funcs.movement.HGVmpMovementManager.DEFAULT_MP;
            
            for i = 1:length(primList)
                if strcmp(primList{i,1}, 'turn')
                    spatial.setPrimitive(i);
                    [~, ~, event,t, y, u] = moveMan.updateLocation(spatial, time_offset);
                    fprintf('MP %d Flag: %d\n', i, event.flag)
                    fprintf('MP %d Event time: %f\n', i, event.time)
                    
                    a4h.funcs.movement.HGVmpMovementManager.graphAngles(t, y, u, sprintf('Turn %.2f%c [MP %d]', rad2deg(primList{i,2}), char(176), i));
                end % if
            end % for
        end % testTurn
        
        function testHitGround()
            clear
            close all
            
            world = a4h.util.AtmosphericWorld();
            position = [world.getRadius() + 40000,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            spatial = a4h.util.SpatialMP(position, velocity, acceleration, a4h.util.HGVType(), world);
            
            moveMan = a4h.funcs.movement.HGVmpMovementManager();
            moveMan.setWorld(world);

            % How long MP will run
            time_offset = 2000; % s
                     
            % MP 0
            spatial.setPrimitive(0);
            [~, ~, event,t, y, u] = moveMan.updateLocation(spatial, time_offset);
            
            fprintf('MP 0 Flag: %d\n', event.flag)
            fprintf('MP 0 Event time: %f\n', event.time)
            
            a4h.funcs.movement.HGVmpMovementManager.graphAngles(t, y, u, sprintf('Hit Ground with FPA Trimmed to %.2f%c', rad2deg(y(1,5)), char(176)));
            a4h.funcs.movement.HGVmpMovementManager.graphStates(t, y, sprintf('Hit Ground with FPA Trimmed to %.2f%c', rad2deg(y(1,5)), char(176)))
        end % testHitGround
        
        function testMP2()
            world = a4h.util.AtmosphericWorld();
            PRED_STATE = [4e4, deg2rad(35.1), deg2rad(-106.6), 6e3, 0, deg2rad(45)];
            [position, velocity] = ...
                a4h.util.Spatial.stateVector2ecef(PRED_STATE, world);
            acceleration = []; % m/s2
            spatial = a4h.util.SpatialMP(position, velocity, acceleration, a4h.util.HGVType(), world);
            
            moveMan = a4h.funcs.movement.HGVmpMovementManager();
            moveMan.setWorld(world);

            % How long MP will run
            time_offset = 3e3; % s
                     
            % MP 0
            spatial.setPrimitive(2);
            [~, ~, ~,t, y, u] = moveMan.updateLocation(spatial, time_offset);
            
            a4h.funcs.movement.HGVmpMovementManager.graphAngles(t, y, u, sprintf('Hit Ground with FPA Trimmed to %.2f%c', rad2deg(y(1,5)), char(176)));
            a4h.funcs.movement.HGVmpMovementManager.graphStates(t, y, sprintf('Hit Ground with FPA Trimmed to %.2f%c', rad2deg(y(1,5)), char(176)))
        end % testMP2
        
        function testMPSeries()
            clear
            close all
          
            world = a4h.util.AtmosphericWorld();
            initialState = [40e3, 0, 0, 6000, 0, 0];
            spatial = a4h.util.SpatialMP.stateVector2spatial(initialState, a4h.util.HGVType(), world);
%             [position, velocity] = a4h.util.Spatial.stateVector2ecef(initialState, world);
% %             position = [world.getRadius() + 40000,0,0];  % m
% %             velocity = [0,6000,0];  % m/s
%             acceleration = [0, 0, 0]; % m/s2
%             spatial = a4h.util.SpatialMP(position, velocity, acceleration, a4h.util.HGVType(), world);
            
            moveMan = a4h.funcs.movement.HGVmpMovementManager();
            moveMan.setWorld(world);
            
            timeStep = 25;
            mpSeries = [3, 5, 3, 7, 11, 8, 11, 9, 11, 6];
            
            mpList = {...
            'turn', deg2rad(-10); ... % MP 1
            'turn', deg2rad(-5); ... % MP 2
            'turn', deg2rad(-2.5); ... % MP 3
            'turn', deg2rad(-1); ... % MP 4
            'pull', deg2rad(0.5); ... % MP 5
            'pull', deg2rad(0); ... % MP 6
            'pull', deg2rad(-0.5); ... % MP 7
            'pull', deg2rad(-1); ... % MP 8
            'pull', deg2rad(-1.5); ... % MP 9
            'pull', deg2rad(-2); ... % MP 10
            'turn', deg2rad(1); ... % MP 11
            'turn', deg2rad(2.5); ... % MP 12
            'turn', deg2rad(5); ... % MP 13
            'turn', deg2rad(10)}; ... % MP 14
            
            moveMan.mpList = mpList;
       
            
            for i = 1:length(mpSeries)
                spatial.setPrimitive(mpSeries(i));
                [spatial, ~, ~,dt, dy, du] = moveMan.updateLocation(spatial, timeStep);
                
                if(i == 1)
                    t = dt;
                    y = dy;
                    u = du;
                else
                    t = [t; dt + t(end)]; %#ok<AGROW>
                    y = [y; dy]; %#ok<AGROW>
                    u = [u; du]; %#ok<AGROW>
                end % if
            end % for
            
%             a4h.funcs.movement.HGVmpMovementManager.graphAngles(t, y, u, 'MP series [3, 5, 3, 7, 11, 8, 11, 9, 11, 6]');
%             a4h.funcs.movement.HGVmpMovementManager.graphStates(t, y, 'MP series [3, 5, 3, 7, 11, 8, 11, 9, 11, 6]');
            
            % Compare with data from Python
            data = table2array(readtable('tmp/mp_comp.csv'))';
            
            t_python = data(:,1);
            y_python = data(:,2:7);
            u_python = data(:,8:9);
            
            % Altitude comparison
            figure()
            subplot(2,1,1)
            plot(t, y(:,1)/1000)
            hold on
            plot(t_python, y_python(:,1)/1000, 'o')
            hold off
            grid on
            xlabel('Time [s]')
            ylabel('Altitude [km]')
            legend('MATLAB', 'Python', 'Location', 'best')
            
            % Velocity comparison
            subplot(2,1,2)
            plot(t, y(:,4)/1000)
            hold on
            plot(t_python, y_python(:,4)/1000, 'o')
            hold off
            grid on
            xlabel('Time [s]')
            ylabel('Velocity [m/s]')
            legend('MATLAB', 'Python', 'Location', 'best')
            
            % Angles comparison
            figure()
            colors = get(gca,'colororder');

            plot(t, rad2deg(y(:,2)), 'Color', colors(1,:))
            hold on
            plot(t, rad2deg(y(:,3)), 'Color', colors(2,:))
            plot(t, rad2deg(y(:,5)), 'Color', colors(3,:))
            plot(t, rad2deg(y(:,6)), 'Color', colors(4,:))
            
            plot(t_python, rad2deg(y_python(:,2)),'o', 'Color', colors(1,:))
            plot(t_python, rad2deg(y_python(:,3)),'o', 'Color', colors(2,:))
            plot(t_python, rad2deg(y_python(:,5)),'o', 'Color', colors(3,:))
            plot(t_python, rad2deg(y_python(:,6)),'o', 'Color', colors(4,:))
            hold off
            grid on
            xlabel('Time [s]')
            ylabel('Angle [deg]')
            legend('\theta (MATLAB)', '\phi (MATLAB)', '\gamma (MATLAB)', '\psi (MATLAB)', ...
                '\theta (Python)', '\phi (Python)', '\gamma (Python)', '\psi (Python)', ...
                'Location', 'best')
            
            figure()
            plot(t, rad2deg(u(:,1)), 'Color', colors(1,:))
            hold on
            plot(t, rad2deg(u(:,2)), 'Color', colors(2,:))
            plot(t_python, rad2deg(u_python(:,1)), 'o', 'Color', colors(1,:))
            plot(t_python, rad2deg(u_python(:,2)), 'o', 'Color', colors(2,:))
            hold off
            grid on
            xlabel('Time [s]')
            ylabel('Control [deg]')
            legend('\alpha (MATLAB)', '\sigma (MATLAB)', '\alpha (Python)', '\sigma (Python)', 'Location', 'best')
        end % testMPSeries
        
        % Tester helper functions
        function graphAngles(t, y, u, plotTitle)
            figure();
            plot(t, rad2deg(y(:,6)), 'LineWidth', 2)
            hold on
            plot(t, rad2deg(y(:,5)), 'LineWidth', 2)
            plot(t, rad2deg(y(:,2)), '--', 'LineWidth', 2)
            plot(t, rad2deg(y(:,3)), '--', 'LineWidth', 2)
            plot(t, rad2deg(u(:,1)), ':', 'LineWidth', 2)
            plot(t, rad2deg(u(:,2)), ':', 'LineWidth', 2)
            hold off
            xlabel('Time [s]')
            ylabel('Angle [deg]')
            title(plotTitle)
            legend('\psi', '\gamma', '\theta', '\phi', '\alpha', '\sigma','Location', 'best')
            grid on
        end
        
        function graphStates(t, y, figureTitle)
            figure()
            sgtitle(figureTitle)
            subplot(3,1,1)
            plot(rad2deg(y(:,2)), rad2deg(y(:,3)), 'LineWidth', 2)
            xlabel('Downrange [deg]')
            ylabel('Crossrange [deg]')
            grid on
            
            subplot(3,1,2)
            plot(t, y(:,1)/1000, 'LineWidth', 2)
            xlabel('Time [s]')
            ylabel('Altitude [km]')
            grid on
            
            subplot(3,1,3)
            plot(t, y(:,4), 'LineWidth', 2)
            xlabel('Time [s]')
            ylabel('Velocity [m/s]')
            grid on
        end
    end % public tester methods
end % CLASS DEF