classdef HGVMovementManager < base.funcs.movement.StateManager
    % Movement manager for HGV vehicle. Updates the location of HGV
    % using atmosphere and aero functions with input state/control.
    
    properties
        upsilon = 1e-5 % minimum time between integration steps
        world % AtmosphericWorld object
        timestepMax % [s] Max time step before updating controller
        logFile
        cl_vec
        cd_vec
        odeOptions
        minAltitude
    end
    
    methods
        function obj = HGVMovementManager()
            obj=obj@base.funcs.movement.StateManager();
            
            obj.timestepMax = 1;
            
            % Coefficients used for aerodynamic properties
            obj.cl_vec = [-0.2317, 0.0513 * 180 / 3.14159, 0.2945, -0.1028];
            obj.cd_vec = [0.024, 7.24e-4 * 180^2 / 3.14159^2, 0.406, -0.323];
            
            % Options used to integrate
            obj.minAltitude = 0;
            obj.odeOptions = odeset('Events',@(t, x) obj.integrationEvents(t, x));
        end % CONSTRUCTOR
        
        % DESCRIPTION
        %   propogates motion from current_state with change in time of
        %   time_offset
        function [new_spatial, start_spatial, event, t, y] = updateLocation(obj, spatial, time_offset) % TODO separate control from spatial? -kezra 06/17/2021
            start_spatial = spatial.newSpatial();
            
            % Returns current state if update is smaller than tolerance
            % else integrates for length of time_offset
            if time_offset < obj.upsilon
                event = [];
                y = spatial.stateVector();
                t = 0;
                new_spatial = spatial.newSpatial();
                return
            end
            
            [~, event, t, y] = obj.rk45step(time_offset, start_spatial.stateVector(), spatial.vehicleType);
            dyfdt = obj.eom(t(end), y(end,:), spatial.vehicleType);
            new_spatial = spatial.newSpatial();
            new_spatial.updateSpatial(y(end,:), dyfdt(4), dyfdt(5), dyfdt(6));
        end % updateLocation
        
        function [new_stateVector, event, t, y] = rk45step(obj, dt, x0, vehicleType)
            [t,y,te,ye,ie] = ode45(@(t, y) obj.eom(t, y, vehicleType), ...
                [0 dt],x0,obj.odeOptions);
            
            event.time = te;
            event.state = ye;
            
            % Assigns 0 flag (i.e. no event occured during integration)
            if isempty(ie)
                ie = 0;
            end % if
            
            event.flag = ie;
            new_stateVector = y(end,:);
        end % rk45step
        
        % Computes coefficient of lift and drag
        function [cl, cd] = calcAero(obj, alpha, mach)
            cl = obj.cl_vec(2) * alpha + obj.cl_vec(3) * exp(obj.cl_vec(4) * mach) + obj.cl_vec(1);
            cd = obj.cd_vec(2) * alpha ^ 2 + obj.cd_vec(3) * exp(obj.cd_vec(4) * mach) + obj.cd_vec(1);
        end
        
        % Function used in the integrator for the time step
        function dxdt = eom(obj, ~, x, vehicleType) % (obj, t, x)
            % Parameters from atmosphere
            [temp, ~, rho] = obj.world.calcAtmosphere(x(1));
            
            [alpha, sigma] = obj.calcControl(x, vehicleType);
            
            r = obj.world.getRadius() + x(1); % radius of vehicle from world center
            
            % Calculate lift and drag
            [cl, cd] = obj.calcAero(alpha, x(4)/obj.world.calcSpeedOfSound(temp));
            lift = 0.5*rho*x(4)^2*cl*vehicleType.reference_area;
            drag = 0.5*rho*x(4)^2*cd*vehicleType.reference_area;
                       
            dxdt = [ ...
                x(4)*sin(x(5)); ... % dh/dt
                x(4)*cos(x(5))*cos(x(6))/r; ... % d_theta/dt
                x(4)*cos(x(5))*sin(x(6))/(r*cos(x(2))); ... % d_phi/dt
                -drag/vehicleType.mass - obj.world.getGravParam()*sin(x(5))/r^2; ... % dv/dt
                lift*cos(sigma)/(vehicleType.mass*x(4)) - obj.world.getGravParam()/(x(4)*r^2)*cos(x(5)) + x(4)/r*cos(x(5)); ... % d_gam/dt
                lift*sin(sigma)/(vehicleType.mass*cos(x(5))*x(4)) + x(4)/r*cos(x(5))*sin(x(6))*tan(x(2))]; % d_psi/dt
        end % eom

        function [alpha, sigma] = calcControl(~, ~, ~) % obj, stateVector, vehicleType
            % Use state and vehicle parameters to determine control input.
            % Default case is 0 control input (AoA, Bank).
            alpha = 0;
            sigma = 0;
        end
        
        function bool = hasTriggeredEvent(obj, spatial)
            eventValues = obj.odeOptions.Events([], spatial.stateVector());
            bool = max(abs(eventValues) < obj.upsilon);
        end % hasTriggeredEvent

        function bool = hasTriggeredIntegrationEvent(obj, spatial)
            eventValues = obj.integrationEvents([], spatial.stateVector());
            bool = max(abs(eventValues) < obj.upsilon);
        end % hasTriggeredIntegrationEvent

        % Events for odeXY integration.
        % value(1) terminates integration when hgv drops below min
        % altitude.
        function [value, isterminal, direction] = integrationEvents(obj,~,x) % (obj, t, x)
            value = x(1) - obj.minAltitude;
            isterminal = 1; % terminates integration
            direction = 0; % bidirectional zero
        end
    end % object methods
    
    % Setters and getters
    methods
        function setWorld(obj, world)
            assert(isa(world, 'a4h.util.AtmosphericWorld'), 'World must be AtmosphericWorld!')
            obj.world = world;
        end % setWorld
    end % setter/getter methods
    
    %%% TEST METHODS %%%
    methods (Static)
        function testRK45()
            position = [6415637,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            
            moveMan = a4h.funcs.movement.HGVMovementManager();
            moveMan.setWorld(a4h.util.AtmosphericWorld());
            spatial = a4h.util.Spatial(position, velocity, acceleration, a4h.util.HGVType(), moveMan.world);
            
            [~, ~, t, y] = moveMan.rk45step(300, spatial.stateVector(), spatial.vehicleType);
            
            figure()
            sgtitle('Hit ground event, 0 Control')
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

        function testUpdateLocation()
            clear
            close all
            %% Integrate Whole trajectory
            position = [6415637,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            maxTime = 50; % s
            moveMan = a4h.funcs.movement.HGVMovementManager();
            moveMan.setWorld(a4h.util.AtmosphericWorld());
            spatial = a4h.util.Spatial(position, velocity, acceleration, a4h.util.HGVType(), moveMan.world);
            
            % Run rk45 for whole trajectory as baseline
            [~, ~, t, y] = moveMan.rk45step(maxTime, spatial.stateVector(), spatial.vehicleType);
            
            %% Record points at each updateLocation
            time_offset = 1; % s
            currentTime = 0;
            i = 0;
            event.flag = 0;
            spatials{1} = spatial.newSpatial();
            
            while event.flag == 0 && currentTime < maxTime
                i = i + 1; % current index
                
                % Takes in spatial converted from a VehicleState object + time
                % before next MP step
                [spatial, ~, event] = moveMan.updateLocation(spatial, time_offset);
                spatials{i+1} = spatial.newSpatial(); %#ok<AGROW>
                % Update current time
                if (event.flag == 0)
                    currentTime = currentTime + time_offset;
                else
                    currentTime = currentTime + event.time;
                end % if
                
                % records state after each location update
                stateVector = spatial.stateVector();
                h(i) = stateVector(1); %#ok<AGROW>
                theta(i) = stateVector(2); %#ok<AGROW>
                phi(i) = stateVector(3); %#ok<AGROW>
                v(i) = stateVector(4); %#ok<AGROW>
                gam(i) = stateVector(5); %#ok<AGROW>
              % psi(i) = new_state.stateVector(6); %#ok<AGROW>
                times(i) = currentTime; %#ok<AGROW>
                
                % Updates the VehicleState using the new spatial after
                % integration
            end % for
            
            fprintf('Final event flag: %d\n', event.flag)
            fprintf('Final event time: %f\n', event.time)
            
            figure()
            sgtitle('Hit ground event, 0 Control')
            subplot(4,1,1)
            plot(rad2deg(y(:,2)), rad2deg(y(:,3)), 'LineWidth', 2)
            hold on
            plot(rad2deg(theta), rad2deg(phi), 'x')
            xlabel('Downrange [deg]')
            ylabel('Crossrange [deg]')
            legend('rk45step','updateLocation steps', 'Location', 'best')
            grid on
            
            subplot(4,1,2)
            plot(t, y(:,1)/1000, 'LineWidth', 2)
            hold on
            plot(times, h/1000, 'x')
            xlabel('Time [s]')
            ylabel('Altitude [km]')
            legend('rk45step','updateLocation steps', 'Location', 'best')
            grid on
            
            subplot(4,1,3)
            plot(t, rad2deg(y(:,5)), 'LineWidth', 2)
            hold on
            plot(times, rad2deg(gam), 'x')
            xlabel('Time [s]')
            ylabel('FPA [deg]')
            legend('rk45step','updateLocation steps', 'Location', 'best')
            grid on
            
            subplot(4,1,4)
            plot(t, y(:,4), 'LineWidth', 2)
            hold on
            plot(times, v, 'x')
            xlabel('Time [s]')
            ylabel('Velocity [m/s]')
            legend('rk45step','updateLocation steps', 'Location', 'best')
            grid on
            
            %% Plot the 3-D plots for the spatials
            positions = zeros(i, 3);
            velocities = zeros(i, 3);
            accelerations = zeros(i, 3);
            for j = 1:i+1
                positions(j,:) = spatials{j}.position;
                velocities(j,:) = spatials{j}.velocity;
                accelerations(j,:) = spatials{j}.acceleration;
            end % for
            
            figure()
            plot3(positions(:,1)/1000, positions(:,2)/1000, positions(:,3)/1000, 'x')
            grid on
            xlabel('x [km]')
            ylabel('y [km]')
            zlabel('z [km]')
            title('Cartesian Position')
            
            figure()
            plot3(velocities(:,1), velocities(:,2), velocities(:,3), 'x')
            grid on
            xlabel('x [m/s]')
            ylabel('y [m/s]')
            zlabel('z [m/s]')
            title('Cartesian Velocities')
            
            figure()
            plot3(accelerations(:,1), accelerations(:,2), accelerations(:,3), 'x')
            grid on
            xlabel('x [m/s^2]')
            ylabel('y [m/s^2]')
            zlabel('z [m/s^2]')
            title('Cartesian Accelerations')
        end % testUpdateLocation
    end % public tester methods
end % CLASS DEF