classdef HGVaoaMovementManager < a4h.funcs.movement.HGVMovementManager
    % Movement manager for HGV vehicle with a constant/static angle of attack
    
    % properties used to set the control 
    properties
        alpha % alpha value targeted
        sigma % sigma value targeted
    end % control properties
    
    methods
        function obj = HGVaoaMovementManager()
            obj=obj@a4h.funcs.movement.HGVMovementManager();
                                  
            % assigning values to new parameters
            obj.sigma = 0; 
        end % CONSTRUCTOR
        
        function [new_spatial, start_spatial, event, t, y, u] = updateLocation(obj, spatial, time_offset)  
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
                u = [obj.alpha, obj.sigma];
                new_spatial = spatial;
                return
            end
             
            obj.alpha = spatial.controller;
            [~, event, t, y] = obj.rk45step(time_offset, start_spatial.stateVector(), spatial.vehicleType);
            dyfdt = obj.eom(t(end), y(end,:), spatial.vehicleType);
            new_spatial = spatial.newSpatial();
            new_spatial.updateSpatial(y(end,:), dyfdt(4), dyfdt(5), dyfdt(6));
            u = [ones(length(t), 1)*obj.alpha, ones(length(t), 1)*obj.sigma];
                       
            end % updateLocation
            
            function dxdt = eom(obj, ~, x, vehicleType) % (obj, t, x)
            h = x(1);
            theta = x(2);
          % phi = x(3);
            v = x(4);
            gam = x(5);
            psi = x(6);
            
            mass = vehicleType.mass;
            mu = obj.world.getGravParam();
                                   
            % Parameters from atmosphere
            [temp, ~, rho] = obj.world.calcAtmosphere(h);
            r = obj.world.getRadius() + h; % radius of vehicle from world center
            q = 0.5*rho*v^2; % dynamic pressure
            
            % Calculate lift and drag
            [cl, cd] = obj.calcAero(obj.alpha, obj.world.calcMach(temp, v));
            lift = q*cl*vehicleType.reference_area;
            drag = q*cd*vehicleType.reference_area;
            
            vDot = -drag/mass - mu*sin(gam)/r^2;
            gamDot = lift*cos(obj.sigma)/(mass*v) - mu/(v*r^2)*cos(gam) + v/r*cos(gam);
            psiDot = lift*sin(obj.sigma)/(mass*cos(gam)*v) + v/r*cos(gam)*sin(psi)*tan(theta);
            
            dxdt = [ ...
                v*sin(gam); ... % dh/dt
                v*cos(gam)*cos(psi)/r; ... % d_theta/dt
                v*cos(gam)*sin(psi)/(r*cos(theta)); ... % d_phi/dt
                vDot; ... % dv/dt
                gamDot; ... % d_gam/dt
                psiDot]; % d_psi/dt
        end % eom
               
    end % object methods
     %%% TEST METHODS %%%
    methods (Static)
        function testRK45()
            clear
            close all
            
            position = [6415637,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            alpha = 0; % deg

            moveMan = a4h.funcs.movement.HGVaoaMovementManager();
            moveMan.setWorld(a4h.util.AtmosphericWorld());
            spatial = a4h.util.Spatial(position, velocity, acceleration, a4h.util.HGVType(), moveMan.world.getRadius());
            spatial.controller = alpha;
            moveMan.alpha = spatial.controller;
            
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
        
        % Tests the aoa case, outputting plots
        function testAoa()
            clear
            close all
            
            world = a4h.util.AtmosphericWorld();
            position = [world.getRadius() + 40000,0,0];  % m
            velocity = [-334.491,3182.47,0];  % m/s
            acceleration = [0, 0, 0]; % m/s2
            alpha = .25; % rad
            
            moveMan = a4h.funcs.movement.HGVaoaMovementManager();
            spatial.controller = alpha;
            moveMan.alpha = spatial.controller;
            spatial = a4h.util.Spatial(position, velocity, acceleration, a4h.util.HGVType(), world.getRadius,moveMan.alpha);
            moveMan.setWorld(world);           

            time_offset = 100; % s
            
            [~, ~, event,t, y, u] = moveMan.updateLocation(spatial, time_offset);
                                    
            fprintf('Flag: %d\n', event.flag)
            fprintf('Event time: %f\n', event.time)
            
            a4h.funcs.movement.HGVaoaMovementManager.graphAngles(t, y, u, sprintf('AOA to %.2f%c', rad2deg(alpha), char(176)));
            a4h.funcs.movement.HGVaoaMovementManager.graphStates(t, y, sprintf('AOA to %.2f%c', rad2deg(alpha), char(176)));
        end % testaoa
        
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
        
    end % static tester methods
end % CLASS DEF

