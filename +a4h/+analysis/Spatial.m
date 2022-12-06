classdef Spatial < stdlib.analysis.CoordinatedAnalyzer
    % Analyze results of hgvmpTest runs
    
    properties
        data
        logger
        csvLogPath = 'tmp/hgvmpLog.csv'
    end
    
    methods
        function obj = Spatial(logger, coordinator)
            if ~exist('coordinator', 'var')
                coordinator = stdlib.analysis.Coordinator();
            end
            obj@stdlib.analysis.CoordinatedAnalyzer(coordinator);
            obj.logger                = logger;
            obj.loadData();
        end % CONSTRUCTOR
        
        function loadData(obj)
            obj.data = base.sim.Loggable.readParamsByClass(obj.logger, 'a4h.agents.concrete.HGVmp', a4h.agents.concrete.HGVAbstract.HGV_LOGGING_KEY);
        end % loadDate
        
        function plotState(obj)
            x = obj.getState();
            t = obj.getTimes();
            
            figure()
            subplot(4,1,1)
            plot(t,x(:,1)/1000)
            grid on
            xlabel('Time [s]')
            ylabel('Altitude [km]')
            
            subplot(4,1,2)
            plot(rad2deg(x(:,2)), rad2deg(x(:,3)))
            grid on
            xlabel('Downrange [deg]')
            ylabel('Crossrange [deg]')
            
            subplot(4,1,3)
            plot(t, x(:,4))
            grid on
            xlabel('Time [s]')
            ylabel('Velocity [m/s]')
            
            subplot(4,1,4)
            plot(t, rad2deg(x(:,5:6)))
            grid on
            xlabel('Time [s]')
            legend('\gamma [deg]', '\psi [deg]', 'Location', 'best')
        end % plotState
        
        function plotPrimitives(obj)
            t = obj.getTimes();
            x = obj.getState();
            prims = obj.getPrimitives();
            
            assert(length(t) <= length(prims), sprintf('Incomplete spatial.controller log: %d primitive(s) not recorded', length(t) - length(prims)))
            assert(length(t) >= length(prims), sprintf('Incomplete t log: %d time(s) not recorded', length(prims) - length(t)))
                
            figure()
            if isvector(prims)
                stairs(t, prims)
                ylabel('Primitives')
                grid on
                xlabel('Time [s]')
            else
                subplot(2,1,1)
                stairs(t, rad2deg(prims(1,:)), '--')
                hold on
                plot(t, rad2deg(x(:,5)))
                hold off
                grid on
                xlabel('Time [s]')
                legend('\gamma_d [deg]', '\gamma [deg]', 'Location', 'best')

                subplot(2,1,2)
                stairs(t, rad2deg(prims(2,:)), '--')
                hold on
                plot(t, rad2deg(x(:,6)))
                hold off
                grid on
                xlabel('Time [s]')
                legend('\psi_d [deg]', '\psi [deg]', 'Location', 'best')
            end % if/else
        end % plotPrimitive
        
        function plotControl(obj)
            figure()
            plot(obj.getTimes(), rad2deg(obj.getControl()))
            grid on
            xlabel('Time [s]')
            ylabel('Control [deg]')
            legend('\alpha', '\sigma', 'Location', 'best')
        end % plotControl
        
        function writeCSV(obj)
            exporter = base.export.LogExportTool(obj.logger.rootPath,obj.csvLogPath);
            exporter.importParamsByClass('a4h.agents.concrete.HGVmp', a4h.agents.concrete.HGVAbstract.HGV_LOGGING_KEY);
            exporter.writeData(); 
        end % writeCSV

        function writeStateCSV(obj)
            state = obj.getState();
            t = obj.getTimes();
            u = obj.getControl;
            prim = obj.getPrimitives();
            csv_data = [t', state, u, prim'];
            writematrix(csv_data, obj.csvLogPath);
        end % writeStateCSV
        
        % getters
        function t = getTimes(obj)
            t = [obj.data.HGV(:).time];
        end % getTimes
        
        function x = getState(obj)
            hgvs = obj.getHGVs();
            worlds = [hgvs(:).world];
            [positions, velocities] = obj.getECEF();
            x = a4h.util.Spatial.ecef2stateVector(positions, velocities, worlds);
        end % getState
        
        function [positions, velocities, accelerations] = getECEF(obj)
            hgvs = obj.getHGVs();
            numHGVS = length(hgvs);
            positions = zeros(numHGVS,3);
            velocities = zeros(numHGVS,3);
            accelerations = zeros(numHGVS,3);
            
            for i = 1:numHGVS
                positions(i,:) = hgvs(i).position;
                velocities(i,:) = hgvs(i).velocity;
                accelerations(i,:) = hgvs(i).acceleration;
            end
        end % getECEF
        
        function hgvs = getHGVs(obj)
            hgvs = [obj.data.HGV(:).value];
        end % getHGVs
        
        function controllers = getControllers(obj)
            hgvs = obj.getHGVs();
            controllers = [hgvs(:).controller];
        end % getControllers
        
        function primitives = getPrimitives(obj)
            controllers = obj.getControllers();
            assert(isfield(controllers(:), 'primitive'), ...
                'No field ''primitive'' in spatial.controller log');
            primitives = [controllers(:).primitive];
        end % getPrimitives
        
        function u = getControl(obj)
            controllers = obj.getControllers();
            assert(isfield(controllers(:), 'alpha'), ...
                'No field ''alpha'' in spatial.controller log');
            assert(isfield(controllers(:), 'sigma'), ...
                'No field ''sigma'' in spatial.controller log');
            % Replace empty with NaN so vector is equal length
            for i = 1:length(controllers)
                if isempty(controllers(i).alpha)
                    controllers(i).alpha = NaN;
                end % if
                if isempty(controllers(i).sigma)
                    controllers(i).sigma = NaN;
                end % if
            end % for

            u = [[controllers(:).alpha]', [controllers(:).sigma]'];
        end % getControl
    end % methods
end % CLASS DEF

