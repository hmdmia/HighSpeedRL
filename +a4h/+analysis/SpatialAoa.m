classdef SpatialAoa < stdlib.analysis.CoordinatedAnalyzer
    % Analyze results of hgvAoaTest runs, will be phased out once
    % SpatialGen complete
    
    properties
        data
        logger
        csvLogPath = 'tmp/hgvAoaLog.csv'
    end
    
    methods
        function obj = SpatialAoa(logger, coordinator)
            if ~exist('coordinator', 'var')
                coordinator = stdlib.analysis.Coordinator();
            end
            obj@stdlib.analysis.CoordinatedAnalyzer(coordinator);
            obj.logger = logger;
            obj.loadData();
        end % CONSTRUCTOR
        
        function loadData(obj)
            obj.data = base.sim.Loggable.readParamsByClass(obj.logger, 'a4h.agents.concrete.HGVAoa', a4h.agents.concrete.HGVAbstract.HGV_LOGGING_KEY);
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
            title('HGV State');
            
            subplot(4,1,2)
            plot(rad2deg(x(:,2)), rad2deg(x(:,3)))
            grid on
            xlabel('Downrange [deg]')
            ylabel('Crossrange [deg]')
            ytickformat('%.1f')
            
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
            exporter.importParamsByClass('a4h.agents.concrete.HGVAoa', a4h.agents.concrete.HGVAbstract.HGV_LOGGING_KEY);
            exporter.writeData(); 
        end
        
        % getters
        function t = getTimes(obj)
            t = [obj.data.HGV(:).time];
        end % getTimes
        
        function x = getState(obj)
            hgvs = obj.getHGVs();
            rWorlds = [hgvs(:).rWorld];
            [positions, velocities] = obj.getECEF();
            x = a4h.util.Spatial.ecef2stateVector(positions, velocities, rWorlds);
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
        end % getController
        
        function u = getControl(obj)
            controllers = obj.getControllers();
            assert(isfield(controllers(:), 'alpha'), ...
                'No field ''alpha'' in spatial.controller log');
            assert(isfield(controllers(:), 'sigma'), ...
                'No field ''sigma'' in spatial.controller log');
            u = [[controllers(:).alpha]', [controllers(:).sigma]'];
        end % getControl
    end % methods
end % CLASS DEF

