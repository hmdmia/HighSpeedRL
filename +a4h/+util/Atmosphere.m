classdef Atmosphere < handle
    % a standard atmosphere with functions to calculate relevent
    % properties to an aeronautical vehicle
    
    properties
        gamma % ratio of specific heats
        gas_constant % [J/kg-K] R
        g0 % [m/s2] world acceleration due to gravity
        r % [m] world radius
        mu % [m3/s2] world graviational parameter
        
        num_layers % number of layers in atmosphere model
        alt_layers % [m] base altitude vector of each layer
        lam_layers % vector of lambdas for standard atmosphere
        temp_layers % vector of base temperatures [K]
        pres_layers % vector of base pressures [Pa]
        dens_layers % vector of base densities [kg/m3]
    end
    
    methods
        function obj = Atmosphere()
            % By default constructs Earth atmosphere
            obj.setEarthAtmosphere();
        end % CONSTRUCTOR
        
        function setEarthAtmosphere(obj)
            obj.gamma = 1.4;
            obj.gas_constant = 287.053; % [J/kg-K] R
            obj.g0 = 9.80665; % [m/s2] Earth acceleration due to gravity
            obj.r = 6.356766e6; % [m] Earth radius
            obj.mu = 3.986e14; % [m3/s2] Earth graviational parameter
            
            obj.alt_layers = [0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 84.8520e3];
            obj.num_layers = length(obj.alt_layers);
            obj.lam_layers = [-6.5e-3, 0, 1e-3, 2.8e-3, 0, -2.8e-3, -2e-3, 0];
            
            % Initialize base values at each layer with values in 1st layer
            obj.temp_layers = ones(1,obj.num_layers)*288.16;
            obj.pres_layers = ones(1,obj.num_layers)*101325;
            obj.dens_layers = ones(1,obj.num_layers)*1.22500;
            
            % Calculate subsequent base values
            for i = 2:obj.num_layers
                [obj.temp_layers(i), obj.pres_layers(i), obj.dens_layers(i)] = ...
                    obj.calcLayer(obj.alt_layers(i), i-1);
            end
            
        end % setEarthAtmosphere
        
        % Calculates temperature, pressure, and density given
        % geopotential altitude hg
        function [temperature, pressure, density] = calcAtmosphere(obj, hg)
            h = max(0, obj.hg2h(hg)); % If input is negative,
                                      % calculates sea-level properties
            
            % Finds index in layers vector corresponding to altitude
            % layerIndex = find(obj.alt_layers <= h, 1, 'last');
            layerIndex = ones(length(h),1);
            for i = 2:length(obj.alt_layers)
                layerIndex = layerIndex + (hg > obj.alt_layers(i));
            end % for
            
            [temperature, pressure, density] = obj.calcLayer( ...
                h, layerIndex);
        end
        
        function a = calcSpeedOfSound(obj, temperature)
            a = sqrt(obj.gamma*obj.gas_constant*temperature);
        end % speedOfSound
        
        function m = calcMach(obj, temperature, velocity)
            m = velocity ./ obj.calcSpeedOfSound(temperature);
        end % mach
        
        % Calculates temperature, pressure, and density given the altitude
        % h and the lam and *0 values for a particular layer
        function [temp, p, rho] = calcLayer(obj, h, layerIndex)
            h0 = obj.alt_layers(layerIndex)';
            lam = obj.lam_layers(layerIndex)';
            temp0 = obj.temp_layers(layerIndex)';
            p0 = obj.pres_layers(layerIndex)';
            rho0 = obj.dens_layers(layerIndex)';
            
            % gradient layer (1) or isothermal layer (0)
            gradient = abs(lam) > 1e-5;
            
            temp = temp0 + gradient .* lam .* (h - h0);
            k = exp(-obj.g0 ./ (obj.gas_constant * temp0) .* (h - h0));
            p = ~gradient .* p0 .* k...
                + gradient .* p0 .* (temp./temp0).^(-obj.g0./(lam*obj.gas_constant));
            rho = ~gradient .* rho0 .* k...
                + gradient .* rho0.*(temp./temp0).^(-1 - obj.g0./(lam * obj.gas_constant));
            
%             % Isothermal layer
%             if abs(lam) < 1e-5
%                 temp = temp0;
%                 k = exp(-obj.g0 / (obj.gas_constant * temp0) * (h - h0));
%                 p = p0 * k;
%                 rho = rho0 * k;
%                 
%             % Gradient layer
%             else
%                 temp = temp0 + lam * (h - h0);
%                 p = p0 * (temp/temp0)^(-obj.g0/(lam*obj.gas_constant));
%                 rho = rho0*(temp/temp0)^(-1 - obj.g0/(lam * obj.gas_constant));
%             end
        end % calcLayer
        
        % Converts geopotential altitude to geometric altitude
        function h = hg2h(obj, hg)
            h = hg * obj.r./(obj.r + hg);
        end
    end % methods
    
    %%% TEST METHODS %%%
    methods (Static)
        function testPlotAtmosphere()
            clear
            close all
            atm = a4h.util.Atmosphere();
            values = 500;
            h_min = 0;
            h_max = 80e3;
            h = linspace(h_min, h_max, values)';
%             temp = zeros(values,1);
%             pres = zeros(values,1);
%             dens = zeros(values,1);
            
            [temp, pres, dens] = atm.calcAtmosphere(h);
            
%             for i = 1:values
%                 [temp(i), pres(i), dens(i)] = atm.calcAtmosphere(h(i));
%             end
            
            figure()
            
            % Temperature plot
            subplot(1,3,1)
            plot(temp, h/1000, 'LineWidth', 2)
            grid on
            xlabel('T [K]')
            ylabel('Altitude [km]')
            
            % Pressure plot
            subplot(1,3,2)
            plot(pres/1000, h/1000, 'LineWidth', 2)
            grid on
            xlabel('P [kPa]')
            ylabel('Altitude [km]')
            
            subplot(1,3,3)
            plot(dens, h/1000, 'LineWidth', 2)
            grid on
            xlabel('\rho [kg/m3]')
            ylabel('Altitude [km]')
        end % testPlotAtmosphere
    end % test static methods
end % CLASS DEF

