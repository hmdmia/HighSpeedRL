classdef SpatialMP < a4h.util.Spatial
    % Extension of Spatial class for methods using mp controller
    
    properties
        ctrlTemplate = struct('primitive', [], 'alpha', [], 'sigma', [])
    end % constant properties
    
    methods
        function obj = SpatialMP(position, velocity, acceleration, vehicleType, world, varargin)
            obj@a4h.util.Spatial(position, velocity, acceleration, vehicleType, world);
            
            % Sets controller fields if no controller given
            if isempty(varargin)
                obj.controller = obj.ctrlTemplate;
            else
                obj.controller = varargin{1};
            end % if
        end % CONSTRUCTOR
        
        % overrides newSpatial to make a new MP spatial
        function spatial = newSpatial(obj)
            spatial = a4h.util.SpatialMP(obj.position, obj.velocity, obj.acceleration, obj.vehicleType, obj.world, obj.controller);
        end % newSpatial
        
        function primitive = getPrimitive(obj)
            primitive = obj.controller.primitive;
        end % getPrimitive
        
        function setPrimitive(obj, primitive)
            obj.controller.primitive = primitive;
        end % setPrimitive
        
        function u = getControl(obj)
            u = [obj.controller.alpha, obj.controller.sigma];
        end % getControl
        
        function setControl(obj, u)
            obj.controller.alpha = u(1);
            obj.controller.sigma = u(2);
        end % setControl
    end % methods

    methods (Static)
        function spatial = stateVector2spatial(stateVector, vehicleType, world, varargin)
            [position, velocity] = a4h.util.Spatial.stateVector2ecef(stateVector, world);
            acceleration = [];
            spatial = a4h.util.SpatialMP(position, velocity, acceleration, vehicleType, world, varargin{:});
        end % stateVector2spatial
    end % Static methods
end % CLASS DEF