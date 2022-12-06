classdef VehicleType < handle
    % Abstract class 
    
    properties
        name
        mass
        reference_area
    end
    
    methods
        function obj = VehicleType(name, mass, reference_area)
            obj.name = name;
            obj.mass = mass;
            obj.reference_area = reference_area;
        end % CONSTRUCTOR
    end % methods
end % CLASS DEF

