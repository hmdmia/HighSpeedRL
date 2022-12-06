classdef HGVType < a4h.util.VehicleType
    % HGV vehicle type with appropriate properties

    methods
        function obj = HGVType()
            name = 'hgv';
            mass = 907.20; % [kg]
            reference_area = 0.4839; % [m2]
            obj@a4h.util.VehicleType(name, mass, reference_area)
        end % CONSTRUCTOR
    end % methods
end % CLASS DEF

