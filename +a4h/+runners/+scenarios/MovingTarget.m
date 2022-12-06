classdef MovingTarget < a4h.util.Scenario
    
    methods
        function obj = MovingTarget(logPath)
            startSec = 0;
            endSec = 700;
            obj.init(logPath, startSec, endSec);
        end % CONSTRUCTOR
        
        function build(obj) %, predState, preyState)
            import base.build.codeBased.*;

            % LINKS _______________________________________________
            NOTNET = obj.NOTNET;

            % LOCATIONS ___________________________________________

            %                      (Type, Lat, Lon, Alt)
            hgvStart    = Location(Location.ABSOLUTE, 35.1, -106.6, 4e4);
            targetStart = Location(Location.ABSOLUTE, 40.4, -86.9, 2e4);
            
            % GROUP TYPES _________________________________________

            %                   Relationship               GLink   Agent Class (1st = center agent)
            target = GroupType(GroupType.NO_RELATIONSHIP);
            target.addCentralUnit('a4h.agents.FixedAltitudeTarget');

            hgv    = GroupType(GroupType.NO_RELATIONSHIP);
            hgv.addCentralUnit('a4h.agents.concrete.HGVmp');

            % GROUPS ______________________________________________
            H1 = Group('Predator' ,hgv);
            T1 = Group('Prey' ,target);

            % SCENARIO ______________________________________________

            % Group    UpGrp Uplink Start Finish
            obj.add(H1, [], NOTNET, hgvStart, hgvStart);
            obj.add(T1 ,[] ,NOTNET, targetStart, hgvStart);
        end % build
    end % methods
end % CLASS DEF

