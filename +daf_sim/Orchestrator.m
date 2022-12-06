classdef Orchestrator < handle

    properties
        simInst
        simObservableManager
        simNetwork
        simDataService
        simTwoWayConnectFunc
    end

    properties
        simEarth
        startTime = 0
        endTime
    end

    properties (SetAccess=private)
        lastUniqueNetworkId = 10000
    end

    methods
        function obj = Orchestrator()
        end

        function setEarth(obj,earth)
            obj.simEarth = earth;
        end

        function setStartTime(obj, startTime)
            obj.startTime = startTime;
        end

        function setEndTime(obj, endTime)
            obj.endTime = endTime;
        end

        function obj = init(obj, logPath)
            obj.simInst = base.sim.Instance(logPath);
            observableManager = base.funcs.groups.ObjectManager(obj.startTime);
            obj.simObservableManager = observableManager;
            network = base.funcs.comms.Network();
            obj.simNetwork = network;
            obj.simInst.AddCallee(obj.simNetwork);
            dataService = base.funcs.comms.DataService();
            obj.simDataService = dataService;
            twoWayConnectFunc = @(source,dest,bandwidth,latency,linkType) obj.twoWayConnect(source,dest,bandwidth,latency,linkType,obj.simNetwork,obj.simDataService);
            obj.simTwoWayConnectFunc = twoWayConnectFunc;
        end

        function fastConnect(obj, uplinkAgent, downlinkAgent)

            assert(isa(uplinkAgent,'base.agents.Networked') && ...
                isa(downlinkAgent,'base.agents.Networked'),...
                ['Agent ' uplinkAgent.commonName ' or ' downlinkAgent.commonName ' not Networked agent!']);

            downlinkAgent.addUpstreamNetworkId(uplinkAgent.getLocalNetworkId);
            uplinkAgent.addDownstreamNetworkId(downlinkAgent.getLocalNetworkId);

            obj.simTwoWayConnectFunc(uplinkAgent,downlinkAgent,inf,0,'FAST');

        end

        function addAgent(obj, agent, name)
            assert(~isempty(obj.simInst),'Orchestrator is not yet initialized!');

            obj.simInst.AddCallee(agent);

            agent.setCommonName(name);

            if isa(agent,'base.agents.Networked')
                agent.setNetworkName(name);
                agent.setLocalNetworkId(obj.lastUniqueNetworkId);
                obj.lastUniqueNetworkId = obj.lastUniqueNetworkId+1;
            end
        end

        function twoWayConnect(obj,source,dest,bandwidth,latency,linkType,network,dataService)
            if isempty(source.clientSwitch)
                sourceSwitch = network.createSwitch();
                obj.simInst.AddCallee(sourceSwitch);
                source.addToNetwork(sourceSwitch,dataService);
            end

            if isempty(dest.clientSwitch)
                destSwitch = network.createSwitch();
                obj.simInst.AddCallee(destSwitch);
                dest.addToNetwork(destSwitch,dataService);
            end

            obj.simNetwork.createP2PLink(source.clientSwitch,dest.clientSwitch,bandwidth,latency,linkType);
            obj.simNetwork.createP2PLink(dest.clientSwitch,source.clientSwitch,bandwidth,latency,linkType);
        end

        function fini(obj)
            obj.simNetwork.updateNextHopList();
        end

        function run(obj)
            assert(~isempty(obj.endTime), 'End time has not been set!');
            obj.simInst.runUntil(obj.startTime,obj.endTime);
        end
    end
end
