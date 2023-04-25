from carla_env.simulator.client import Client
from carla_env.simulator.route_manager import RouteManager
from carla_env.simulator.vehicles.auto_vehicle_manager import \
    AutoVehicleManager
from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle
from utils.config import ExperimentConfigs


class Simulator:
    """The simulator of the environment. This class is responsible for creating the
    client and the world of the simulator.
    
    Args:
        config (ExperimentConfigs): The experiment configurations.

    """

    def __init__(self, config: ExperimentConfigs):
        self.__client = Client(config.carla_ip, 2000 - config.num_routes * 5)
        self.__world = self.__client.get_world()

        self.__route_manager = RouteManager(world=self.__world, config=config)
        self.__ego_vehicle = EgoVehicle(simulator=self, config=config)
        self.__ego_vehicle.spawn()

        self.__is_multi_agent = config.multiagent
        self.__num_auto_vehicles = config.num_vehicles
        if self.is_multi_agent:
            self.__auto_vehicles = AutoVehicleManager(
                num_vehicles=self.__num_auto_vehicles,
                simulator=self,
            )
        else:
            self.__auto_vehicles = None

    def reset(self):
        self.__route_manager.select_route()

        self.ego_vehicle.destroy()
        self.ego_vehicle.spawn(self.__route_manager.initial_transform)

        if self.auto_vehicles is not None:
            self.auto_vehicles.destroy()
            self.auto_vehicles.spawn()
            self.client.traffic_manager.reset()

    @property
    def client(self):
        """The client of the simulator."""
        return self.__client

    @property
    def world(self):
        """The world of the simulator."""
        return self.__world

    @property
    def ego_vehicle(self):
        """The ego vehicle of the simulator."""
        return self.__ego_vehicle

    @property
    def is_multi_agent(self):
        """Whether the simulator is multi-agent."""
        return self.__is_multi_agent

    @property
    def num_auto_vehicles(self):
        """The number of vehicles. If the simulator is not multi-agent, this value is
        0."""
        return self.__num_auto_vehicles if self.is_multi_agent else 0

    @property
    def auto_vehicles(self):
        """The auto vehicles of the simulator."""
        if not self.is_multi_agent:
            raise ValueError("Multi-agent is not enabled.")
        return self.__auto_vehicles
