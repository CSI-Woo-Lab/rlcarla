import carla
from typing_extensions import override

from carla_env.simulator.spectator import Spectator
from carla_env.simulator.vehicles.vehicle import Vehicle
from utils.logger import Logging

logger = Logging.get_logger(__name__)


class World(carla.World):
    """The world of the simulator. This class is a wrapper of the carla.World class.
    
    Args:
        world (carla.World): The world of the simulator.

    """

    def __new__(cls, world: carla.World) -> carla.World:
        return world

    def __init__(self, world: carla.World) -> None:
        self.__world = world
        self.__map = world.get_map()
        self.__weather = world.get_weather()
        self.__blueprint_library = world.get_blueprint_library()

        self.__world.tick()
        self.__removing_old_actors()

    def __removing_old_actors(self):
        actors = self.__world.get_actors()
        for vehicle in actors.filter("*vehicle*"):
            logger.warn(f"Destroying old vehicle {vehicle.id}.")
            vehicle.destroy()
        for sensor in actors.filter("*sensor*"):
            logger.warn(f"Destroying old sensor {sensor.id}.")
            sensor.destroy()

    @override
    def get_spectator(self) -> Spectator:
        return Spectator(super().get_spectator())

    @property
    def map(self):
        """The map of the world."""
        return self.__map

    @property
    def weather(self):
        """The weather of the world."""
        return self.__weather

    @property
    def blueprint_library(self):
        """The blueprint library of the world."""
        return self.__blueprint_library
