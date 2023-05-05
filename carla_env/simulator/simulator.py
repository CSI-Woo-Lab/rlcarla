import asyncio
import time
from threading import Event, Thread
from typing import List, Optional

import carla

from configs.config import ExperimentConfigs


def tick(world, fps, event):
    while not event.is_set():
        world.tick()
        time.sleep(1 / fps)


class Simulator:
    """The simulator of the environment. This class is responsible for creating the
    client and the world of the simulator.
    
    Args:
        config (ExperimentConfigs): The experiment configurations.

    """

    def __init__(self, config: ExperimentConfigs):
        from carla_env.simulator.client import Client
        from carla_env.simulator.route_manager import RouteManager
        from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle
        from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle

        self.__config = config

        self.__client = Client(self, config.carla_ip, 2000 - config.num_routes * 5)

        self.__world = self.__client.world
        self.__world.weather = getattr(carla.WeatherParameters, config.weather)

        self.__route_manager = RouteManager(world=self.__world, config=config)

        self.__is_multi_agent = config.multiagent
        self.__num_auto_vehicles = config.num_vehicles
        self.__fps = config.fps

        self.__ego_vehicle: Optional[EgoVehicle] = None
        self.__auto_vehicles: Optional[List[AutoVehicle]] = None

    async def reset(self):
        from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle
        from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle

        event = Event()
        tick_thread = Thread(target=tick, args=(self.__world, self.__fps, event))
        tick_thread.start()

        # Destroy the auto vehicles.
        if self.__auto_vehicles:
            await asyncio.gather(*(
                auto_vehicle.destroy()
                for auto_vehicle in self.__auto_vehicles
            ))

        # Spawn the ego vehicle.
        self.__route_manager.select_route()
        if not self.__ego_vehicle:
            self.__ego_vehicle = await EgoVehicle.spawn(
                simulator=self,
                config=self.__config,
                initial_transform=self.route_manager.initial_transform,
            )
        else:
            self.__ego_vehicle.transform = self.route_manager.initial_transform
            self.__ego_vehicle.reset()

        # Spawn the auto vehicles.
        if self.is_multi_agent:
            self.__auto_vehicles = await asyncio.gather(*(
                AutoVehicle.spawn(simulator=self)
                for _ in range(self.__num_auto_vehicles)
            ))

        event.set()
        tick_thread.join()

    async def finish(self):
        """Finish the simulator."""
        if self.__ego_vehicle and self.__ego_vehicle.is_alive:
            await self.__ego_vehicle.destroy()

        if self.__auto_vehicles:
            await asyncio.gather(*(
                auto_vehicle.destroy()
                for auto_vehicle in self.__auto_vehicles
                if auto_vehicle.is_alive
            ))

    @property
    def client(self):
        """The client of the simulator."""
        return self.__client

    @property
    def world(self):
        """The world of the simulator."""
        return self.__world

    @property
    def route_manager(self):
        """The route manager of the simulator."""
        return self.__route_manager

    @property
    def ego_vehicle(self):
        """The ego vehicle of the simulator."""
        if not self.__ego_vehicle:
            raise ValueError("Ego vehicle is not initialized. Call reset() first.")
        return self.__ego_vehicle

    @property
    def vehicle_location(self):
        """The location of the ego vehicle."""
        return self.ego_vehicle.location

    @property
    def target_location(self):
        """The target location of the ego vehicle."""
        return self.route_manager.target_transform.location

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
        if self.is_multi_agent and not self.__auto_vehicles:
            raise ValueError("Auto vehicles are not initialized. Call reset() first.")
        return self.__auto_vehicles

    @property
    def fps(self):
        """The fps of the simulator."""
        return self.__fps
