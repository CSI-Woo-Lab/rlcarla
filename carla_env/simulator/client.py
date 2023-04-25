from typing import Callable, List, Optional

import carla

from carla_env.simulator.command import CarlaCommand
from carla_env.simulator.traffic_manager import TrafficManager
from carla_env.simulator.world import World


class Client(carla.Client):
    """The client of the simulator. This class is a wrapper of the carla.Client class.
    
    Args:
        host (str): The host of the simulator.
        
        port (int): The port of the simulator.
        
    """

    def __init__(self, host: str, port: int) -> None:
        super().__init__(host, port)
        self.__command_queue: List[CarlaCommand] = []
        self.__callback_queue: List[
            Optional[Callable[[carla.command.Response], None]]
        ] = []

        self.set_timeout(10.)
        self.world.on_tick(self.__apply_commands)

    def enqueue_command(
        self,
        command: CarlaCommand,
        callback: Optional[Callable[[carla.command.Response], None]] = None
    ) -> None:
        self.__command_queue.append(command)
        self.__callback_queue.append(callback)

    def __apply_commands(self, _):
        if len(self.__command_queue) > 0:
            for res, callback in zip(
                self.apply_batch_sync(self.__command_queue), self.__callback_queue
            ):
                if callback is not None:
                    callback(res)
            self.__command_queue.clear()
            self.__callback_queue.clear()

    def get_world(self) -> World:
        return World(super().get_world())

    def get_trafficmanager(self, client_connection: int = 8000) -> TrafficManager:
        return TrafficManager(super().get_trafficmanager(client_connection))

    @property
    def world(self) -> World:
        """The world of the simulator."""
        return self.get_world()

    @property
    def traffic_manager(self) -> TrafficManager:
        """The traffic manager of the simulator."""
        return self.get_trafficmanager()
