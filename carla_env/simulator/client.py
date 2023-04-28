from queue import Queue
from typing import Callable, List, Optional

import carla

from carla_env.simulator.command import CarlaCommand
from carla_env.simulator.simulator import Simulator
from carla_env.simulator.traffic_manager import TrafficManager
from carla_env.simulator.world import World


class Client(carla.Client):
    """The client of the simulator. This class is a wrapper of the carla.Client class.
    
    Args:
        host (str): The host of the simulator.
        
        port (int): The port of the simulator.
        
    """

    def __init__(self, simulator: Simulator, host: str, port: int) -> None:
        super().__init__(host, port)
        self.__simulator = simulator
        self.__command_queue: Queue[CarlaCommand] = Queue()
        self.__callback_queue: Queue[
            Optional[Callable[[carla.command.Response], None]]
        ] = Queue()

        self.set_timeout(10.)
        self.world.on_tick(self.__apply_commands)

    def enqueue_command(
        self,
        command: CarlaCommand,
        callback: Optional[Callable[[carla.command.Response], None]] = None
    ) -> None:
        self.__command_queue.put(command)
        self.__callback_queue.put(callback)

    def __apply_commands(self, _):
        if self.__command_queue.empty():
            return

        commands: List[CarlaCommand] = []
        callbacks: List[Optional[Callable[[carla.command.Response], None]]] = []

        while not self.__command_queue.empty():
            commands.append(self.__command_queue.get())
            callbacks.append(self.__callback_queue.get())

        print(f"Applying {len(commands)} commands...")

        for res, callback in zip(self.apply_batch_sync(commands), callbacks):
            if callback is not None:
                callback(res)

    def get_world(self) -> World:
        return World(super().get_world(), self.__simulator)

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
