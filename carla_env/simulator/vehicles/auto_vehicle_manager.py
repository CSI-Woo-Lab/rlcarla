from typing import Iterator, Sequence

from typing_extensions import override

from carla_env.simulator.simulator import Simulator
from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle


class AutoVehicleManager(Sequence[AutoVehicle]):
    def __init__(self, simulator: Simulator, num_vehicles: int):
        self.__vehicles = tuple(
            AutoVehicle(simulator=simulator) for _ in range(num_vehicles)
        )

    def destroy(self) -> None:
        for vehicle in self.__vehicles:
            vehicle.destroy()

    def spawn(self):
        for vehicle in self.__vehicles:
            vehicle.spawn()

    @override
    def __len__(self) -> int:
        return len(self.__vehicles)

    @override
    def __getitem__(self, index: int) -> AutoVehicle:
        return self.__vehicles[index]
