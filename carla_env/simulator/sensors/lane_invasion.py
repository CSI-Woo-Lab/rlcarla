from typing import Set, cast

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator


class LaneInvasionSensor(Sensor):
    def __init__(self, simulator: Simulator):
        blueprint_library = simulator.world.blueprint_library
        blueprint = blueprint_library.find("sensor.other.lane_invasion")
        super().__init__(simulator, blueprint)

        self.__lane_types: Set[carla.LaneMarkingType] = set()

    @override
    def spawn(self, parent: Actor):
        super().spawn(carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.0),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        ), parent)
        self.listen(self._callback__on_invasion)

    def reset(self):
        self.__lane_types.clear()

    @override
    def destroy(self) -> None:
        self.reset()
        super().destroy()

    def _callback__on_invasion(self, data: carla.SensorData):
        event = cast(carla.LaneInvasionEvent, data)
        self.__lane_types = set(marking.type for marking in event.crossed_lane_markings)

    @property
    def lane_types(self) -> Set[carla.LaneMarkingType]:
        return self.__lane_types
