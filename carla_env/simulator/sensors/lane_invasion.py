from typing import Set, cast

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator


class LaneInvasionSensor(Sensor):
    def init(self):
        self.__lane_types: Set[carla.LaneMarkingType] = set()
        self.listen(self._callback__on_invasion)

    @classmethod
    @override
    async def spawn(
        cls,
        simulator: Simulator,
        parent: Actor,
    ):
        blueprint_library = simulator.world.blueprint_library
        blueprint = blueprint_library.find("sensor.other.lane_invasion")
        
        sensor = await super().spawn(
            simulator, blueprint, attach_to=parent
        )
        if not sensor:
            return None

        sensor.init()
        return sensor

    def reset(self):
        self.__lane_types.clear()

    @override
    async def destroy(self) -> None:
        self.reset()
        await super().destroy()

    def _callback__on_invasion(self, data: carla.SensorData):
        event = cast(carla.LaneInvasionEvent, data)
        self.__lane_types = set(marking.type for marking in event.crossed_lane_markings)

    @property
    def lane_types(self) -> Set[carla.LaneMarkingType]:
        return self.__lane_types
