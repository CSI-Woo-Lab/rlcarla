import math
from dataclasses import dataclass
from typing import List, cast

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator


@dataclass
class CollisionEvent:
    """Collision event data class."""

    actor_id: int
    """The id of the actor that collided with the ego vehicle."""

    frame: int
    """The frame in which the collision occurred."""

    intensity: float
    """The intensity of the collision."""


class CollisionSensor(Sensor):
    def init(self, max_queue: int = 4000):
        self.__collided = False
        self.__collision_history: List[CollisionEvent] = []
        self.__max_queue = max_queue

        self.listen(self._callback__on_collision)

    @classmethod
    @override
    async def spawn(
        cls,
        simulator: Simulator,
        parent: Actor,
        max_queue: int = 4000,
    ):
        blueprint_library = simulator.world.blueprint_library
        blueprint = blueprint_library.find("sensor.other.collision")
        
        sensor = await super().spawn(
            simulator=simulator,
            blueprint=blueprint,
            attach_to=parent,
        )
        if not sensor:
            return None

        sensor.init(max_queue=max_queue)
        return sensor

    def _callback__on_collision(self, data: carla.SensorData):
        event = cast(carla.CollisionEvent, data)
        self.__collided = True

        if len(self.__collision_history) >= self.__max_queue:
            self.__collision_history.pop(0)

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.__collision_history.append(
            CollisionEvent(
                actor_id=event.other_actor.id,
                frame=event.frame,
                intensity=intensity,
            )
        )

    def reset(self) -> None:
        self.__collided = False
        self.__collision_history.clear()

    @property
    def has_collided(self) -> bool:
        """Whether the ego vehicle collided with other vehicles or obstacles."""
        return self.__collided

    @property
    def object(self) -> Actor:
        """The object that the ego vehicle collided with."""
        return self.simulator.world.get_actor(self.__collision_history[-1].actor_id)

    @property
    def objects(self) -> List[Actor]:
        """The objects that the ego vehicle collided with."""
        ids = set(history.actor_id for history in self.__collision_history)
        return self.simulator.world.get_actors(ids)
