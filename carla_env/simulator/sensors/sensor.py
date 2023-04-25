import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor


class Sensor(Actor[carla.Sensor]):
    @override
    def spawn(self, transform: carla.Transform, parent: Actor):
        return super().spawn(transform=transform, attach_to=parent)
