import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator
from utils.config import ExperimentConfigs


class CollisionSensor(Sensor):
    def __init__(self, simulator: Simulator, config: ExperimentConfigs):
        blueprint_library = simulator.world.get_blueprint_library()
        blueprint = blueprint_library.find("sensor.other.collision")
        super().__init__(simulator=simulator, blueprint=blueprint)

        self.__collided = False

    @override
    def spawn(self, parent: Actor):
        super().spawn(
            transform=carla.Transform(
                carla.Location(x=2.5, y=0.7, z=0.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
            ),
            parent=parent,
        )

    @override
    def _on_spawn(self, response: carla.command.Response) -> None:
        super()._on_spawn(response)
        self.actor.listen(self._callback__on_collision)

    def _callback__on_collision(self, _: carla.SensorData):
        self.__collided = True

    @property
    def is_collided(self) -> bool:
        """Whether the ego vehicle collided with other vehicles or obstacles."""
        return self.__collided
