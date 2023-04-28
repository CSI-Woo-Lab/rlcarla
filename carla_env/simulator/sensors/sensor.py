from typing import Callable

import carla

from carla_env.simulator.actor import Actor


class Sensor(Actor[carla.Sensor]):
    def listen(self, callback: Callable[[carla.SensorData], None]):
        self.carla.listen(callback)
