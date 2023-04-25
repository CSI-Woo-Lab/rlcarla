from typing import Optional, cast

import carla
import numpy as np
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator
from utils.config import ExperimentConfigs


class CameraSensor(Sensor):
    def __init__(self, simulator: Simulator, config: ExperimentConfigs):
        blueprint = simulator.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(config.vision_size))
        blueprint.set_attribute("image_size_y", str(config.vision_size))
        blueprint.set_attribute("fov", str(config.vision_fov))
        super().__init__(simulator, blueprint)

        self.__vision_size = config.vision_size
        self.__fov = config.vision_fov
        self.__image = np.zeros(
            (self.__vision_size, self.__vision_size), dtype=np.uint8
        )

    @override
    def spawn(self, parent: Actor):
        super().spawn(
            transform=carla.Transform(
                carla.Location(x=1.5, y=0.0, z=0.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
            ),
            parent=parent,
        )

    def _on_spawn(self, response: carla.command.Response) -> None:
        super()._on_spawn(response)
        self.actor.listen(self._callback__on_image)

    def _callback__on_image(self, data: carla.SensorData):
        image: np.ndarray = np.frombuffer(
            cast(carla.Image, data).raw_data, dtype=np.uint8
        )
        image = image.reshape((self.__vision_size, self.__vision_size, 4))
        image = image[:, :, 3]
        image = np.fliplr(image)
        self.__image[...] = image

    @property
    def image(self) -> np.ndarray:
        """The current image observed by the camera sensor."""
        return self.__image
