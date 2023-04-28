from typing import Optional, cast

import carla
import numpy as np
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator
from utils.config import ExperimentConfigs


class CameraSensor(Sensor):
    def init(self, config: ExperimentConfigs):
        self.__vision_size = config.vision_size
        self.__image = np.zeros(
            (self.__vision_size, self.__vision_size), dtype=np.uint8
        )
        self.listen(self._callback__on_image)

    @classmethod
    @override
    async def spawn(
        cls,
        simulator: Simulator,
        config: ExperimentConfigs,
        parent: Actor,
    ):
        blueprint = simulator.world.blueprint_library.find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(config.vision_size))
        blueprint.set_attribute("image_size_y", str(config.vision_size))
        blueprint.set_attribute("fov", str(config.vision_fov))
        
        sensor = await super().spawn(
            simulator=simulator,
            blueprint=blueprint,
            attach_to=parent,
        )
        if not sensor:
            return None

        sensor.init(config)
        return sensor

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
