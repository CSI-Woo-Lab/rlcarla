import asyncio
from typing import Optional

import carla
from typing_extensions import override

from carla_env.simulator.sensors.camera import CameraSensor
from carla_env.simulator.sensors.collision import CollisionSensor
from carla_env.simulator.sensors.lane_invasion import LaneInvasionSensor
from carla_env.simulator.sensors.lidar import LidarSensor
from carla_env.simulator.simulator import Simulator
from carla_env.simulator.vehicles.vehicle import Vehicle
from utils.config import ExperimentConfigs


class EgoVehicle(Vehicle):
    def init(
        self,
        config: ExperimentConfigs,
        lidar_sensor: LidarSensor,
        camera: CameraSensor,
        collision_sensor: CollisionSensor,
        lane_invasion_sensor: LaneInvasionSensor,
    ):
        self.__vehicle_type = config.vehicle_type
        self.__lidar_sensor = lidar_sensor
        self.__camera = camera
        self.__collision_sensor = collision_sensor
        self.__lane_invasion_sensor = lane_invasion_sensor

    @classmethod
    async def spawn(
        cls,
        simulator: Simulator,
        config: ExperimentConfigs,
        initial_transform: Optional[carla.Transform] = None,
    ):
        blueprint = simulator.world.blueprint_library.find(
            f"vehicle.{config.vehicle_type}"
        )
        blueprint.set_attribute("role_name", "ego")

        vehicle = await super().spawn(
            simulator=simulator,
            blueprint=blueprint,
            transform=initial_transform,
        )

        if not vehicle:
            return None

        lidar_sensor, camera, collision_sensor, lane_invasion = await asyncio.gather(
            LidarSensor.spawn(simulator, config, vehicle),
            CameraSensor.spawn(simulator, config, vehicle),
            CollisionSensor.spawn(simulator, vehicle),
            LaneInvasionSensor.spawn(simulator, vehicle),
        )

        if (
            lidar_sensor is None
            or camera is None
            or collision_sensor is None
            or lane_invasion is None
        ):
            return None

        vehicle.init(config, lidar_sensor, camera, collision_sensor, lane_invasion)

        vehicle.velocity = carla.Vector3D(x=.0, y=.0, z=.0)
        vehicle.angular_velocity = carla.Vector3D(x=.0, y=.0, z=.0)

        simulator.world.get_spectator().follow(vehicle)

        return vehicle

    @override
    async def destroy(self):
        await asyncio.gather(
            self.__lidar_sensor.destroy(),
            self.__camera.destroy(),
            self.__collision_sensor.destroy(),
            self.__lane_invasion_sensor.destroy(),
        )
        await super().destroy()

    @property
    def lidar_sensor(self) -> LidarSensor:
        """Lidar sensor of the ego vehicle."""
        return self.__lidar_sensor

    @property
    def camera(self) -> CameraSensor:
        """Camera sensor of the ego vehicle."""
        return self.__camera
    
    @property
    def collision_sensor(self) -> CollisionSensor:
        """Collision sensor of the ego vehicle."""
        return self.__collision_sensor

    @property
    def lane_invasion_sensor(self) -> LaneInvasionSensor:
        """Lane invasion sensor of the ego vehicle."""
        return self.__lane_invasion_sensor

    @property
    def vehicle_type(self):
        return self.__vehicle_type
