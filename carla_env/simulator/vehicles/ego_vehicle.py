from typing import Optional

import carla

from carla_env.simulator.sensors.camera import CameraSensor
from carla_env.simulator.sensors.collision import CollisionSensor
from carla_env.simulator.sensors.lane_invasion import LaneInvasionSensor
from carla_env.simulator.sensors.lidar import LidarSensor
from carla_env.simulator.simulator import Simulator
from carla_env.simulator.vehicles.vehicle import Vehicle
from utils.config import ExperimentConfigs


class EgoVehicle(Vehicle):
    def __init__(self, simulator: Simulator, config: ExperimentConfigs):
        super().__init__(
            simulator=simulator, blueprint=f"vehicle.{config.vehicle_type}"
        )
        self.__vehicle_type = config.vehicle_type
        self.blueprint.set_attribute("role_name", "ego")

        self.__lidar_sensor = LidarSensor(simulator, config)
        self.__camera = CameraSensor(simulator, config)
        self.__collision_sensor = CollisionSensor(simulator)
        self.__lane_invasion_sensor = LaneInvasionSensor(simulator)

    def spawn(self, initial_transform: Optional[carla.Transform] = None):
        if initial_transform is None:
            initial_transform = carla.Transform(
                carla.Location(x=.0, y=.0, z=.0), carla.Rotation(yaw=.0)
            )
        super().spawn(transform=initial_transform)

        self.__lidar_sensor.spawn(parent=self)
        self.__camera.spawn(parent=self)
        self.__collision_sensor.spawn(parent=self)
        self.__lane_invasion_sensor.spawn(parent=self)

        self.velocity = carla.Vector3D(x=.0, y=.0, z=.0)
        self.angular_velocity = carla.Vector3D(x=.0, y=.0, z=.0)

        self.world.get_spectator().follow(self)

    def destroy(self) -> None:
        self.__lidar_sensor.destroy()
        self.__camera.destroy()
        self.__collision_sensor.destroy()
        self.__lane_invasion_sensor.destroy()
        super().destroy()

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
