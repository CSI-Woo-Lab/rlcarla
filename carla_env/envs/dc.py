import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import carla
import numpy as np
from dotmap import DotMap

from carla_env.envs.base import BaseCarlaEnv
from utils.lidar import generate_lidar_bin
from utils.vector import rotation_to_array, vector_to_array


class DCCarlaEnv(BaseCarlaEnv):
    def _init(self):
        self.lidar_sensor = self.world.try_spawn_actor(
            self.lidar_obj,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0.0)),
            attach_to=self.vehicle,
        )
        if self.record_dir is None:
            self.record_dir = self.__create_record_dirpath()

        record_path = Path(self.record_dir) / "record"
        record_path.mkdir(parents=True, exist_ok=True)

        # dummy variables, to match deep mind control's APIs
        low = -1.0
        high = 1.0

        self.action_space = DotMap()  # type: ignore
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = (2,)
        self.observation_space = DotMap()  # type: ignore
        self.observation_space.dtype = np.dtype(np.uint8)

        self.action_space.sample = lambda: np.random.uniform(
            low=low, high=high, size=self.action_space.shape[0]  # type: ignore
        ).astype(np.float32)

        # roaming carla agent
        self.world.tick()

    def __create_record_dirpath(self):
        now = datetime.datetime.now()

        # Example: carla_data/carla-town01-224x224-fov90-1k-2020-05-20-15-00-00
        return os.path.join(
            "carla_data",
            "-".join(
                [
                    "carla",
                    self.map.name.lower().split("/")[-1],
                    f"{self.vision_size}x{self.vision_size}",
                    f"fov{self.vision_fov}",
                    f"{self.frame_skip}" if self.frame_skip > 1 else "",
                    "multiagent" if self.multiagent else "",
                    "lights" if self.follow_traffic_lights else "",
                    f"{self.max_episode_steps // 1000}k",
                    now.strftime("%Y-%m-%d-%H-%M-%S"),
                ]
            ),
        )

    def goal_reaching_reward(self, vehicle: carla.Vehicle):
        total_reward, reward_dict, done_dict = super().goal_reaching_reward(vehicle)

        dist = self.get_distance_vehicle_target(vehicle)

        reward_dict = {
            "dist": dist,
            **reward_dict,
        }
        done_dict = {
            "dist_done": dist < 5,
            **done_dict,
        }

        return total_reward, reward_dict, done_dict

    def step(
        self,
        action: Optional[carla.VehicleControl] = None,
        traffic_light_color: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, Dict[str, Any]]:
        return super().step(cast(np.ndarray, action), traffic_light_color)

    def _simulator_step(
        self,
        action: Optional[carla.VehicleControl] = None,
        traffic_light_color: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, Dict[str, Any]]:
        if action is None:
            throttle, steer, brake = 0.0, 0.0, 0
        else:
            throttle, steer, brake = action.throttle, action.steer, action.brake
            if float(brake) < 0.01:
                brake = 0.0

            vehicle_control = carla.VehicleControl(
                throttle=throttle,  # [0,1]
                steer=steer,  # [-1,1]
                brake=brake,  # [0,1]
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=0,
            )
            self.vehicle.apply_control(vehicle_control)

        # Advance the simulation and wait for the data.
        _, lidar_sensor = self.sync_mode.tick(timeout=10.0)
        lidar_bin = generate_lidar_bin(lidar_sensor, self.num_theta_bin, self.range)

        reward, reward_dict, done_dict = self.goal_reaching_reward(self.vehicle)

        self.count += 1

        rotation = self.vehicle.get_transform().rotation
        next_obs = {
            "lidar": np.array(lidar_bin),
            "control": np.array([throttle, steer, brake]),
            "acceleration": vector_to_array(self.vehicle.get_acceleration()),
            "angular_veolcity": vector_to_array(self.vehicle.get_angular_velocity()),
            "location": vector_to_array(self.vehicle.get_location()),
            "rotation": rotation_to_array(rotation),
            "forward_vector": vector_to_array(rotation.get_forward_vector()),
            "veolcity": vector_to_array(self.vehicle.get_velocity()),
            "target_location": vector_to_array(self.target_location),
        }
        next_obs_sensor = np.hstack(
            [value for key, value in next_obs.items() if key != "image"]
        )

        info = {
            **{f"reward_{key}": value for key, value in reward_dict.items()},
            **{f"done_{key}": value for key, value in done_dict.items()},
            "control_repeat": self.frame_skip,
            "weather": self.weather,
            "settings_map": self.map.name,
            "settings_multiagent": self.multiagent,
            "traffic_lights_color": "UNLABELED",
            "reward": reward,
        }

        done = any(done_dict.values())

        return (
            {"sensor": next_obs_sensor},
            reward,
            done,
            info,
        )
