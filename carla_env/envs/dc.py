import datetime
import os
from typing import Any, Dict, Optional, Tuple, Union, cast

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
            self.record_dir = os.path.join(
                "carla_data",
                "-".join(
                    [
                        "carla",
                        self.map.name.lower().split("/")[-1],
                        f"{self.vision_size}x{self.vision_size}",
                        f"fov{self.vision_fov}",
                    ]
                ),
            )
            if self.frame_skip > 1:
                self.record_dir += f"-{self.frame_skip}"
            if self.multiagent:
                self.record_dir += "-mutiagent"
            if self.follow_traffic_lights:
                self.record_dir += "-lights"
            self.record_dir += f"-{self.max_episode_steps // 1000}k"

            now = datetime.datetime.now()
            self.record_dir += now.strftime("-%Y-%m-%d-%H-%M-%S")
        if not os.path.exists(self.record_dir):
            os.mkdir(self.record_dir)
        if not os.path.exists(os.path.join(self.record_dir, "record")):
            os.mkdir(os.path.join(self.record_dir, "record"))

        # dummy variables, to match deep mind control's APIs
        low = -1.0
        high = 1.0

        self.action_space = DotMap()    # type: ignore
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = (2,)
        self.observation_space = DotMap()   # type: ignore
        self.observation_space.dtype = np.dtype(np.uint8)

        self.action_space.sample = lambda: np.random.uniform(
            low=low, high=high, size=self.action_space.shape[0]  # type: ignore
        ).astype(np.float32)

        # roaming carla agent
        self.world.tick()

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
        action: Optional[Union[np.ndarray, carla.VehicleControl]] = None,
        traffic_light_color: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, Dict[str, Any]]:
        if action is None:
            throttle, steer, brake = 0.0, 0.0, 0
        elif isinstance(action, np.ndarray):
            control = carla.VehicleControl(0.0, 0.0, 0.0)
            throttle, steer, brake = (
                control.throttle,
                control.steer,
                control.brake,
            )
            vehicle_control = carla.VehicleControl(
                throttle=throttle,  # [0,1]
                steer=steer,  # [-1,1]
                brake=brake,  # [0,1]
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
            print(
                vehicle_control.throttle,
                vehicle_control.steer,
                vehicle_control.brake,
            )
            self.vehicle.apply_control(vehicle_control)

        else:
            throttle, steer, brake = action.throttle, action.steer, action.brake
            # throttle = clamp(throttle, minimum=0.005, maximum=0.995) + np.random.uniform(low=-0.003, high=0.003)
            # steer = clamp(steer, minimum=-0.995, maximum=0.995) + np.random.uniform(low=-0.003, high=0.003)
            # brake = clamp(brake, minimum=0.000, maximum=0.995) + np.random.uniform(low=-0.005, high=0.005)
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
        # # To inspect images, run:
        # import pdb; pdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.imshow(next_obs)
        # plt.show()

        done = False  # self.count >= self.max_episode_steps
        if done:
            print(
                "Episode success: I've reached the episode horizon ({}).".format(
                    self.max_episode_steps
                )
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
        next_obs_sensor = np.hstack(
            [value for key, value in next_obs.items() if key != "image"]
        )

        return (
            {"sensor": next_obs_sensor},
            reward,
            done,
            info,
        )  # , 'image': next_obs_image
