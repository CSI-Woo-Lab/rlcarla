import datetime
import os
import time

import carla
import numpy as np
from dotmap import DotMap

from carla_env.envs.base import BaseCarlaEnv
from utils.cart import cart2pol


class DCCarlaEnv(BaseCarlaEnv):
    def _init(self):
        self.lidar_sensor = self.world.try_spawn_actor(
            self.lidar_obj,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0.0)),
            attach_to=self.vehicle,
        )
        self.actor_list.append(self.lidar_sensor)
        if self.record_dir is None:
            self.record_dir = os.path.join(
                "carla_data",
                "-".join([
                    "carla",
                    self.map.name.lower().split("/")[-1],
                    f"{self.vision_size}x{self.vision_size}",
                    f"fov{self.vision_fov}",
                ])
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

        self.action_space = DotMap()
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = (2,)
        self.observation_space = DotMap()
        # self.observation_space.shape = (3, self.vision_size, self.vision_size)
        self.observation_space.dtype = np.dtype(np.uint8)

        self.action_space.sample = lambda: np.random.uniform(
            low=low, high=high, size=self.action_space.shape[0] # type: ignore
        ).astype(np.float32)

        # roaming carla agent
        self.world.tick()

    def goal_reaching_reward(self, vehicle):
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

    def _simulator_step(self, action, traffic_light_color=None):
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
            else:
                brake = brake

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
        snapshot, lidar_sensor = self.sync_mode.tick(timeout=10.0)

        # Format rl lidar
        lidar = np.frombuffer(lidar_sensor.raw_data, dtype=np.float32).reshape((-1, 4))

        # (x,y,z) to (min_dist,theta,z)
        lidar_xy = lidar[:, :2]
        lidar_z = lidar[:, 2]

        lidar_xy_cart2pol = np.array(
            list(map(lambda x: cart2pol(x[0], x[1]), lidar_xy))
        )
        lidar_z = np.expand_dims(lidar_z, axis=1)
        lidar_cylinder = np.concatenate((lidar_xy_cart2pol, lidar_z), axis=1)

        lidar_bin = []
        empty_cnt = 0
        # discretize theta
        for i in range(-1 * int(self.num_theta_bin / 2), int(self.num_theta_bin / 2)):
            low_deg = 2 * i * np.pi / self.num_theta_bin
            high_deg = 2 * (i + 1) * np.pi / self.num_theta_bin
            points = lidar_cylinder[
                (lidar_cylinder[:, 1] > low_deg) * (lidar_cylinder[:, 1] < high_deg)
            ][:, 0]

            if not points.any():
                # print(f'{i} ~ {i+1} bin is empty')
                empty_cnt += 1
                lidar_bin.append(np.array([self.range]))
            else:
                max_idx = points.argmax()  # standard (x,y) or (x,y,z)
                lidar_bin.append(
                    lidar_cylinder[
                        (lidar_cylinder[:, 1] > low_deg)
                        * (lidar_cylinder[:, 1] < high_deg)
                    ][max_idx][0]
                )

        reward, reward_dict, done_dict = self.goal_reaching_reward(self.vehicle)

        self.count += 1

        acceleration = self.vehicle.get_acceleration()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        location = self.vehicle.get_location()
        rotation = self.vehicle.get_transform().rotation
        forward_vector = rotation.get_forward_vector()

        next_obs = {
            "lidar": np.array(lidar_bin),
            "control": np.array([throttle, steer, brake]),
            "acceleration": np.array([acceleration.x, acceleration.y, acceleration.z]),
            "veolcity": np.array([velocity.x, velocity.y, velocity.z]),
            "angular_veolcity": np.array(
                [angular_velocity.x, angular_velocity.y, angular_velocity.z]
            ),
            "location": np.array([location.x, location.y, location.z]),
            "rotation": np.array([rotation.pitch, rotation.yaw, rotation.roll]),
            "forward_vector": np.array(
                [forward_vector.x, forward_vector.y, forward_vector.z]
            ),
            "target_location": np.array(
                [
                    self.target_location.x,
                    self.target_location.y,
                    self.target_location.z,
                ]
            ),
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
        next_obs_sensor = np.hstack([
            value for key, value in next_obs.items() if key != "image"
        ])

        return (
            {"sensor": next_obs_sensor},
            reward,
            done,
            info,
        )  # , 'image': next_obs_image
