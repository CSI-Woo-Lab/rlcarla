"""Behavior Cloning Environment."""

from pathlib import Path
from typing import Any, Dict, Optional

import carla
import flax
import gym.spaces
import numpy as np
import tqdm

from carla_env.base import BaseCarlaEnvironment
from carla_env.dataset import load_datasets
from carla_env.weathers import WEATHERS
from offline_baselines_jax.bc.bc import BC
from offline_baselines_jax.bc.policies import MultiInputPolicy
from utils.config import ExperimentConfigs
from utils.lidar import generate_lidar_bin
from utils.sensors import CollisionSensor, LaneInvasionSensor
from utils.vector import to_array

Params = flax.core.FrozenDict[str, Any]


class BehaviorCloningCarlaEnvironment(BaseCarlaEnvironment):
    """Behavior Cloning Environment."""

    def _init(self):
        # dummy variables, to match deep mind control's APIs
        self.action_space = gym.spaces.Box(shape=(2,), low=-1, high=1)
        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(shape=(24 + self.num_theta_bin,), low=-1, high=1),
                "task": gym.spaces.Box(shape=(12,), low=0, high=1),
                "module_select": gym.spaces.Box(shape=(36,), low=0, high=1),
            }
        )
        # roaming carla agent
        self.world.tick()

        self.collision_sensor = CollisionSensor(self.vehicle)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self)

        self.lane_invasion: Optional[Dict[carla.LaneMarkingType, Any]] = None

    def reset_init(self):
        super().reset_init()

        if self.lane_invasion_sensor.sensor is not None:
            self.lane_invasion_sensor.sensor.stop()
            self.lane_invasion_sensor.sensor.destroy()

        if self.collision_sensor.sensor is not None:
            self.collision_sensor.sensor.stop()
            self.collision_sensor.sensor.destroy()

        self.collision_sensor = CollisionSensor(self.vehicle)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self)
        self.lane_invasion = None

    def reset(self):
        self.reset_init()
        return super().reset()

    def goal_reaching_reward(self, vehicle: carla.Vehicle):
        colhist = self.collision_sensor.get_collision_history()
        lane_invasion = self.lane_invasion
        if lane_invasion is None:
            lane_invasion = {}
        lane_done = (
            colhist
            or carla.LaneMarkingType.Solid in lane_invasion
            or carla.LaneMarkingType.SolidSolid in lane_invasion
        )

        dist = self.get_distance_vehicle_target(vehicle)

        total_reward, reward_dict, done_dict = super().goal_reaching_reward(vehicle)

        done_dict = {
            "lane_collision_done": lane_done,
            "dist_done": dist < 15,
            **done_dict,
        }
        return total_reward, reward_dict, done_dict

    def _simulator_step(
        self,
        action: Optional[np.ndarray] = None,
        traffic_light_color: Optional[str] = None,
    ):
        expert_action = self.compute_action()[0]

        if action is None:
            throttle, steer, brake = 0.0, 0.0, 0.0
        else:
            steer = float(action[1])
            if action[0] >= 0.0:
                throttle = float(action[0])
                brake = 0.0
            else:
                throttle = 0.0
                brake = float(action[0])

            vehicle_control = carla.VehicleControl(
                throttle=throttle,  # [0,1]
                steer=steer,  # [-1,1]
                brake=brake,  # [0,1]
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
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
            "acceleration": to_array(self.vehicle.get_acceleration()),
            "angular_veolcity": to_array(self.vehicle.get_angular_velocity()),
            "location": to_array(self.vehicle.get_location()),
            "rotation": to_array(rotation),
            "forward_vector": to_array(rotation.get_forward_vector()),
            "veolcity": to_array(self.vehicle.get_velocity()),
            "target_location": to_array(self.target_location),
        }

        done = self.count >= self.max_episode_steps
        if done:
            print(
                f"Episode success: I've reached the episode horizon "
                f"({self.max_episode_steps})."
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
            "expert_action": np.array(
                [
                    expert_action.throttle - expert_action.brake,
                    expert_action.steer,
                ],
                dtype=np.float64,
            ),
        }

        next_obs_sensor = np.hstack(
            [value for key, value in next_obs.items() if key != "image"]
        )

        # next_obs_sensor = next_obs_sensor
        task = np.zeros(12)
        task[self.route] = 1
        return (
            {
                "obs": next_obs_sensor,
                "task": task,
                "module_select": np.ones(36),
            },
            reward,
            done or any(done_dict.values()),
            info,
        )


def behavior_cloning(config: ExperimentConfigs):
    """Behavior cloning experiment.
    
    Args:
        config (ExperimentConfigs): Experiment configs.
    """
    if config.carla_ip is None:
        print("Please pass your carla IP address")
        return

    data_path = config.data_path
    if data_path is not None and data_path.exists():
        datasets = load_datasets(data_path)
    else:
        datasets = None

    env = BehaviorCloningCarlaEnvironment(
        args=config,
        image_model=None,
        weather=WEATHERS[0],
        carla_ip=config.carla_ip,
        carla_port=2000 - config.num_routes * 5,
    )
    policy_kwargs = {"net_arch": [256, 256, 256, 256]}
    model = BC(
        policy=MultiInputPolicy,  # type: ignore
        env=env,
        verbose=1,
        gradient_steps=5,
        train_freq=1,
        batch_size=1024,
        learning_rate=3e-4,
        tensorboard_log="log",
        policy_kwargs=policy_kwargs,
        without_exploration=False,
    )
    # SpiRL

    if datasets is not None and model.replay_buffer is not None:
        for dataset in tqdm.tqdm(datasets):
            for i in range(dataset["observations"]["sensor"].shape[0] - 1):
                action = np.array(
                    [
                        dataset["actions"][i][0] - dataset["actions"][i][2],
                        dataset["actions"][i][1],
                    ]
                )
                info = {
                    **dataset["infos"][i],
                    "expert_action": action,
                }
                task = dataset["observations"]["task"][i]
                next_task = dataset["observations"]["task"][i + 1]
                model.replay_buffer.add(
                    obs=np.array(
                        {
                            "obs": np.hstack([dataset["observations"]["sensor"][i]]),
                            "task": task,
                            "module_select": np.ones(36),
                        }
                    ),
                    action=action,
                    reward=dataset["rewards"][i],
                    done=dataset["terminals"][i],
                    next_obs=np.array(
                        {
                            "obs": np.hstack(
                                [dataset["observations"]["sensor"][i + 1]]
                            ),
                            "task": next_task,
                            "module_select": np.ones(36),
                        }
                    ),
                    infos=[info],
                )

    for i in range(15):
        model.learn(total_timesteps=10000, log_interval=1)
        model.save(
            Path.cwd() / "models" / f"{config.mode}_model_route_{config.num_routes}_{i}"
        )
