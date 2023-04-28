"""Collect data from Carla simulator.

example call:
./PythonAPI/util/config.py --map Town01 --delta-seconds 0.05
python PythonAPI/carla/agents/navigation/data_collection_agent.py \
    --vision_size 256 --vision_fov 90 --steps 10000 --weather --lights
"""

import datetime
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carla
import flax
import numpy as np
from dotmap import DotMap

from carla_env.base import BaseCarlaEnvironment
from carla_env.dataset import Dataset, dump_dataset
from utils.config import ExperimentConfigs
from utils.lidar import generate_lidar_bin
from utils.logger import Logging
from utils.vector import to_array

logger = Logging.get_logger(__name__)
Params = flax.core.FrozenDict[str, Any]


class DataCollectingCarlaEnvironment(BaseCarlaEnvironment):
    """Carla environment for data collection."""

    def __init__(self, config: ExperimentConfigs):
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

        super().__init__(config)

        self.record_dir = self.__create_record_dirpath(config.data_path)

        record_path = self.record_dir / "record"
        record_path.mkdir(parents=True, exist_ok=True)

    def __create_record_dirpath(self, base_dir: Optional[Path] = None):
        """Create a directory path to save the collected data.
        
        Returns:
            Path: Path to the directory to save the collected data.
            
        Example:
            carla_data/carla-town01-224x224-fov90-1k-2020-05-20-15-00-00
        """
        if base_dir is None:
            base_dir = Path.cwd() / "carla_data"
        now = datetime.datetime.now()

        # Example: carla_data/carla-town01-224x224-fov90-1k-2020-05-20-15-00-00
        return (
            base_dir
            / "-".join(
                x for x in
                [
                    "carla",
                    self.sim.world.map.name.lower().split("/")[-1],
                    f"{self.config.vision_size}x{self.config.vision_size}",
                    f"fov{self.config.vision_fov}",
                    f"{self.config.frame_skip}" if self.config.frame_skip > 1 else "",
                    "multiagent" if self.config.multiagent else "",
                    "lights" if self.config.lights else "",
                    f"{self.config.max_steps // 1000}k",
                    now.strftime("%Y-%m-%d-%H-%M-%S"),
                ]
                if x
            )
        )

    def goal_reaching_reward(self):
        total_reward, reward_dict, done_dict = super().goal_reaching_reward()

        dist = self.get_distance_vehicle_target()

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
        traffic_light_color: Optional[str] = "",
    ) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        """Step the environment.
        
        Args:
            action (carla.VehicleControl, optional): Vehicle control. Defaults to None.
            traffic_light_color (str, optional): Traffic light color. Defaults to "".
            
        Returns:
            Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
                next_obs (Dict[str, Any]): Next observation.
                reward (np.ndarray): Reward.
                done (bool): Whether the episode is done.
                info (Dict[str, Any]): Info.

        Raises:
            ValueError: Raises an error if frame_skip is less than 1.
        """
        rewards: List[np.ndarray] = []
        next_obs, done, info = None, None, None
        for _ in range(self.config.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(
                action, traffic_light_color
            )
            rewards.append(reward)

            if done:
                break

        if next_obs is None or done is None or info is None:
            raise ValueError("frame_skip must be greater than 0.")
        return next_obs, np.mean(rewards), done, info

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
            self.sim.ego_vehicle.apply_control(vehicle_control)

        if self.count == 0:
            logger.info(
                "Vehicle starts at: %s",
                to_array(self.sim.ego_vehicle.location),
            )

        # Advance the simulation and wait for the data.
        _, lidar_sensor = self.sync_mode.tick(timeout=10.0)
        lidar_bin = generate_lidar_bin(
            lidar_sensor, self.config.lidar.num_theta_bin, self.config.lidar.max_range
        )
        self.count += 1

        reward, reward_dict, done_dict = self.goal_reaching_reward()

        rotation = self.sim.ego_vehicle.rotation
        next_obs = {
            "lidar": np.array(lidar_bin),
            "control": np.array([throttle, steer, brake]),
            "acceleration": to_array(self.sim.ego_vehicle.acceleration),
            "angular_veolcity": to_array(self.sim.ego_vehicle.angular_velocity),
            "location": to_array(self.sim.ego_vehicle.location),
            "rotation": to_array(rotation),
            "forward_vector": to_array(rotation.get_forward_vector()),
            "veolcity": to_array(self.sim.ego_vehicle.velocity),
            "target_location": to_array(self.sim.target_location),
        }
        next_obs_sensor = np.hstack(
            [value for key, value in next_obs.items() if key != "image"]
        )

        info = {
            **{f"reward_{key}": value for key, value in reward_dict.items()},
            **{f"done_{key}": value for key, value in done_dict.items()},
            "control_repeat": self.config.frame_skip,
            "weather": self.weather,
            "settings_map": self.sim.world.map.name,
            "settings_multiagent": self.config.multiagent,
            "traffic_lights_color": "UNLABELED",
            "reward": reward,
        }

        done = any(done_dict.values())

        if self.count % 50 == 0 or done:
            logger.info("Step: %s", self.count)
            logger.info("Vehicle: %s", next_obs["location"])
            logger.info("Target: %s", next_obs["target_location"])
            logger.info("Reward: %s (%s)", reward, reward_dict)
            logger.info("Done: %s (%s)", done, done_dict)

        if done_dict["reached_max_steps"]:
            logger.warning("Episode reached max steps. Terminating episode.")

        return (
            {"sensor": next_obs_sensor, "image": self.sim.ego_vehicle.camera.image},
            reward,
            done,
            info,
        )


def collect_data(config: ExperimentConfigs):
    """Collect data from CARLA simulator.
    
    Args:
        config (ExperimentConfigs): Experiment configs.
        
    Raises:
        ValueError: Raises an error if carla_ip is None.
    """
    if config.carla_ip is None:
        print("Please pass your carla IP address")
        return

    env = DataCollectingCarlaEnvironment(config)

    curr_steps = 0
    # for weather in weather_list:
    weather = "ClearNoon"
    env.weather = "ClearNoon"

    record_dirname_per_weather = env.record_dir / "record" / weather
    record_dirname_per_weather.mkdir(parents=True, exist_ok=True)

    total_step = 0
    for j in range(12000):
        curr_steps = 0
        observations_sensor: List[np.ndarray] = []
        observations_image: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[np.ndarray] = []
        terminals: List[bool] = []
        infos: List[dict] = []

        logger.info("EPISODE: %s (%s/1,000,000)", j, format(total_step, ","))

        env.reset()
        done = False
        while not done:
            curr_steps += 1
            action, _ = env.compute_action()
            # action.steer = random.random() * 2 - 1
            next_obs, reward, done, info = env.step(action)

            action = np.array([action.throttle, action.steer, action.brake])
            observations_sensor.append(next_obs["sensor"].copy())
            observations_image.append(next_obs["image"].copy())
            actions.append(action.copy())
            rewards.append(reward)
            terminals.append(done)
            infos.append(info)

        total_step += curr_steps

        if total_step > 1_000_000:
            logger.info("Finished collecting data")
            break

        dataset: Dataset = {
            "observations": {
                "sensor": np.array(observations_sensor),
                "image": np.array(observations_image),
            },
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "terminals": np.array(terminals),
            "infos": infos,
            "lidar_bin": config.lidar.num_theta_bin,
        }

        if infos[-1]["done_dist_done"]:
            filename = env.record_dir / f"episode_{j}.pkl"
        else:
            filename = env.record_dir / f"episode_{j}_failed.pkl"
        dump_dataset(dataset, filename)
