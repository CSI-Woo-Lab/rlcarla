import abc
import glob
import math
import os
import pickle as pkl
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import carla
import gym
import gym.spaces
import numpy as np
import pygame
from typing_extensions import Literal

from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import is_within_distance_ahead
from carla_env.dataset import Dataset, load_datasets
from carla_env.utils import build_goal_candidate
from utils.carla_sync_mode import CarlaSyncMode
from utils.config import ExperimentConfigs
from utils.roaming_agent import RoamingAgent
from utils.route_planner import CustomGlobalRoutePlanner


class BaseCarlaEnvironment(abc.ABC, gym.Env[dict, np.ndarray]):
    """Base Carla Environment.

    This class is the base class for all Carla environments. It provides the basic
    functionality to connect to a Carla server, spawn a vehicle, and control it. It also
    provides the basic functionality to record the data from the sensors.

    Args:
        config: Experiment configs.
        image_model: Image model to be used for image processing.
        weather: Weather to be used in the environment.
        carla_ip: IP address of the Carla server.
        carla_port: Port of the Carla server.
    """

    OBS_IDX = {
        "control": np.array([0, 1, 2]),
        "acceleration": np.array([3, 4, 5]),
        "angular_velocity": np.array([6, 7, 8]),
        "location": np.array([9, 10, 11]),
        "rotation": np.array([12, 13, 14]),
        "forward_vector": np.array([15, 16, 17]),
        "veloctiy": np.array([18, 19, 20]),
        "target_location": np.array([21, 22, 23]),
    }

    def __init__(
        self,
        config: ExperimentConfigs,
        image_model: Optional[Any],
        weather: str,
        carla_ip: str,
        carla_port: int,
    ):
        # New Hyperparameter
        self.random_route = config.random_route
        self.image_model = image_model
        self.weather = weather

        self.record_dir = config.data_path

        self.vision_size = config.vision_size
        self.vision_fov = config.vision_fov

        self.frame_skip = config.frame_skip
        self.max_episode_steps = config.max_steps
        self.multiagent = config.multiagent
        self.start_lane = config.lane
        self.follow_traffic_lights = config.lights
        self.route = 1
        self.route_list = config.routes
        self.vehicle_type = config.vehicle_type

        self.client = carla.Client(carla_ip, carla_port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.map = self.world.get_map()

        ## Define the route planner
        self.route_planner_dao = GlobalRoutePlannerDAO(
            self.map, sampling_resolution=0.1
        )
        self.route_planner = CustomGlobalRoutePlanner(self.route_planner_dao)
        ## Initialize the route planner
        self.route_planner.setup()

        # tests specific to map 4:
        if self.start_lane and self.map.name != "Town04":
            raise NotImplementedError

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()

        self.vehicle_ids: List[int] = []  # their ids
        self.reset_vehicle()  # creates self.vehicle

        blueprint_library = self.world.get_blueprint_library()

        # set the attributes, all values set as strings
        self.upper_fov = config.upper_fov
        self.lower_fov = config.lower_fov
        self.rotation_frequency = config.rotation_frequency
        self.range = config.max_range
        self.num_theta_bin = config.num_theta_bin

        self.dropoff_general_rate = config.dropoff_general_rate
        self.dropoff_intensity_limit = config.dropoff_intensity_limit
        self.dropoff_zero_intensity = config.dropoff_zero_intensity
        self.points_per_second = config.points_per_second

        self.lidar_obj = blueprint_library.find("sensor.lidar.ray_cast")
        self.lidar_obj = self.get_lidar_sensor()
        location = carla.Location(x=1.6, z=1.7)

        self.lidar_sensor = cast(
            carla.Sensor,
            self.world.try_spawn_actor(
                self.lidar_obj,
                carla.Transform(location, carla.Rotation(yaw=0.0)),
                attach_to=self.vehicle,
            ),
        )

        #  dataset
        self.data_path = config.data_path

        #  sync mode
        self.sync_mode = CarlaSyncMode(self.world, self.lidar_sensor, fps=20)

        self._init()
        self.reset_init()  # creates self.agent

        ## Collision detection
        self._proximity_threshold = 10.0
        self._traffic_light_threshold = 5.0
        self.actor_list = self.world.get_actors()
        for idx, actor in enumerate(self.actor_list):
            print(idx, actor)

        self.vehicle_list: List[carla.Vehicle] = self.actor_list.filter("*vehicle*")
        self.lights_list: List[carla.TrafficLight] = self.actor_list.filter(
            "*traffic_light*"
        )
        self.object_list: List[carla.Vehicle] = self.actor_list.filter("*traffic.*")

        ## Initialize the route planner
        self.route_planner.setup()

        ## This is a dummy for the target location, we can make this an input
        ## to the env in RL code.
        self.target_location = carla.Location(x=-13.473097, y=134.311234, z=-0.010433)

        ## Now reset the env once
        self.reset()

    def get_lidar_sensor(self, role_name: str = "lidar") -> carla.ActorBlueprint:  # @
        # set the attributes, all values set as strings
        attributes = [
            "upper_fov",
            "lower_fov",
            "rotation_frequency",
            "range",
            "dropoff_general_rate",
            "dropoff_intensity_limit",
            "dropoff_zero_intensity",
            "points_per_second",
        ]

        for attr in attributes:
            self.lidar_obj.set_attribute(attr, str(getattr(self, attr)))
        return self.lidar_obj

    def _init(self) -> None:
        ...

    def reset_init(self):
        waypoint_list = self.reset_vehicle()
        self.world.tick()
        self.reset_other_vehicles()
        self.world.tick()

        self.world.set_weather(getattr(carla.WeatherParameters, self.weather))

        # self.weather.tick()
        self.agent = RoamingAgent(
            self.vehicle, follow_traffic_lights=self.follow_traffic_lights
        )
        # pylint: disable=protected-access
        self.agent._local_planner.set_global_plan(waypoint_list)

        self.count = 0

    def reset(self):
        # get obs:
        obs, _, _, _ = self.step()
        return obs

    def seed(self, seed: int):
        return seed

    def compute_action(
        self,
    ) -> Tuple[
        carla.VehicleControl,
        Union[Tuple[Literal[True], carla.Actor], Tuple[Literal[False], None]],
    ]:
        return self.agent.run_step()

    def reset_vehicle(self) -> List[carla.Waypoint]:
        waypoint_list: List[carla.Waypoint]

        if self.map.name == "Town04":
            start_x = 5.0
            vehicle_init_transform = carla.Transform(
                carla.Location(x=start_x, y=0, z=0.1), carla.Rotation(yaw=-90)
            )
            waypoint_list = []
        else:
            init_transforms = self.world.get_map().get_spawn_points()

            if not self.route_list:
                route_candidate = [
                    [1, 37],
                    [0, 87],
                    [21, 128],
                    [0, 152],
                    [21, 123],
                    [21, 6],
                    [126, 4],
                    [135, 37],
                    [126, 143],
                    [129, 119],
                    [129, 60],
                    [129, 146],
                ]
            else:
                route_candidate = self.route_list

            self.route = (self.route + 1) % len(route_candidate)

            choice_route = route_candidate[self.route]

            vehicle_init_transform = init_transforms[choice_route[0]]
            self.target_location = init_transforms[choice_route[1]].location

            if self.random_route:
                vehicle_init_transform = random.choice(init_transforms)
                goal_candidate = build_goal_candidate(
                    self.world, vehicle_init_transform.location
                )
                if not goal_candidate:
                    goal_candidate = build_goal_candidate(
                        self.world, vehicle_init_transform.location, threshold=100
                    )
                self.target_location = random.choice(goal_candidate).location

            waypoint_list = self.route_planner.trace_route(
                vehicle_init_transform.location, self.target_location
            )

        # TODO(aviral): start lane not defined for town, also for the town, we may not
        # want to have the lane following reward, so it should be okay.

        if not hasattr(self, "vehicle"):  # then create the ego vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find(f"vehicle.{self.vehicle_type}")
            vehicle = self.world.spawn_actor(vehicle_blueprint, vehicle_init_transform)
            if not isinstance(vehicle, carla.Vehicle):
                raise ValueError
            self.vehicle = vehicle

        self.vehicle.set_transform(vehicle_init_transform)
        self.vehicle.set_target_velocity(carla.Vector3D())
        self.vehicle.set_target_angular_velocity(carla.Vector3D())

        return waypoint_list

    def reset_other_vehicles(self):
        if not self.multiagent:
            return

        # clear out old vehicles
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.vehicle_ids]
        )
        self.world.tick()
        # self.sensor.tick()
        self.vehicle_ids = []

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        blueprints = [
            actor
            for actor in self.world.get_blueprint_library().filter("vehicle.*")
            if int(actor.get_attribute("number_of_wheels")) == 4
        ]

        num_vehicles = 20
        if self.map.name == "Town04":
            road_id = 47
            road_length = 117.0
            init_transforms = [
                self.map.get_waypoint_xodr(
                    road_id,
                    random.choice([-1, -2, -3, -4]),
                    np.random.uniform(road_length),
                ).transform
                for _ in range(num_vehicles)
            ]
        else:
            init_transforms = self.world.get_map().get_spawn_points()
            init_transforms = random.choices(init_transforms, k=num_vehicles)

        # --------------
        # Spawn vehicles
        # --------------
        batch: List[carla.command.SpawnActor] = []
        for transform in init_transforms:
            # otherwise can collide with the road it starts on
            transform.location.z += 0.1

            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")

            batch.append(
                carla.command.SpawnActor(blueprint, transform).then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor, True  # type: ignore
                    )  # type: ignore
                )
            )

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicle_ids.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicle_ids.append(response.actor_id)

        traffic_manager.global_percentage_speed_difference(30.0)

    def step(
        self,
        action: Optional[np.ndarray] = None,
        traffic_light_color: Optional[str] = "",
    ) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        rewards: List[np.ndarray] = []
        next_obs, done, info = None, None, None
        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(
                action, traffic_light_color
            )
            rewards.append(reward)

            if done:
                break

        if next_obs is None or done is None or info is None:
            raise ValueError("frame_skip >= 1")
        return next_obs, np.mean(rewards), done, info

    def _is_vehicle_hazard(self, vehicle: carla.Vehicle, targets: List[carla.Vehicle]):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
            - bool_flag is True if there is a vehicle ahead blocking us and False otherwise
            - vehicle is the blocker object itself
        """

        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in targets:
            # do not account for the ego vehicle
            if target_vehicle.id == vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(
                target_vehicle.get_location()
            )
            if (
                target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id
                or target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id
            ):
                continue

            if is_within_distance_ahead(
                target_vehicle.get_transform(),
                vehicle.get_transform(),
                self._proximity_threshold / 10.0,
            ):
                return True, -1.0, target_vehicle

        return False, 0.0, None

    def _is_object_hazard(self, vehicle: carla.Vehicle, targets: List[carla.Vehicle]):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in targets:
            # do not account for the ego vehicle
            if target_vehicle.id == vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(
                target_vehicle.get_location()
            )
            if (
                target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id
                or target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id
            ):
                continue

            if is_within_distance_ahead(
                target_vehicle.get_transform(),
                vehicle.get_transform(),
                self._proximity_threshold / 40.0,
            ):
                return True, -1.0, target_vehicle

        return False, 0.0, None

    def _get_trafficlight_trigger_location(
        self, traffic_light: carla.TrafficLight
    ):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point: carla.Vector3D, radians: float):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def _is_light_red(self, vehicle: carla.Actor):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.
        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for traffic_light in self.lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self.map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(
                object_waypoint.transform,
                vehicle.get_transform(),
                self._traffic_light_threshold,
            ):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return True, -0.1, traffic_light

        return False, 0.0, None

    def _get_collision_reward(self, vehicle: carla.Vehicle):
        vehicle_hazard, reward, _ = self._is_vehicle_hazard(vehicle, self.vehicle_list)
        return vehicle_hazard, reward

    def _get_traffic_light_reward(self, vehicle: carla.Vehicle):
        traffic_light_hazard, _, _ = self._is_light_red(vehicle)
        return traffic_light_hazard, 0.0

    def _get_object_collided_reward(self, vehicle: carla.Vehicle):
        object_hazard, reward, _ = self._is_object_hazard(vehicle, self.object_list)
        return object_hazard, reward

    def get_distance_vehicle_target(self, vehicle: carla.Vehicle):
        vehicle_location = vehicle.get_location()
        target_location = self.target_location
        return np.linalg.norm(
            np.array(
                [
                    vehicle_location.x - target_location.x,
                    vehicle_location.y - target_location.y,
                    vehicle_location.z - target_location.z,
                ]
            )
        )

    def goal_reaching_reward(self, vehicle: carla.Vehicle):
        # Now we will write goal_reaching_rewards
        vehicle_location = vehicle.get_location()
        target_location = self.target_location

        # This is the distance computation
        """
        dist = self.route_planner.compute_distance(vehicle_location, target_location)

        base_reward = -1.0 * dist
        collided_done, collision_reward = self._get_collision_reward(vehicle)
        traffic_light_done, traffic_light_reward = self._get_traffic_light_reward(vehicle)
        object_collided_done, object_collided_reward = self._get_object_collided_reward(vehicle)
        total_reward = base_reward + 100 * collision_reward + 100 * traffic_light_reward + 100.0 * object_collided_reward
        """
        vehicle_velocity = vehicle.get_velocity()

        # dist = self.route_planner.compute_distance(vehicle_location, target_location)
        vel_forward, vel_perp = self.route_planner.compute_direction_velocities(
            vehicle_location, vehicle_velocity, target_location
        )

        # print('[GoalReachReward] VehLoc: %s Target: %s Dist: %s VelF:%s' % (str(vehicle_location), str(target_location), str(dist), str(vel_forward)))
        # base_reward = -1.0 * (dist / 100.0) + 5.0
        base_reward = vel_forward
        collided_done, collision_reward = self._get_collision_reward(vehicle)
        _, traffic_light_reward = self._get_traffic_light_reward(vehicle)
        (
            object_collided_done,
            object_collided_reward,
        ) = self._get_object_collided_reward(vehicle)
        total_reward: np.ndarray = (
            base_reward + 100 * collision_reward
        )  # + 100 * traffic_light_reward + 100.0 * object_collided_reward

        reward_dict = {
            "collision": collision_reward,
            "traffic_light": traffic_light_reward,
            "object_collision": object_collided_reward,
            "base_reward": base_reward,
            "vel_forward": vel_forward,
            "vel_perp": vel_perp,
        }

        done_dict = {
            "collided_done": collided_done,
            "traffic_light_done": False,
            "object_collided_done": object_collided_done,
            "reached_max_steps": self.count >= self.max_episode_steps,
        }

        return total_reward, reward_dict, done_dict

    def _simulator_step(
        self,
        action: Optional[np.ndarray] = None,
        traffic_light_color: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        raise NotImplementedError

    def finish(self):
        print("destroying actors.")
        for actor in self.actor_list:
            actor.destroy()
        print("\ndestroying %d vehicles" % len(self.vehicle_ids))
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.vehicle_ids]
        )
        time.sleep(0.5)
        pygame.quit()
        print("done.")

    def get_dataset(self) -> List[Dataset]:
        if self.data_path is None or not self.data_path.exists():
            return []
        return list(load_datasets(self.data_path))
