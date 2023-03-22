import abc
import glob
import math
import os
import pickle as pkl
import random
import time
from typing import Optional

import carla
import gym
import gym.spaces
import numpy as np
import pygame

from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import is_within_distance_ahead
from utils.arguments import EnvArguments
from utils.carla_sync_mode import CarlaSyncMode
from utils.roaming_agent import RoamingAgent
from utils.route_planner import CustomGlobalRoutePlanner


class BaseCarlaEnv(abc.ABC, gym.Env):
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
        args: EnvArguments,
        image_model,
        weather: str,
        carla_ip: str,
        carla_port: int,
    ):
        # New Hyperparameter
        self.random_route = args.random_route
        self.image_model = image_model
        self.record_display = False
        self.record_dir = "./carla_data"
        self.weather = weather

        self.frame_skip = args.frame_skip
        self.max_episode_steps = args.steps
        self.multiagent = args.multiagent
        self.start_lane = args.lane
        self.follow_traffic_lights = args.lights
        self.mode = args.mode
        if self.record_display:
            assert self.render_display
        self.route = 1
        self.route_list = args.route_list
        self.video = None
        self.actor_list = []

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

        self.vehicle = None
        self.vehicles_list = []  # their ids
        self.reset_vehicle()  # creates self.vehicle
        self.actor_list.append(self.vehicle)

        blueprint_library = self.world.get_blueprint_library()

        # set the attributes, all values set as strings
        self.upper_fov = args.upper_fov
        self.lower_fov = args.lower_fov
        self.rotation_frequency = args.rotation_frequency
        self.range = args.max_range
        self.num_theta_bin = args.num_theta_bin

        self.dropoff_general_rate = args.dropoff_general_rate
        self.dropoff_intensity_limit = args.dropoff_intensity_limit
        self.dropoff_zero_intensity = args.dropoff_zero_intensity
        self.points_per_second = args.points_per_second

        self.lidar_obj = blueprint_library.find("sensor.lidar.ray_cast")
        self.lidar_obj = self.get_lidar_sensor()
        location = carla.Location(x=1.6, z=1.7)

        self.reward_range = None
        self.metadata = None
        self.lidar_sensor = self.world.try_spawn_actor(
            self.lidar_obj,
            carla.Transform(location, carla.Rotation(yaw=0.0)),
            attach_to=self.vehicle,
        )

        #  dataset
        self.data_path = args.data_path

        #  sync mode
        self.sync_mode = CarlaSyncMode(self.world, self.lidar_sensor, fps=20)

        self._init()
        self.reset_init()  # creates self.agent

        ## Collision detection
        self._proximity_threshold = 10.0
        self._traffic_light_threshold = 5.0
        self.actor_list = self.world.get_actors()
        for idx in range(len(self.actor_list)):
            print(idx, self.actor_list[idx])

        self.vehicle_list = self.actor_list.filter("*vehicle*")
        self.lights_list = self.actor_list.filter("*traffic_light*")
        self.object_list = self.actor_list.filter("*traffic.*")

        ## Initialize the route planner
        self.route_planner.setup()

        ## The map is deterministic so for reward relabelling, we can
        ## instantiate the environment object and then query the distance function
        ## in the env, which directly uses this map_graph, and we need not save it.
        self._map_graph = self.route_planner._graph

        ## This is a dummy for the target location, we can make this an input
        ## to the env in RL code.
        self.target_location = carla.Location(x=-13.473097, y=134.311234, z=-0.010433)

        ## Now reset the env once
        self.reset()

    def get_lidar_sensor(self, role_name="lidar"):  # @
        # set the attributes, all values set as strings
        self.lidar_obj.set_attribute("upper_fov", str(self.upper_fov))
        self.lidar_obj.set_attribute("lower_fov", str(self.lower_fov))
        self.lidar_obj.set_attribute("rotation_frequency", str(self.rotation_frequency))
        self.lidar_obj.set_attribute("range", str(self.range))

        # self.lidar_obj.set_attribute("role_name", role_name)
        self.lidar_obj.set_attribute(
            "dropoff_general_rate", str(self.dropoff_general_rate)
        )
        self.lidar_obj.set_attribute(
            "dropoff_intensity_limit", str(self.dropoff_intensity_limit)
        )
        self.lidar_obj.set_attribute(
            "dropoff_zero_intensity", str(self.dropoff_zero_intensity)
        )
        self.lidar_obj.set_attribute("points_per_second", str(self.points_per_second))

        return self.lidar_obj

    def _init(self):
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
        self.ts = int(time.time())

    def reset(self):
        # get obs:
        obs, _, _, _ = self.step()
        return obs

    def seed(self, seed: int):
        return seed

    def compute_action(self):
        return self.agent.run_step()

    def reset_vehicle(self):
        if self.map.name == "Town04":
            start_x = 5.0
            vehicle_init_transform = carla.Transform(
                carla.Location(x=start_x, y=0, z=0.1), carla.Rotation(yaw=-90)
            )
        else:
            init_transforms = self.world.get_map().get_spawn_points()

            if len(self.route_list) == 0:
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

                goal_candidate = []
                for ts in self.world.get_map().get_spawn_points():
                    if ts.location.distance(vehicle_init_transform.location) > 150:
                        goal_candidate.append(ts)
                try:
                    self.target_location = np.random.choice(goal_candidate).location
                except:
                    for ts in self.world.get_map().get_spawn_points():
                        if ts.location.distance(vehicle_init_transform.location) > 100:
                            goal_candidate.append(ts)
                    self.target_location = np.random.choice(goal_candidate).location

            waypoint_list = self.route_planner.trace_route(
                vehicle_init_transform.location, self.target_location
            )

        # TODO(aviral): start lane not defined for town, also for the town, we may not want to have
        # the lane following reward, so it should be okay.

        if self.vehicle is None:  # then create the ego vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find("vehicle.audi.a2")
            self.vehicle = self.world.spawn_actor(
                vehicle_blueprint, vehicle_init_transform
            )

        self.vehicle.set_transform(vehicle_init_transform)
        self.vehicle.set_target_velocity(carla.Vector3D())
        self.vehicle.set_target_angular_velocity(carla.Vector3D())

        return waypoint_list

    def reset_other_vehicles(self):
        if not self.multiagent:
            return

        # clear out old vehicles
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.vehicles_list]
        )
        self.world.tick()
        # self.sensor.tick()
        self.vehicles_list = []

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]

        num_vehicles = 20
        if self.map.name == "Town04":
            road_id = 47
            road_length = 117.0
            init_transforms = []
            for _ in range(num_vehicles):
                lane_id = random.choice([-1, -2, -3, -4])
                vehicle_s = np.random.uniform(road_length)  # length of road 47
                init_transforms.append(
                    self.map.get_waypoint_xodr(road_id, lane_id, vehicle_s).transform
                )
        else:
            init_transforms = self.world.get_map().get_spawn_points()
            init_transforms = np.random.choice(init_transforms, num_vehicles)

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for transform in init_transforms:
            transform.location.z += (
                0.1  # otherwise can collide with the road it starts on
            )
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
                    carla.command.SetAutopilot(carla.command.FutureActor, True)
                )
            )

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

        traffic_manager.global_percentage_speed_difference(30.0)

    def step(self, action=None, traffic_light_color=""):
        rewards = []
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

    def _is_vehicle_hazard(self, vehicle, vehicle_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
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
                return (True, -1.0, target_vehicle)

        return (False, 0.0, None)

    def _is_object_hazard(self, vehicle, object_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in object_list:
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
                return (True, -1.0, target_vehicle)

        return (False, 0.0, None)

    def _get_trafficlight_trigger_location(
        self, traffic_light
    ):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point, radians):
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

    def _is_light_red(self, vehicle):
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
                    return (True, -0.1, traffic_light)

        return (False, 0.0, None)

    def _get_collision_reward(self, vehicle):
        vehicle_hazard, reward, _ = self._is_vehicle_hazard(
            vehicle, self.vehicle_list
        )
        return vehicle_hazard, reward

    def _get_traffic_light_reward(self, vehicle):
        traffic_light_hazard, _, _ = self._is_light_red(vehicle)
        return traffic_light_hazard, 0.0

    def _get_object_collided_reward(self, vehicle):
        object_hazard, reward, _ = self._is_object_hazard(
            vehicle, self.object_list
        )
        return object_hazard, reward

    def get_distance_vehicle_target(self, vehicle):
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

    def goal_reaching_reward(self, vehicle):
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
        (
            traffic_light_done,
            traffic_light_reward,
        ) = self._get_traffic_light_reward(vehicle)
        (
            object_collided_done,
            object_collided_reward,
        ) = self._get_object_collided_reward(vehicle)
        total_reward = (
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
        }

        return total_reward, reward_dict, done_dict

    def _simulator_step(self, action, traffic_light_color=None):
        raise NotImplementedError

    def finish(self):
        print("destroying actors.")
        for actor in self.actor_list:
            actor.destroy()
        print("\ndestroying %d vehicles" % len(self.vehicles_list))
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.vehicles_list]
        )
        time.sleep(0.5)
        pygame.quit()
        print("done.")

    def get_dataset(self):
        datasets_list = []
        data_path = self.data_path

        if data_path is not None and os.path.exists(data_path):
            for filename in glob.glob(data_path + "/*.pkl"):
                with open(os.path.join(os.getcwd(), filename), "rb") as f:
                    datasets = pkl.load(f)
                    datasets_list.append(datasets)

        return datasets_list
