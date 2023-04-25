import random

import carla
import numpy as np

from carla_env.simulator.simulator import Simulator
from carla_env.simulator.vehicles.vehicle import Vehicle


class AutoVehicle(Vehicle):
    _LANE_ID_CANDIDATES = [-1, -2, -3, -4]

    def __init__(self, simulator: Simulator):
        super().__init__(simulator=simulator, blueprint=self.__get_blueprint())

        self.__spawn_points = self.world.map.get_spawn_points()

        if self.world.map.name == "Town04":
            self.__get_initial_transform = self.__town04__get_initial_transform
        else:
            self.__get_initial_transform = self.__default__get_initial_transform

    def spawn(self):
        super().spawn(self.__get_initial_transform(), autopilot=True)

    def __town04__get_initial_transform(self) -> carla.Transform:
        road_id = 47
        road_length = 117.
        initial_transform = self.world.map.get_waypoint_xodr(
            road_id=road_id,
            lane_id=random.choice(self._LANE_ID_CANDIDATES),
            s=np.random.random.uniform(road_length),
        ).transform
        return initial_transform

    def __default__get_initial_transform(self) -> carla.Transform:
        return random.choice(self.__spawn_points)

    @staticmethod
    def __blueprint_filter(blueprint: carla.ActorBlueprint) -> bool:
        """Filter the blueprints of the vehicles."""
        return int(blueprint.get_attribute("number_of_wheels")) == 4

    def __get_blueprint(self) -> carla.ActorBlueprint:
        """Get a random blueprint of the vehicle."""
        blueprints = list(filter(
            self.__blueprint_filter,
            self.world.blueprint_library.filter("vehicle.*")
        ))
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

        return blueprint
