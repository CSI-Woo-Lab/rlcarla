
import random
from typing import List, Optional, Tuple

import carla

from carla_env.simulator.world import World
from carla_env.utils import build_goal_candidate


class RouteSelector:
    _DEFAULT_ROUTE_CANDIDATES: List[Tuple[int, int]] = [
        (1, 37),
        (0, 87),
        (21, 128),
        (0, 152),
        (21, 123),
        (21, 6),
        (126, 4),
        (135, 37),
        (126, 143),
        (129, 119),
        (129, 60),
        (129, 146),
    ]

    def __init__(
        self,
        world: World,
        route_candidates: Optional[List[Tuple[int, int]]],
        random_route: bool = False,
    ):
        self.__world = world
        self.__map = world.map
        self.__spawn_points = self.__map.get_spawn_points()

        if not route_candidates:
            route_candidates = self._DEFAULT_ROUTE_CANDIDATES
        self.__route_list = route_candidates
        self.__current_route_idx = 0
        self.__random_route = random_route

        if self.__map.name == "Town04":
            self.__get_next_route = self.__town04__get_next_route
        elif self.is_random:
            self.__get_next_route = self.__random__get_next_route
        else:
            self.__get_next_route = self.__default__get_next_route

    def __town04__get_next_route(
        self, start_x: float = 5.
    ) -> Tuple[carla.Transform, carla.Transform]:
        initial_transform = carla.Transform(
            carla.Location(x=start_x, y=.0, z=.1), carla.Rotation(yaw=90.)
        )
        return initial_transform, initial_transform

    def __random__get_next_route(self) -> Tuple[carla.Transform, carla.Transform]:
        initial_transform = random.choice(self.__spawn_points)
        goal_candidate = build_goal_candidate(
            self.__world, initial_transform.location
        )
        if not goal_candidate:
            goal_candidate = build_goal_candidate(
                self.__world, initial_transform.location, threshold=100.
            )
        target_transform = random.choice(goal_candidate)
        return initial_transform, target_transform
    
    def __default__get_next_route(self) -> Tuple[carla.Transform, carla.Transform]:
        self.__current_route_idx += 1
        self.__current_route_idx %= len(self.__route_list)

        current_route = self.__route_list[self.__current_route_idx]
        initial_transform = self.__spawn_points[current_route[0]]
        target_transform = self.__spawn_points[current_route[1]]
        return initial_transform, target_transform
    
    def next(self):
        return self.__get_next_route()

    @property
    def is_random(self):
        return self.__random_route
