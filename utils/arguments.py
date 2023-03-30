import argparse
import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ClassMode(Enum):
    """Class mode for data collection."""

    BEHAVIOR_CLONING = "bc"
    DATA_COLLECTION = "dc"


@dataclass
class ExperimentArguments:
    """Arguments for running env.py."""

    vision_size: int
    vision_fov: int
    weather: bool
    frame_skip: int
    multiagent: bool
    lane: int
    lights: bool
    mode: str

    num_routes: int
    """Number of routes to use"""

    route_list: List[List[int]]
    """List of routes to use"""

    random_route: bool
    """Whether to use random route"""

    max_steps: int
    """Maximum number of steps per episode"""

    class_mode: str
    """bc: for behavior_cloning, dc: for data_collection_down"""

    carla_ip: Optional[str]
    """IP address of the carla server"""

    data_path: Optional[Path]
    """Path to the data directory to save the episode data and logs"""

    upper_fov: float
    lower_fov: float
    rotation_frequency: float
    max_range: float
    num_theta_bin: int
    dropoff_general_rate: float
    dropoff_intensity_limit: float
    dropoff_zero_intensity: float
    points_per_second: float


def make_route_list_from_str(route_str: str) -> List[List[int]]:
    """Make route list from string.

    Args:
        route_str (str): String of routes

    Returns:
        List[List[int]]: List of routes

    Raises:
        ValueError: If the route string is not valid
    """
    route_list = ast.literal_eval(route_str)
    if not isinstance(route_list, list):
        raise ValueError("Invalid route list")
    for route in route_list:
        if not isinstance(route, list):
            raise ValueError("Invalid route list")
        for route_id in route:
            if not isinstance(route_id, int):
                raise ValueError("Invalid route list")
    return route_list


def check_is_ip(ip: str) -> str:
    """Check if the ip is valid.

    Args:
        ip (str): IP address

    Returns:
        str: Returns the ip string if the ip is valid, raises exception otherwise

    Raises:
        ValueError: If the ip is not valid
    """
    def checker(ip: str) -> bool:
        if ip == "localhost":
            return True
        ip_split = ip.split(".")
        if len(ip_split) != 4:
            return False
        for i in ip_split:
            if not i.isdigit():
                return False
            i = int(i)
            if i < 0 or i > 255:
                return False
        return True

    if not checker(ip):
        raise ValueError("Invalid IP address")
    return ip


def parse_args():
    """Parse arguments.

    Return:
        EnvArguments: Arguments for running env.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_size", type=int, default=224)
    parser.add_argument("--vision_fov", type=int, default=90)
    parser.add_argument("--weather", default=False, action="store_true")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000, dest="max_steps")
    parser.add_argument("--multiagent", default=False, action="store_true")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lights", default=False, action="store_true")
    parser.add_argument("--route", type=int, default=0, dest="num_routes")
    parser.add_argument("--mode", default="ours", type=str)
    parser.add_argument("--route_list", default=[], type=make_route_list_from_str)

    parser.add_argument("--random_route", action="store_true")

    parser.add_argument(
        "--class_mode",
        default=ClassMode.BEHAVIOR_CLONING,
        type=ClassMode,
        help="bc: for behavior_cloning, dc: for data_collection_town",
    )
    parser.add_argument("--carla_ip", type=check_is_ip)
    parser.add_argument("--data_path", type=Path)

    # Lidar
    parser.add_argument("--num_theta_bin", type=int, default=80)

    parser.add_argument("--upper_fov", type=float, default=5.0)
    parser.add_argument("--lower_fov", type=float, default=-30.0)
    parser.add_argument("--rotation_frequency", type=float, default=20.0)
    parser.add_argument(
        "--max_range",
        type=float,
        default=20.0,
        help="Maximum distance to measure/raycast in meters",
    )

    parser.add_argument("--dropoff_general_rate", type=float, default=0.1)
    parser.add_argument("--dropoff_intensity_limit", type=float, default=0.2)
    parser.add_argument("--dropoff_zero_intensity", type=float, default=0.2)
    parser.add_argument("--points_per_second", type=int, default=120000)

    args = parser.parse_args()
    return ExperimentArguments(**vars(args))
