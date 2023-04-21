from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import yaml


@dataclass
class ExperimentConfigs:
    """Arguments for running env.py."""

    vision_size: int
    """Size of the vision sensor"""

    vision_fov: int
    """Field of view of the vision sensor"""

    weather: bool
    frame_skip: int
    multiagent: bool
    lane: int
    lights: bool
    mode: str

    num_routes: int
    """Number of routes to use"""

    routes: List[Tuple[int, int]]
    """List of routes to use"""

    vehicle_type: str
    """Type of vehicle to use. Example: audi.a2"""

    random_route: bool
    """Whether to use random route"""

    max_steps: int
    """Maximum number of steps per episode"""

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


def check_route_list(routes: Any) -> List[Tuple[int, int]]:
    """Check if the route list is valid.

    Args:
        route_str (str): String of routes

    Returns:
        List[List[int]]: List of routes

    Raises:
        ValueError: If the route string is not valid
    """
    if not isinstance(routes, list):
        raise ValueError("Invalid route list. Must be a list of lists or tuples.")
    for route in routes:
        if not isinstance(route, (list, tuple)):
            raise ValueError("Invalid route list. Must be a list of lists or tuples.")
        if len(route) != 2:
            raise ValueError(
                "Invalid route list. Each element must be a list or tuple of length 2."
            )
        for route_id in route:
            if not isinstance(route_id, int):
                raise ValueError("Invalid route list. Each element must be an integer.")
    return routes


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


def parse_config(filename: Union[str, Path]):
    """Parse arguments.

    Return:
        EnvArguments: Arguments for running env.py
    """
    with open(filename, "r") as f:
        config = yaml.safe_load(f)

        # Check if the config is valid
        if "carla_ip" in config:
            config["carla_ip"] = check_is_ip(config["carla_ip"])
        if "data_path" in config and config["data_path"] is not None:
            config["data_path"] = Path(config["data_path"])
        if "routes" in config:
            config["routes"] = check_route_list(config["routes"])
        return ExperimentConfigs(**config)
