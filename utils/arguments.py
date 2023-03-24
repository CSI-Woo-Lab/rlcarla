import argparse
import ast
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EnvArguments:
    vision_size: int
    vision_fov: int
    weather: bool
    frame_skip: int
    steps: int
    multiagent: bool
    lane: int
    lights: bool
    route: int
    mode: str
    route_list: List[List[int]]
    random_route: bool

    class_mode: str
    """bc: for behavior_cloning, dc: for data_collection_down"""

    carla_ip: Optional[str]
    data_path: Optional[str]

    upper_fov: float
    lower_fov: float
    rotation_frequency: float
    max_range: float
    num_theta_bin: int
    dropoff_general_rate: float
    dropoff_intensity_limit: float
    dropoff_zero_intensity: float
    points_per_second: float


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
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--multiagent", default=False, action="store_true")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lights", default=False, action="store_true")
    parser.add_argument("--route", type=int, default=0)
    parser.add_argument("--mode", default="ours", type=str)
    parser.add_argument("--route_list", default=[], type=ast.literal_eval)

    parser.add_argument("--random_route", action="store_true")

    parser.add_argument(
        "--class_mode",
        default="bc",
        type=str,
        choices=["bc", "dc"],
        help="bc: for behavior_cloning, dc: for data_collection_town",
    )
    parser.add_argument("--carla_ip", type=str)
    # '/workspace/disk/shared_disk/data/carla/carla-town10hd-route-{}-expert
    #   /carla_expert_embedding_image_{}.pkl'.format(args.route, args.mode)
    parser.add_argument("--data_path", type=str)

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
    return EnvArguments(**vars(args))
