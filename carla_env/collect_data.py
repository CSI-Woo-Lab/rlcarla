"""python collect_data.py"""

import os
from typing import Any, List

import flax
import numpy as np

from carla_env import DCCarlaEnv
from carla_env.dataset import Dataset, dump_dataset
from carla_env.weathers import WEATHERS
from utils.arguments import EnvArguments

Params = flax.core.FrozenDict[str, Any]


def collect_data(args: EnvArguments):
    # example call:
    # ./PythonAPI/util/config.py --map Town01 --delta-seconds 0.05
    # python PythonAPI/carla/agents/navigation/data_collection_agent.py \
    #   --vision_size 256 --vision_fov 90 --steps 10000 --weather --lights
    if args.carla_ip is None:
        print("Please pass your carla IP address")
        return

    env = DCCarlaEnv(
        args=args,
        image_model=None,
        weather=WEATHERS[0],
        carla_ip=args.carla_ip,
        carla_port=2000 - args.route * 5,
    )

    curr_steps = 0
    # for weather in weather_list:
    weather = "ClearNoon"
    env.weather = "ClearNoon"

    record_dirname_per_weather = os.path.join(env.record_dir, "record", weather)
    if not os.path.exists(record_dirname_per_weather):
        os.mkdir(record_dirname_per_weather)

    total_step = 0
    for j in range(12000):
        curr_steps = 0
        observations_sensor: List[np.ndarray] = []
        observations_image: List[np.ndarray] = []
        observations_task: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[np.ndarray] = []
        terminals: List[bool] = []
        infos: List[dict] = []

        env.reset_init()
        env.reset()
        done = False
        while not done:
            curr_steps += 1
            action, _ = env.compute_action()
            next_obs, reward, done, info = env.step(action)
            task = np.zeros(12)
            task[env.route] = 1

            action = np.array([action.throttle, action.steer, action.brake])
            observations_sensor.append(next_obs["sensor"].copy())
            observations_task.append(task)
            actions.append(action.copy())
            rewards.append(reward)
            terminals.append(done)
            infos.append(info)

        total_step += curr_steps

        print("Total step: ", total_step)
        if total_step > 1_000_000:
            print("Data collection complete!")
            break

        dataset: Dataset = {
            "observations": {
                "sensor": np.array(observations_sensor),
                "image": np.array(observations_image),
                "task": np.array(observations_task),
            },
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "terminals": np.array(terminals),
            "infos": infos,
        }

        print("  Sensor shape =", dataset["observations"]["sensor"].shape)
        print("   Image shape =", dataset["observations"]["image"].shape)
        print("    Task shape =", dataset["observations"]["task"].shape)
        print("  Action shape =", dataset["actions"].shape)
        print(" Rewards shape =", dataset["rewards"].shape)
        print("Terminal shape =", dataset["terminals"].shape)
        print(dataset["infos"][-1])

        if infos[-1]["done_dist_done"]:
            filename = os.path.join(env.record_dir, f"episode_{j}.pkl")
            dump_dataset(dataset, filename)
