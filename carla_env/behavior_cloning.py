import glob
import os
from typing import Any

import flax
import numpy as np
import tqdm

from carla_env import BCCarlaEnv
from carla_env.dataset import load_dataset
from carla_env.weathers import WEATHERS
from offline_baselines_jax.bc.bc import BC
from offline_baselines_jax.bc.policies import MultiInputPolicy
from utils.arguments import EnvArguments

Params = flax.core.FrozenDict[str, Any]


def behavior_cloning(args: EnvArguments):
    if args.carla_ip is None:
        print("Please pass your carla IP address")
        return

    # Resnet18, variables = pretrained_resnet(18)
    # model_def = nn.Sequential(Resnet18().layers[:11])
    # variables = slice_variables(variables, 0, 11)
    # image_model = Model.create(model_def, params=variables)
    data_path = args.data_path
    if data_path is not None and os.path.exists(data_path):
        datasets = [
            load_dataset(filename)
            for filename in glob.glob(os.path.join(data_path + "*.pkl"))
        ]
    else:
        datasets = None

    env = BCCarlaEnv(
        args=args,
        image_model=None,
        weather=WEATHERS[0],
        carla_ip=args.carla_ip,
        carla_port=2000 - args.route * 5,
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
                action = np.zeros(2)
                action[0] = dataset["actions"][i][0] - dataset["actions"][i][2]
                action[1] = dataset["actions"][i][1]
                info = dataset["infos"][i]
                info["expert_action"] = action
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
        model.save(os.path.join("models", f"{args.mode}_model_route_{args.route}_{i}"))
