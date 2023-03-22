import os
import pickle as pkl

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_dataset(data_path, replay_buffer):
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            datasets = pkl.load(f)

    for i in tqdm.tqdm(range(datasets["observations"]["sensor"].shape[0] - 1)):
        action = np.zeros(2)
        action[0] = datasets["actions"][i][0] - datasets["actions"][i][2]
        action[1] = datasets["actions"][i][1]

        info = datasets["infos"][i]
        info["expert_action"] = action
        image = datasets["observations"]["image"][i]
        next_image = datasets["observations"]["image"][i + 1]
        task = datasets["observations"]["task"][i]
        next_task = datasets["observations"]["task"][i + 1]

        replay_buffer.add(
            obs={
                "image": image,
                "sensor": np.hstack(
                    [
                        datasets["observations"]["sensor"][i][3:9],
                        datasets["observations"]["sensor"][i][18:21],
                    ]
                ),
                "task": task,
            },
            action=action,
            reward=datasets["rewards"][i],
            done=datasets["terminals"][i],
            next_obs={
                "image": next_image,
                "sensor": np.hstack(
                    [
                        datasets["observations"]["sensor"][i + 1][3:9],
                        datasets["observations"]["sensor"][i + 1][18:21],
                    ]
                ),
                "task": next_task,
            },
            infos=[info],
        )


def collect_dataset(env, video_write=False, verbose=False):
    if not os.path.exists(env.record_dir + "/record/{}".format(env.weather)):
        os.mkdir(env.record_dir + "/record/{}".format(env.weather))

    for j in range(120):
        curr_steps = 0
        (
            observations_sensor,
            observations_image,
            observations_task,
            actions,
            rewards,
            terminals,
            infos,
        ) = ([], [], [], [], [], [], [])
        if video_write:
            video = cv2.VideoWriter(
                env.record_dir + "/record//episode_{}.avi".format(j), 0, 20, (224, 224)
            )
            video_segment = cv2.VideoWriter(
                env.record_dir + "/record//episode_{}_segment.avi".format(j),
                0,
                20,
                (224, 224),
            )

        env.reset_init()
        env.reset()

        done = False

        while not done:
            curr_steps += 1
            action, traffic_light_color = env.compute_action()
            next_obs, reward, done, info = env.step(action, None)
            task = np.zeros(12)
            task[env.route] = 1

            if video_write:
                BGR = cv2.cvtColor(next_obs["image"], cv2.COLOR_RGB2BGR)
                video.write(BGR)

                BGR_seg = cv2.cvtColor(info["seg"], cv2.COLOR_RGB2BGR)
                print(BGR_seg.shape)
                video_segment.write(BGR_seg)

            if verbose:
                print(
                    "CurrStep: ",
                    curr_steps,
                    "Reward: ",
                    reward,
                    "Done: ",
                    done,
                    "Location: ",
                    env.vehicle.get_location(),
                    "Target_Location: ",
                    env.target_location,
                )
                print("Goal Distance: ", info["reward_dist"])

            if done and video_write:
                cv2.destroyAllWindows()
                video.release()
                video_segment.release()

            action = np.array([action.throttle, action.steer, action.brake])
            observations_sensor.append(next_obs["sensor"].copy())
            observations_image.append(next_obs["image"].copy())
            observations_task.append(task)
            actions.append(action.copy())
            rewards.append(reward)
            terminals.append(done)
            infos.append(info)

        data_dict = {}
        data_dict["observations"] = {}
        data_dict["observations"]["sensor"] = np.array(observations_sensor)
        data_dict["observations"]["image"] = np.array(observations_image)
        data_dict["observations"]["task"] = np.array(observations_task)
        data_dict["actions"] = np.array(actions)
        data_dict["rewards"] = np.array(rewards)
        data_dict["terminals"] = np.array(terminals)
        data_dict["infos"] = infos

        if verbose:
            print(data_dict["observations"]["sensor"].shape)
            print(data_dict["observations"]["image"].shape)
            print(data_dict["observations"]["task"].shape)
            print(data_dict["actions"].shape)
            print(data_dict["rewards"].shape)
            print(data_dict["terminals"].shape)
            print(data_dict["infos"][-1])

        if info["done_dist_done"]:
            with open(env.record_dir + "/episode_{}.pkl".format(j), "wb") as f:
                pkl.dump(data_dict, f)
