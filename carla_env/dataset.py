import os
import pickle as pkl
from typing import List

import numpy as np
from typing_extensions import TypedDict


class Observations(TypedDict):
    sensor: np.ndarray
    image: np.ndarray
    task: np.ndarray


class Dataset(TypedDict):
    observations: Observations
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    infos: List[dict]


def load_dataset(filename: str) -> Dataset:
    with open(os.path.join(os.getcwd(), filename), "rb") as f:
        dataset = pkl.load(f)
    return dataset


def dump_dataset(dataset: Dataset, filename: str):
    with open(filename, "wb") as f:
        pkl.dump(dataset, f)
