import pickle as pkl
from pathlib import Path
from typing import (Dict, Iterable, List, Optional, Sequence, Tuple, Union,
                    overload)

import cv2
import fire
import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing_extensions import Literal, Required, TypedDict, override

from carla_env.dataset import Dataset

ROUTE_IMAGE = "route_image/Route_point.png"


@overload
def draw_path(
    dataset: Dataset,
    output_filepath: None = None,
    background: Optional[Union[cv2.Mat, np.ndarray]] = None,
) -> cv2.Mat:
    ...

@overload
def draw_path(
    dataset: Dataset,
    output_filepath: Union[str, Path],
    background: Optional[Union[cv2.Mat, np.ndarray]] = None,
) -> None:
    ...


def draw_path(
    dataset: Dataset,
    output_filepath: Optional[Union[str, Path]] = None,
    background: Optional[Union[cv2.Mat, np.ndarray]] = None,
):
    """Draw path of the dataset on the route image.

    The start point is a green circle, the end point is a purple circle, and the path is
    a blue line. The target location is a red circle.

    Args:
        dataset (Dataset): Dataset

    Returns:
        cv2.Mat: Image if output_filepath is None, otherwise None
    
    """
    # Read background image
    if background is not None:
        image = background
    else:
        image = cv2.imread(ROUTE_IMAGE)

    # Read dataset
    observations = dataset["observations"]
    sensor = observations["sensor"]
    lidar_bin = dataset.get("lidar_bin", 80)

    def transform(p: np.ndarray):
        """
        Transform from sensor coordinate to image coordinate
        Sensor coordinate: X - [-125, 120], Y - [-73, 146]
        Image coordinate: X - [0, 1472], Y - [0, 1321]
        Rotation: 0.0125rad, counterclockwise

        """

        # translation
        p[:, 0], p[:, 1] = (
            (p[:, 0] + 125) * 1472 / 245,
            (p[:, 1] + 73) * 1321 / 219,
        )

        # rotation
        p[:, 0], p[:, 1] = (
            p[:, 0] * np.cos(-0.0125) - p[:, 1] * np.sin(-0.0125),
            p[:, 0] * np.sin(-0.0125) + p[:, 1] * np.cos(-0.0125),
        )

        return p

    # Draw path
    offset = lidar_bin + 12
    path = transform(
        sensor[:, offset:(offset + 2)]
    ).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [path], False, (255, 70, 70), 5, lineType=cv2.LINE_AA)

    # Draw start and end point
    start = path[0].reshape(-1)
    end = path[-1].reshape(-1)

    cv2.circle(image, tuple(start), 12, (70, 255, 70), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, tuple(end), 12, (128, 0, 128), -1, lineType=cv2.LINE_AA)

    # Draw target location
    # offset += 6
    # target = transform(
    #     sensor[-1, (offset):(offset + 2)][None, :]
    # ).astype(np.int32).reshape(-1)
    # cv2.circle(image, tuple(target), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # Put text of the reason of done
    info = dataset["infos"][-1]
    reason = ""
    for key, value in info.items():
        if key.startswith("done_") and value:
            reason = key[5:]
            break

    if reason:
        cv2.putText(
            image,
            reason,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    if output_filepath is not None:
        # Save image
        print(f"Saving image to {output_filepath} ...")
        cv2.imwrite(str(output_filepath), image)
    else:
        return image


class PlottingParameter(TypedDict, total=False):
    datasets: Required[Sequence[Dataset]]
    label: str
    color: str


def plot_action_distribution(
    datasets: Iterable[PlottingParameter],
    output_filepath: Optional[Union[str, Path]] = None,
    x_range: Dict[str, Tuple[float, float]] = {},
    y_max: Dict[str, float] = {},
    title_font_size: float = 15,
    y_label_font_size: float = 15,
    sparse: bool = False,
):
    """Plot the action distribution of the dataset as a frequency polygon.
    
    The action distribution is plotted as a frequency polygon, which is a line graph
    that displays the distribution of a continuous variable. The x-axis represents the
    action value and the y-axis represents the frequency of the action value.

    It plots to the current axes if `output_filepath` is None. Otherwise, it saves the
    figure to the `output_filepath`.
    
    """
    if isinstance(output_filepath, Path):
        output_filepath = str(output_filepath)

    labels = [param.get("label", f"Dataset {i}") for i, param in enumerate(datasets)]
    colors = [param.get("color", f"blue") for param in datasets]

    freq: Dict[str, Dict[str, np.ndarray]] = {
        "throttle": {}, "steering": {}, "brake": {}
    }

    for dataset, label in zip(datasets, labels):
        actions = np.concatenate(list(data["actions"] for data in tqdm(
            dataset["datasets"], desc=f"Loading {label} dataset"
        )), axis=0)
        freq["throttle"][label] = actions[:, 0]
        freq["steering"][label] = actions[:, 1]
        freq["brake"][label] = actions[:, 2]

    print("Plotting ...")

    rows = len(labels) if sparse else 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))

    x_range = {
        **{
            "throttle": (0, 1),
            "steering": (-1, 1),
            "brake": (0, 1),
        },
        **x_range,
    }
    x_range = {
        key: (
            max(min(v.min() for v in value.values()), x_min),
            min(max(v.max() for v in value.values()), x_max),
        )
        for (key, value), (x_min, x_max) in zip(freq.items(), x_range.values())
    }
    x_space = {
        key: np.linspace(*r, 1000) for key, r in x_range.items()
    }
    y_values: Dict[str, np.ndarray] = {}
    for key, value in freq.items():
        if key in y_max:
            continue
        y_values[key] = np.zeros(1000)
        for v in value.values():
            try:
                kde = scipy.stats.gaussian_kde(v)(x_space[key])
            except:
                kde = np.zeros(1000)
            y_values[key] += kde

    # Get max of kde without outliers
    y_max = {
        **y_max,
        **{
            key: float(y[np.abs(y - y.mean()) < y.std()].max())
            for key, y in y_values.items()
        },
    }

    y_labels = labels if sparse else ("Density",)
    for i, (dataset_label, color) in enumerate(zip(y_labels, colors)):
        for j, (key, value) in enumerate(freq.items()):
            if sparse:
                ax = axes[i][j]    # type: ignore
            else:
                ax = axes[j]    # type: ignore
            if i == 0:
                ax.set_title(key.capitalize()).set_size(title_font_size)
            if i == len(y_labels) - 1:
                ax.set_xlabel("Action")
            if j == 0:
                ax.set_ylabel(dataset_label.capitalize()).set_size(y_label_font_size)
            else:
                ax.set_ylabel(" ")

            x_min, x_max = x_range[key]

            ax.set_xlim(x_min, x_max)

            ax.set_ylim(0, y_max[key])
            ax.grid(True)

            params = {"ax": ax, "fill": True, "common_norm": False, "alpha": 0.5}
            if sparse:
                sns.kdeplot(data=value[dataset_label], color=color, **params)
            else:
                sns.kdeplot(data=value, **params)

    if output_filepath is None:
        plt.show()
    else:
        fig.savefig(output_filepath)


class DatasetSequence(Sequence[Dataset]):
    def __init__(self, datasets: Iterable[Union[str, Path]], no_validate: bool = False):
        self.__datasets: List[str] = []
        if not no_validate:
            for filename in tqdm(datasets, desc="Validating datasets"):
                try:
                    with open(filename, "rb") as f:
                        pkl.load(f)
                except pkl.UnpicklingError as e:
                    print(f"Failed to load {filename}: {e}")
                else:
                    self.__datasets.append(str(filename))
        else:
            self.__datasets = [str(filename) for filename in datasets]

    @override
    def __len__(self):
        return len(self.__datasets)

    @override
    def __getitem__(self, idx: int) -> Dataset:
        with open(self.__datasets[idx], "rb") as f:
            return pkl.load(f)


class Program:
    def draw_path(self, src: str, dst: Optional[str] = None):
        self.__src = Path(src)
        if self.__src.is_file() and dst is None:
            raise ValueError("dst must be specified when src is a file.")
        self.__dst = Path(dst) if dst is not None else self.__src
        if self.__dst.is_file():
            raise ValueError("dst must be a directory.")

        if self.__src.is_dir():
            for filename in self.__src.glob("*.pkl"):
                try:
                    with open(filename, "rb") as f:
                        dataset = pkl.load(f)
                except pkl.UnpicklingError as e:
                    print(f"Failed to load {filename}: {e}")
                    continue
                draw_path(dataset, output_filepath=self.__dst / f"{filename.stem}.png")
        else:
            with open(self.__src, "rb") as f:
                dataset = pkl.load(f)
            draw_path(dataset, output_filepath=self.__dst / f"{self.__src.stem}.png")

    def plot_action_distribution(
        self,
        *srcs: str,
        dst: Optional[str] = None,
        sparse: bool = False,
    ):
        if len(srcs) % 2 != 0:
            raise ValueError("srcs must be a list of (dataset, label) pairs.")

        if dst is not None:
            self.__dst = Path(dst)
            if self.__dst.is_dir():
                raise ValueError("dst must be a file.")
        else:
            self.__dst = None

        self.__srcs: List[DatasetSequence] = []
        self.__labels: List[str] = []
        for i in range(0, len(srcs), 2):
            label = srcs[i]
            pth = Path(srcs[i + 1])
            dataset = DatasetSequence(pth.glob("**/*.pkl"))

            print(f"Loaded {len(dataset)} datasets for {label} label.")

            self.__srcs.append(dataset)
            self.__labels.append(label)

        plot_action_distribution(
            (
                {"datasets": datasets, "label": label}
                for datasets, label in zip(self.__srcs, self.__labels)
            ),
            output_filepath=self.__dst,
            sparse=sparse,
        )


if __name__ == "__main__":
    fire.Fire(Program)
