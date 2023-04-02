from pathlib import Path
from typing import Optional, Union, overload

import cv2
import numpy as np

from carla_env.dataset import Dataset

ROUTE_IMAGE = "route_image/Route_point.png"


@overload
def draw_path(dataset: Dataset) -> cv2.Mat:
    ...

@overload
def draw_path(dataset: Dataset, output_filepath: Union[str, Path]) -> None:
    ...


def draw_path(dataset: Dataset, output_filepath: Optional[Union[str, Path]] = None):
    """Draw path of the dataset on the route image.

    The start point is a green circle, the end point is a purple circle, and the path is
    a blue line. The target location is a red circle.

    Args:
        dataset (Dataset): Dataset

    Returns:
        cv2.Mat: Image if output_filepath is None, otherwise None
    
    """
    # Read background image
    image = cv2.imread(ROUTE_IMAGE)

    # Read dataset
    observations = dataset["observations"]
    sensor = observations["sensor"]
    lidar_bin = 80
    offset = lidar_bin + 9

    def transform(p: np.ndarray):
        # Transform from sensor coordinate to image coordinate
        # Sensor coordinate: X - [-122, 115], Y - [-73, 140]
        # Image coordinate: X - [0, 1472], Y - [0, 1321]
        p[:, 0] = (p[:, 0] + 122) * 1472 / 237
        p[:, 1] = (p[:, 1] + 73) * 1321 / 213
        return p

    # Draw path
    path = transform(
        sensor[:, offset:(offset + 2)]
    ).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [path], False, (255, 0, 0), 5, lineType=cv2.LINE_AA)

    # Draw start and end point
    start = path[0].reshape(-1)
    end = path[-1].reshape(-1)

    cv2.circle(image, tuple(start), 12, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, tuple(end), 12, (128, 0, 128), -1, lineType=cv2.LINE_AA)

    # Draw target location
    target = transform(
        sensor[-1, (offset + 12):(offset + 14)][None, :]
    ).astype(np.int32).reshape(-1)
    cv2.circle(image, tuple(target), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)

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