from typing import Union

import carla
import numpy as np


def to_array(
    array_like: Union[carla.Vector3D, carla.Rotation]
):
    if isinstance(array_like, carla.Vector3D):
        return vector_to_array(array_like)

    if isinstance(array_like, carla.Rotation):
        return rotation_to_array(array_like)

    raise TypeError(
        f"Expected array-like object, got {type(array_like)}"
    )


def vector_to_array(vector: carla.Vector3D):
    return np.array([vector.x, vector.y, vector.z])


def rotation_to_array(rotation: carla.Rotation):
    return np.array([rotation.pitch, rotation.yaw, rotation.roll])
