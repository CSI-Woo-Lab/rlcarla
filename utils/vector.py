import carla
import numpy as np


def vector_to_array(vector: carla.Vector3D):
    return np.array([vector.x, vector.y, vector.z])


def rotation_to_array(rotation: carla.Rotation):
    return np.array([rotation.pitch, rotation.yaw, rotation.roll])
