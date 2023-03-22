import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner


class CustomGlobalRoutePlanner(GlobalRoutePlanner):
    def __init__(self, dao):
        super(CustomGlobalRoutePlanner, self).__init__(dao=dao)

    """
    def compute_distance(self, origin, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)
        distance = 0.0
        for idx in range(len(node_list) - 1):
            distance += (super(CustomGlobalRoutePlanner, self)._distance_heuristic(node_list[idx], node_list[idx+1]))
        # print ('Distance: ', distance)
        return distance
    """

    def compute_direction_velocities(self, origin, velocity, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(
            origin=origin, destination=destination
        )

        origin_xy = np.array([origin.x, origin.y])
        velocity_xy = np.array([velocity.x, velocity.y])

        first_node_xy = self._graph.nodes[node_list[1]]["vertex"]
        first_node_xy = np.array([first_node_xy[0], first_node_xy[1]])
        target_direction_vector = first_node_xy - origin_xy
        target_unit_vector = np.array(target_direction_vector) / np.linalg.norm(
            target_direction_vector
        )

        vel_s = np.dot(velocity_xy, target_unit_vector)

        unit_velocity = velocity_xy / (np.linalg.norm(velocity_xy) + 1e-8)
        angle = np.arccos(np.clip(np.dot(unit_velocity, target_unit_vector), -1.0, 1.0))
        vel_perp = np.linalg.norm(velocity_xy) * np.sin(angle)
        return vel_s, vel_perp

    def compute_distance(self, origin, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(
            origin=origin, destination=destination
        )
        # print('Node list:', node_list)
        first_node_xy = self._graph.nodes[node_list[0]]["vertex"]
        # print('Diff:', origin, first_node_xy)

        # distance = 0.0
        distances = []
        distances.append(
            np.linalg.norm(
                np.array([origin.x, origin.y, 0.0]) - np.array(first_node_xy)
            )
        )

        for idx in range(len(node_list) - 1):
            distances.append(
                super(CustomGlobalRoutePlanner, self)._distance_heuristic(
                    node_list[idx], node_list[idx + 1]
                )
            )
        # print('Distances:', distances)
        # import pdb; pdb.set_trace()
        return np.sum(distances)
