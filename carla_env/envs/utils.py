import carla


def build_goal_candidate(
    world: carla.World,
    target_location: carla.Location,
    threshold: float = 150,
):
    return [
        ts
        for ts in world.get_map().get_spawn_points()
        if ts.location.distance(target_location) > threshold
    ]
