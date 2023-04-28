import carla

from carla_env.simulator.world import World


def build_goal_candidate(
    world: World,
    target_location: carla.Location,
    threshold: float = 150,
):
    return [
        ts
        for ts in world.map.get_spawn_points()
        if ts.location.distance(target_location) > threshold
    ]
