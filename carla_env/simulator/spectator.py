from enum import Enum

import carla

from carla_env.simulator.actor import Actor
from carla_env.simulator.carla_wrapper import CarlaWrapper


class Spectator(CarlaWrapper[carla.Actor]):
    class FollowMode(Enum):
        BEHIND = 0
        ABOVE = 1
        INSIDE = 2

    def follow(
        self, actor: Actor, mode: FollowMode = FollowMode.ABOVE, cascade: bool = True
    ):
        world = self.carla.get_world()

        if mode == Spectator.FollowMode.BEHIND:
            def transform():
                return carla.Transform(
                    actor.transform.location + carla.Location(x=-50),
                    carla.Rotation(pitch=-15),
                )
        elif mode == Spectator.FollowMode.ABOVE:
            def transform():
                return carla.Transform(
                    actor.transform.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90),
                )
        elif mode == Spectator.FollowMode.INSIDE:
            # Get height of the actor
            height = actor.carla.bounding_box.extent.z * 2

            def transform():
                return carla.Transform(
                    actor.transform.location + carla.Location(z=height * 0.8),
                    carla.Rotation(pitch=-5, yaw=actor.transform.rotation.yaw),
                )
        callback_id = world.on_tick(lambda _: self.carla.set_transform(transform()))

        if cascade:
            actor.on_destroy(lambda: world.remove_on_tick(callback_id))
