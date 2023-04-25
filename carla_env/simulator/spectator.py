from enum import Enum

import carla

from carla_env.simulator.actor import Actor


class Spectator(carla.Actor):
    class FollowMode(Enum):
        BEHIND = 0
        ABOVE = 1
        INSIDE = 2

    def __new__(cls, actor: carla.Actor):
        return actor

    def follow(
        self, actor: Actor, mode: FollowMode = FollowMode.ABOVE, cascade: bool = True
    ):
        world = self.get_world()

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
            def transform():
                return carla.Transform(
                    actor.transform.location + carla.Location(z=5),
                    carla.Rotation(pitch=-5),
                )
        callback_id = world.on_tick(lambda _: self.set_transform(transform()))

        if cascade:
            actor.on_destroy(lambda _: world.remove_on_tick(callback_id))
