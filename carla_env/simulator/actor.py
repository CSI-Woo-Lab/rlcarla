import asyncio
from typing import Callable, Generic, List, Optional, Type, TypeVar, cast

import carla
from typing_extensions import TypeGuard, override

from carla_env.simulator.carla_wrapper import CarlaWrapper
from carla_env.simulator.simulator import Simulator
from utils.logger import Logging

logger = Logging.get_logger(__name__)

T = TypeVar("T", bound=carla.Actor)


class Actor(Generic[T], CarlaWrapper[T]):
    def __init__(self, simulator: Simulator, actor: T):
        super().__init__(actor)
        self.__simulator = simulator
        self.__on_destroy_callbacks: List[Callable[[], None]] = []
        self.__destroyed = False

    @classmethod
    async def spawn(
        cls,
        simulator: Simulator,
        blueprint: carla.ActorBlueprint,
        transform: Optional[carla.Transform] = None,
        attach_to: Optional["Actor"] = None,
        **kwargs,
    ):
        actor = await cls._try_spawn_actor(
            simulator, blueprint, transform, attach_to, **kwargs
        )
        return cls(simulator, actor) if actor else None

    @classmethod
    async def _try_spawn_actor(
        cls,
        simulator: Simulator,
        blueprint: carla.ActorBlueprint,
        transform: Optional[carla.Transform] = None,
        attach_to: Optional["Actor"] = None,
        **kwargs,
    ):
        actor: List[Optional[T]] = []

        def on_spawn(response: carla.command.Response) -> None:
            if response.has_error():
                msg = "Failed to spawn %s: %s" % (blueprint, response.error)
                logger.error(msg)
                actor.append(None)
            else:
                spawned = cast(T, simulator.world.carla.get_actor(response.actor_id))
                logger.info("Spawn %s", spawned)
                actor.append(spawned)

        simulator.client.enqueue_command(
            cls._create_spawn_command(blueprint, transform, attach_to, **kwargs),
            on_spawn,
        )

        while not actor:
            await asyncio.sleep(0.01)
        return actor[0]

    @staticmethod
    def _create_spawn_command(
        blueprint: carla.ActorBlueprint,
        transform: Optional[carla.Transform] = None,
        attach_to: Optional["Actor"] = None,
        **kwargs,
    ):
        if not transform:
            transform = carla.Transform(
                carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0, pitch=0, roll=0)
            )
        if attach_to:
            return carla.command.SpawnActor(
                blueprint, transform, parent=attach_to.carla, **kwargs
            )
        else:
            return carla.command.SpawnActor(blueprint, transform, **kwargs)

    @property
    def simulator(self):
        """The simulator of the actor."""
        return self.__simulator

    @property
    def client(self):
        """The client of the actor."""
        return self.simulator.client

    @property
    def world(self):
        """The world of the actor."""
        return self.simulator.world

    @property
    def is_alive(self) -> bool:
        """Whether the actor is alive."""
        return self.carla.is_alive

    def on_destroy(self, callback: Callable[[], None]) -> None:
        """Add a callback when the actor is destroyed."""
        self.__on_destroy_callbacks.append(callback)

    async def destroy(self):
        if not self.is_alive or self.__destroyed:
            return

        self.__destroyed = True
        actor_desc = str(self.carla)

        done = []

        def on_destroy(response: carla.command.Response) -> None:
            if response.has_error():
                msg = "Failed to destroy %s: %s" % (actor_desc, response.error)
                logger.error(msg)
                done.append(False)
            else:
                logger.info("Destroy %s", actor_desc)
                for callback in self.__on_destroy_callbacks:
                    callback()
                done.append(True)

        self.simulator.client.enqueue_command(
            carla.command.DestroyActor(self.carla), on_destroy
        )

        while not done:
            await asyncio.sleep(0.01)
        return done[0]

    def __del__(self):
        if not self.__destroyed:
            print(f"Warning: Actor {self.carla} is not destroyed before deletion")
            asyncio.run(self.destroy())

    @property
    def transform(self) -> carla.Transform:
        """The transform of the actor."""
        return self.carla.get_transform()

    @transform.setter
    def transform(self, transform: carla.Transform) -> None:
        self.carla.set_transform(transform)

    @property
    def location(self) -> carla.Location:
        """The location of the actor."""
        return self.transform.location

    @location.setter
    def location(self, location: carla.Location) -> None:
        self.carla.set_location(location)

    @property
    def rotation(self) -> carla.Rotation:
        """The rotation of the actor."""
        return self.transform.rotation
    
    @rotation.setter
    def rotation(self, rotation: carla.Rotation) -> None:
        self.transform = carla.Transform(self.transform.location, rotation)

    @property
    def velocity(self) -> carla.Vector3D:
        """The velocity of the actor."""
        return self.carla.get_velocity()

    @property
    def angular_velocity(self) -> carla.Vector3D:
        """The angular velocity of the actor."""
        return self.carla.get_angular_velocity()

    @property
    def acceleration(self) -> carla.Vector3D:
        """The acceleration of the actor."""
        return self.carla.get_acceleration()

    def add_force(self, force: carla.Vector3D) -> None:
        """Add a force to the actor."""
        self.carla.add_force(force)

    def add_torque(self, torque: carla.Vector3D) -> None:
        """Add a torque to the actor."""
        self.carla.add_torque(torque)

    def add_impulse(self, impulse: carla.Vector3D) -> None:
        """Add an impulse to the actor."""
        self.carla.add_impulse(impulse)

    def isinstance(self, _type: Type[T]) -> TypeGuard["Actor[T]"]:
        return isinstance(self.carla, _type)

    @override
    def __repr__(self):
        return f"{self.carla}"
