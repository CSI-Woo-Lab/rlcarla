from typing import (Callable, Generic, List, Optional, TypeVar, Union, cast,
                    overload)

import carla

from carla_env.simulator.client import Client
from carla_env.simulator.command import CarlaCommand
from carla_env.simulator.simulator import Simulator
from carla_env.simulator.world import World
from utils.logger import Logging

logger = Logging.get_logger(__name__)

T = TypeVar("T", bound=carla.Actor, covariant=True)


class Actor(Generic[T]):
    @overload
    def __init__(self, simulator: Simulator, blueprint: str):
        ...

    @overload
    def __init__(self, simulator: Simulator, blueprint: carla.ActorBlueprint):
        ...

    def __init__(
        self, simulator: Simulator, blueprint: Union[str, carla.ActorBlueprint]
    ) -> None:
        self.__simulator = simulator
        if isinstance(blueprint, str):
            self.__blueprint = self.world.blueprint_library.find(blueprint)
        else:
            self.__blueprint = blueprint

        self.__actor: Optional[T] = None
        self.__on_create_callbacks: List[Callable[["Actor"], None]] = []
        self.__on_destroy_callbacks: List[Callable[["Actor"], None]] = []

    def spawn(self, transform: carla.Transform, attach_to: Optional["Actor"] = None):
        if attach_to:
            cmd = carla.command.SpawnActor(
                self.__blueprint, transform, parent=attach_to.actor
            )
        else:
            cmd = carla.command.SpawnActor(self.__blueprint, transform)
        self.client.enqueue_command(cmd, self._on_spawn)

    def _on_spawn(self, response: carla.command.Response) -> None:
        self.__actor = cast(T, self.world.get_actor(response.actor_id))
        for callback in self.__on_create_callbacks:
            callback(self)
        logger.debug("Spawn %s", self.__actor)

    def on_create(self, callback: Callable[["Actor"], None]) -> None:
        """Add a callback function to be called when the actor is created."""
        self.__on_create_callbacks.append(callback)

    @property
    def simulator(self) -> Simulator:
        """The simulator of the actor."""
        return self.__simulator

    @property
    def client(self) -> Client:
        """The client of the actor."""
        return self.simulator.client

    @property
    def world(self) -> World:
        """The world of the actor."""
        return self.simulator.world

    @property
    def actor(self) -> T:
        """The base carla actor object of the simulator."""
        if self.__actor is None:
            raise RuntimeError("The actor is not spawned yet.")
        return self.__actor

    @property
    def blueprint(self) -> carla.ActorBlueprint:
        """The blueprint of the actor."""
        return self.__blueprint

    @property
    def transform(self) -> carla.Transform:
        """The transform of the actor."""
        return self.actor.get_transform()

    @transform.setter
    def transform(self, transform: carla.Transform) -> None:
        self.actor.set_transform(transform)

    @property
    def location(self) -> carla.Location:
        """The location of the actor."""
        return self.transform.location

    @location.setter
    def location(self, location: carla.Location) -> None:
        self.actor.set_location(location)

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
        return self.actor.get_velocity()

    @velocity.setter
    def velocity(self, velocity: carla.Vector3D) -> None:
        self.actor.set_target_velocity(velocity)

    @property
    def angular_velocity(self) -> carla.Vector3D:
        """The angular velocity of the actor."""
        return self.actor.get_angular_velocity()

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity: carla.Vector3D) -> None:
        self.actor.set_target_angular_velocity(angular_velocity)

    @property
    def acceleration(self) -> carla.Vector3D:
        """The acceleration of the actor."""
        return self.actor.get_acceleration()

    def add_force(self, force: carla.Vector3D) -> None:
        """Add a force to the actor."""
        self.actor.add_force(force)

    def add_torque(self, torque: carla.Vector3D) -> None:
        """Add a torque to the actor."""
        self.actor.add_torque(torque)

    def add_impulse(self, impulse: carla.Vector3D) -> None:
        """Add an impulse to the actor."""
        self.actor.add_impulse(impulse)

    @property
    def is_spawned(self) -> bool:
        """Whether the actor is spawned."""
        return self.__actor is not None

    @property
    def is_alive(self) -> bool:
        """Whether the actor is alive."""
        return self.__actor is not None and self.__actor.is_alive

    def destroy(self) -> None:
        """Destroy the actor."""
        for callback in self.__on_destroy_callbacks:
            callback(self)
        self.simulator.client.enqueue_command(carla.command.DestroyActor(self.actor))

    def on_destroy(self, callback: Callable[["Actor"], None]) -> None:
        """Add a callback when the actor is destroyed."""
        self.__on_destroy_callbacks.append(callback)

    def __del__(self) -> None:
        self.destroy()
