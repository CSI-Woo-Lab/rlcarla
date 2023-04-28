from typing import Optional

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.simulator import Simulator
from utils.logger import Logging

logger = Logging.get_logger(__name__)


class Vehicle(Actor[carla.Vehicle]):
    @classmethod
    @override
    async def spawn(
        cls,
        simulator: Simulator,
        blueprint: carla.ActorBlueprint,
        transform: Optional[carla.Transform] = None,
        attach_to: Optional[Actor] = None,
        autopilot: bool = False,
        **kwargs,
    ):
        actor = await cls._try_spawn_actor(
            simulator,
            blueprint,
            transform,
            attach_to,
            autopilot=autopilot,
            **kwargs,
        )
        return cls(simulator, actor) if actor else None

    @staticmethod
    def _create_spawn_command(
        blueprint: carla.ActorBlueprint,
        transform: carla.Transform,
        attach_to: Optional[Actor] = None,
        autopilot: bool = False,
        **kwargs,
    ):
        cmd = Actor._create_spawn_command(blueprint, transform, attach_to, **kwargs)
        if autopilot:
            cmd = cmd.then(
                carla.command.SetAutopilot(carla.command.FutureActor, autopilot)
            )
        return cmd

    def apply_control(self, control: carla.VehicleControl):
        self.carla.apply_control(control)

    def stop(self):
        self.velocity = carla.Vector3D(0, 0, 0)
        self.angular_velocity = carla.Vector3D(0, 0, 0)

    @property
    def velocity(self) -> carla.Vector3D:
        return super().velocity

    @velocity.setter
    def velocity(self, velocity: carla.Vector3D):
        self.carla.set_target_velocity(velocity)

    @property
    def angular_velocity(self) -> carla.Vector3D:
        return super().angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity: carla.Vector3D):
        self.carla.set_target_angular_velocity(angular_velocity)
