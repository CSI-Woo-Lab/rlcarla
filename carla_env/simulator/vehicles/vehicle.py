from typing import Optional

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor


class Vehicle(Actor[carla.Vehicle]):
    @override
    def spawn(
        self,
        transform: carla.Transform,
        attach_to: Optional[Actor] = None,
        autopilot: bool = False,
    ):
        if attach_to:
            cmd = carla.command.SpawnActor(
                self.blueprint, transform, parent=attach_to.actor
            )
        else:
            cmd = carla.command.SpawnActor(self.blueprint, transform)
        
        if autopilot:
            cmd = cmd.then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)
            )

        self.simulator.client.enqueue_command(cmd, self._on_spawn)

    def apply_control(self, control: carla.VehicleControl):
        self.actor.apply_control(control)
