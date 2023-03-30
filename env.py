import logging
import os
from typing import Any, Callable, Generic, TypeVar

import flax
import flax.linen as nn
from flax.struct import dataclass, field
from typing_extensions import Concatenate, ParamSpec

from carla_env.behavior_cloning import behavior_cloning
from carla_env.collect_data import collect_data
from utils.arguments import parse_args
from utils.logger import Logging

P = ParamSpec("P")
R = TypeVar("R")
Params = flax.core.FrozenDict[str, Any]


@dataclass
class Model(Generic[P, R]):
    step: int
    apply_fn: Callable[Concatenate[Params, P], R] = field(pytree_node=False)
    params: Params

    @classmethod
    def create(cls, model_def: nn.Module, params: Params) -> "Model":
        return cls(step=1, apply_fn=model_def.apply, params=params)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        # pylint: disable=not-callable
        return self.apply_fn(self.params, *args, **kwargs)


def main():
    args = parse_args()

    logging_path = os.path.join(args.data_path or os.getcwd(), "outputs.log")
    print("Logging to", logging_path)
    Logging.setup(
        filepath=logging_path,
        level=logging.INFO,
        formatter="(%(asctime)s) [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.class_mode == "bc":
        behavior_cloning(args)
    elif args.class_mode == "dc":
        collect_data(args)


if __name__ == "__main__":
    main()
