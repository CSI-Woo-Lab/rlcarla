from typing import Any, Callable, Sequence

import flax
import flax.linen as nn
import jax
import numpy as np
from jax_resnet import pretrained_resnet, slice_variables

Params = flax.core.FrozenDict[str, Any]


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params

    @classmethod
    def create(cls, model_def: nn.Module, params: Params) -> "Model":
        return cls(step=1, apply_fn=model_def.apply, params=params)

    def __call__(self, *args, **kwargs):
        return self.apply_fn(self.params, *args, **kwargs)


class ResnetExtractor(nn.Module):
    layers: Sequence[Callable[..., Any]]

    @nn.compact
    def __call__(self, *args, **kwargs):
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")

        outputs = self.layers[0](*args, **kwargs)
        for layer in self.layers[1:]:
            outputs = layer(outputs)
        return outputs


@jax.jit
def resnet_image_embedding(model, input_image):
    return model(input_image, mutable=False)


def resnet_image_preprocess(image):
    image = image.astype(np.float) / 255
    image[..., 0] = (image[..., 0] - 0.485) / 0.229
    image[..., 1] = (image[..., 1] - 0.456) / 0.224
    image[..., 2] = (image[..., 2] - 0.406) / 0.225
    return image


class ImageModel(object):
    def __init__(self):
        Resnet18, variables = pretrained_resnet(18)
        model_def = nn.Sequential(Resnet18().layers[:11])
        variables = slice_variables(variables, 0, 11)
        self.image_model = Model.create(model_def, params=variables)

    def __call__(self, image):
        preprocessed_image = resnet_image_preprocess(image)
        return self.image_model(np.expand_dims(preprocessed_image, axis=0))[0]
