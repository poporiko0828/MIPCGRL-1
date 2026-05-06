import math
from functools import partial
from typing import Tuple, Sequence

import jax.lax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

from conf.config import EncoderConfig
import numpy as np
import jax.numpy as jnp

from models import crop_arf_vrf


class ConvForward(nn.Module):
    arf_size: int
    vrf_size: int
    num_layers: int
    hidden_size: int
    output_size: int
    activation: str = "relu"
    output_dim: int = 1
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, map_x, train=False, rng=None):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        act, critic = crop_arf_vrf(map_x, self.arf_size, self.vrf_size)

        # Conv Layers
        for _ in range(self.num_layers):
            act = nn.Conv(
                features=self.hidden_size // 2, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )(act)
            act = activation(act)
            act = nn.Dropout(rate=self.dropout_rate)(act, deterministic=not train)

        # Flatten
        act = act.reshape((act.shape[0], -1))

        # Dense Layers
        for _ in range(self.num_layers):
            act = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            )(act)

            act = activation(act)
            act = nn.Dropout(rate=self.dropout_rate)(act, deterministic=not train)

        # Output Layer
        act = nn.Dense(
            self.output_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        act = act.reshape((1, -1))

        return act



def apply_model(config: EncoderConfig):
    encoding_model = ConvForward(
        num_layers=config.num_layers,
        hidden_size=config.hidden_dim,
        output_size=config.output_dim,
        activation=config.activation,
        arf_size=config.arf_size,
        vrf_size=config.vrf_size,
        output_dim=config.output_dim,
        dropout_rate=config.encoder.dropout_rate,
    )
    return encoding_model