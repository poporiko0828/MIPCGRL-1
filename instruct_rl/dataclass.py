from dataclasses import dataclass
from enum import IntEnum
from functools import partial
import os
import time
from timeit import default_timer as timer
from typing import Optional
import chex
import gymnax
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
import numpy as np
from PIL import Image

# import tkinter as tk


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


@struct.dataclass
class Instruct:
    reward_i: int
    condition: chex.Array
    embedding: chex.Array


@struct.dataclass
class EmbeddingBufferReward:
    embedding: chex.Array
    buffer: chex.Array
    reward: chex.Array


class NormalizationWeights(IntEnum):
    REGION = 100
    PATHLENGTH = 60
    WALL = 160
    MONSTER = 75
    DIRECTION = 55
