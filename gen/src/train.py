import functools
from gan.models import create_generator
import os
from gan.serialize import deserialize_model, serialize_model
import tensorflow as tf
from typing import Optional


class TrainingOptions:
    def __init__(
            self,
            resolution: int,
            latent_size = 512
            ):

        self.resolution = resolution
        self.latent_size = latent_size

    @property
    def epoch_count(self):
        return self.total_sample_count // self.epoch_sample_count


class TrainingState:
    def __init__(
            self,
            options: TrainingOptions,
            visualization_generator: Optional[tf.keras.Model] = None,
            epoch_i: int = 0):
        self.options = options
        self.visualization_generator = visualization_generator
        self.epoch_i = epoch_i

    def __getstate__(self):
        state = self.__dict__.copy()
        state['visualization_generator'] = serialize_model(self.visualization_generator)
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.visualization_generator = deserialize_model(
            self.visualization_generator,
            functools.partial(create_generator, self.options.resolution, self.options.latent_size))














