# -*- coding: utf-8 -*-

from typing import Callable

import tensorflow as tf
from langml import TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
else:
    import keras

from langml.plm.layers import (
    TokenEmbedding, AbsolutePositionEmbedding,
    EmbeddingMatching, Masked,
)

custom_objects = {}
custom_objects.update(TokenEmbedding.get_custom_objects())
custom_objects.update(AbsolutePositionEmbedding.get_custom_objects())
custom_objects.update(EmbeddingMatching.get_custom_objects())
custom_objects.update(Masked.get_custom_objects())

keras.utils.get_custom_objects().update(custom_objects)


def load_variables(checkpoint_path: str) -> Callable:
    def wrap(varname: str):
        return tf.train.load_variable(checkpoint_path, varname)
    return wrap
