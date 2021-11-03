# -*- coding: utf-8 -*-

from typing import Dict, Any


class BaselineModel:
    def build_model(self, *args, **kwargs):
        raise NotImplementedError


class Parameters:
    """ Hyper-Parameters
    """
    def __init__(self, data: Dict):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value: Any):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Parameters(value) if isinstance(value, dict) else value
