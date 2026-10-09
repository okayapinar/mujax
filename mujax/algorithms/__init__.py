"""Algorithms; each module defines an `Algorithm` (see `mujax.algorithm`). New algorithms are registered in `ALGORITHMS`."""

from __future__ import annotations

from mujax.algorithm import Algorithm
from mujax.algorithms.gmz import GMZ, GMZConfig
from mujax.algorithms.smz import SMZ, SMZConfig
from mujax.config import MuZeroConfig

ALGORITHMS: dict[str, Algorithm] = {SMZ.name: SMZ, GMZ.name: GMZ}


def algorithm_for(config: MuZeroConfig) -> Algorithm:
    """Algorithm by config class: an exact class match first, then a subclass match."""
    for algorithm in ALGORITHMS.values():
        if type(config) is algorithm.config_cls:
            return algorithm
    for algorithm in ALGORITHMS.values():
        if isinstance(config, algorithm.config_cls):
            return algorithm
    raise ValueError(f"{type(config).__name__} icin algoritma bulunamadi; ExperimentConfig.algorithm ver.")


__all__ = ["ALGORITHMS", "GMZ", "SMZ", "GMZConfig", "SMZConfig", "algorithm_for"]
