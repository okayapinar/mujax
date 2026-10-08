"""Algorithms; each module defines an `Algorithm` (see `mujax.algorithm`). New algorithms are registered in `ALGORITHMS`."""

from __future__ import annotations

from mujax.algorithm import Algorithm
from mujax.algorithms.gmz import GMZ, GMZConfig
from mujax.algorithms.mz import MZ, MZConfig
from mujax.algorithms.smz import SMZ, SMZConfig
from mujax.config import MuZeroConfig

ALGORITHMS: dict[str, Algorithm] = {MZ.name: MZ, SMZ.name: SMZ, GMZ.name: GMZ}


def algorithm_for(config: MuZeroConfig) -> Algorithm:
    """Algorithm by config class (subclasses match too)."""
    for algorithm in ALGORITHMS.values():
        if isinstance(config, algorithm.config_cls):
            return algorithm
    raise ValueError(f"{type(config).__name__} icin algoritma bulunamadi; ExperimentConfig.algorithm ver.")


__all__ = ["ALGORITHMS", "GMZ", "MZ", "SMZ", "GMZConfig", "MZConfig", "SMZConfig", "algorithm_for"]
