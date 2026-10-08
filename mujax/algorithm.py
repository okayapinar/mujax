"""The contract between the shared infrastructure and an algorithm in `mujax.algorithms`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from mujax.config import MuZeroConfig
from mujax.types import EnvironmentSpec, SearchOutput

# Batched search policy: (params, obs, key, invalid_actions, learner_steps) -> SearchOutput.
# All inputs and outputs have shape (num_envs, ...).
SearchPolicy = Callable[[Any, Any, jax.Array, Any, Any], SearchOutput]

# Search-free root value: (params, obs) -> (N,); used for the bootstrap tail in reanalyze.
ValueFn = Callable[[Any, Any], jnp.ndarray]

# Algorithm loss: (networks, config, params, batch) -> (loss, metrics).
LossFn = Callable[[Any, MuZeroConfig, Any, Any], tuple[jnp.ndarray, dict[str, jnp.ndarray]]]


@dataclasses.dataclass(frozen=True)
class Algorithm:
    """Algorithm-specific parts; everything else (replay, reanalyze, actor, learner, loop) is shared.

    Attributes:
        name: Short name written to checkpoint metadata ("mz", "smz", "gmz").
        config_cls: Config class of this algorithm; `algorithm_for` uses it to pick the algorithm from a config.
        make_networks: (spec, config) -> networks (NamedTuple).
        init_params: (networks, spec, key) -> params (NamedTuple).
        make_policy: (networks, spec, config, evaluation) -> SearchPolicy.
        loss: See `LossFn`.
        make_value_fn: (networks, config) -> ValueFn.
    """

    name: str
    config_cls: type[MuZeroConfig]
    make_networks: Callable[[EnvironmentSpec, MuZeroConfig], Any]
    init_params: Callable[[Any, EnvironmentSpec, jax.Array], Any]
    make_policy: Callable[[Any, EnvironmentSpec, MuZeroConfig, bool], SearchPolicy]
    loss: LossFn
    make_value_fn: Callable[[Any, MuZeroConfig], ValueFn]
