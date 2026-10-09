"""The contract between the shared infrastructure and an algorithm in `mujax.algorithms`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from mujax.config import MuZeroConfig
from mujax.types import EnvironmentSpec, SearchOutput

# Batched search policy: (params, obs, key, invalid_actions, learner_steps) -> SearchOutput.
# All inputs and outputs have shape (num_envs, ...).
SearchPolicy = Callable[[Any, Any, jax.Array, Any, Any], SearchOutput]

# Search-free root value: (params, obs) -> (N,); used for the bootstrap tail in reanalyze.
ValueFn = Callable[[Any, Any], jnp.ndarray]

# Algorithm loss: (graphdef, config, params, batch) -> (loss, metrics).
LossFn = Callable[[Any, MuZeroConfig, Any, Any], tuple[jnp.ndarray, dict[str, jnp.ndarray]]]


@dataclasses.dataclass(frozen=True)
class Algorithm:
    """Algorithm-specific parts; everything else (replay, reanalyze, actor, learner, loop) is shared.

    The model is a single `nnx.Module`; `init` splits it into a graphdef (static structure) and params
    (`nnx.State`, a pytree). Params travel separately (learner, actor threads, mctx, checkpoints) and the
    algorithm functions rebuild the model with `nnx.merge(graphdef, params)`.

    Attributes:
        name: Short name written to checkpoint metadata ("gmz", "smz").
        config_cls: Config class of this algorithm; `algorithm_for` uses it to pick the algorithm from a config.
        make_model: (spec, config, rngs) -> model (nnx.Module).
        make_policy: (graphdef, spec, config, evaluation) -> SearchPolicy.
        loss: See `LossFn`.
        make_value_fn: (graphdef, config) -> ValueFn.
    """

    name: str
    config_cls: type[MuZeroConfig]
    make_model: Callable[[EnvironmentSpec, MuZeroConfig, nnx.Rngs], nnx.Module]
    make_policy: Callable[[nnx.GraphDef, EnvironmentSpec, MuZeroConfig, bool], SearchPolicy]
    loss: LossFn
    make_value_fn: Callable[[nnx.GraphDef, MuZeroConfig], ValueFn]

    def init(self, spec: EnvironmentSpec, config: MuZeroConfig, key: jax.Array) -> tuple[nnx.GraphDef, nnx.State]:
        """(graphdef, params) of a freshly initialized model."""
        return nnx.split(self.make_model(spec, config, nnx.Rngs(key)))
