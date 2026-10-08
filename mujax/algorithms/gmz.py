"""Gumbel MuZero: config and search; the model and loss are MuZero's (mz.py)."""

from __future__ import annotations

import dataclasses
import functools

import mctx

from mujax.algorithm import Algorithm, SearchPolicy
from mujax.algorithms import muzero, mz
from mujax.algorithms.muzero import Support
from mujax.config import MuZeroConfig
from mujax.types import EnvironmentSpec, SearchOutput


@dataclasses.dataclass
class GMZConfig(MuZeroConfig):
    """Gumbel MuZero hyperparameters."""

    num_simulations: int = 16
    max_depth: int | None = None
    max_num_considered_actions: int = 16  # clipped to the number of actions
    # Gumbel noise is gmz's only source of exploration (mz and smz use temperature + Dirichlet).
    gumbel_scale: float = 1.0
    value_scale: float = 0.1
    maxvisit_init: float = 50.0

    dynamics_layer_sizes: tuple[int, ...] = (256, 256, 256)


def make_policy(networks: mz.MZNetworks, spec: EnvironmentSpec, config: GMZConfig, evaluation: bool) -> SearchPolicy:
    """Batched Gumbel MuZero search. `evaluation=True` turns off the Gumbel noise."""
    support = Support.from_config(config)
    max_num_considered_actions = min(int(config.max_num_considered_actions), spec.num_actions)
    qtransform = functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=config.value_scale, maxvisit_init=config.maxvisit_init)

    def policy(params, obs, key, invalid_actions, learner_steps) -> SearchOutput:
        del learner_steps
        out = mctx.gumbel_muzero_policy(
            params,
            key,
            muzero.root_output(networks, support, params, obs),
            mz.recurrent_fn(networks, support, config),
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            max_depth=config.max_depth,
            qtransform=qtransform,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=0.0 if evaluation else config.gumbel_scale,
        )
        return SearchOutput(out.action, out.action_weights, out.search_tree.summary().value)

    return policy


GMZ = Algorithm("gmz", GMZConfig, mz.make_networks, mz.init_params, make_policy, mz.loss, muzero.make_value_fn)
