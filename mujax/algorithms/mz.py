"""MuZero: config, networks, search and loss.

The deterministic model here (representation, dynamics, prediction) and its loss are also used by Gumbel MuZero (gmz.py),
which only replaces the search.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx

from mujax.algorithm import Algorithm, SearchPolicy
from mujax.algorithms import muzero
from mujax.algorithms.muzero import Embedding, Head, Prediction, PUCTConfig, Support, Targets
from mujax.types import EnvironmentSpec, SearchOutput

if TYPE_CHECKING:
    from mujax.algorithms.gmz import GMZConfig


@dataclasses.dataclass
class MZConfig(PUCTConfig):
    """MuZero hyperparameters."""

    max_depth: int | None = None
    dynamics_layer_sizes: tuple[int, ...] = (256, 256, 256)


class Dynamics(nn.Module):
    layer_sizes: Sequence[int]
    embedding_dim: int
    num_actions: int
    num_bins: int

    @nn.compact
    def __call__(self, latent, action) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(next_latent, reward_logits)."""
        next_latent = Embedding(self.layer_sizes, self.embedding_dim)(muzero.one_hot_concat(latent, action, self.num_actions))
        return next_latent, Head(self.layer_sizes, self.num_bins)(next_latent)


class MZNetworks(NamedTuple):
    representation: Embedding
    prediction: Prediction
    dynamics: Dynamics


class MZParams(NamedTuple):
    representation: dict
    prediction: dict
    dynamics: dict


def make_networks(spec: EnvironmentSpec, config: MZConfig | GMZConfig) -> MZNetworks:
    return MZNetworks(
        representation=Embedding(config.representation_layer_sizes, config.embedding_dim),
        prediction=Prediction(config.prediction_layer_sizes, spec.num_actions, config.num_bins),
        dynamics=Dynamics(config.dynamics_layer_sizes, config.embedding_dim, spec.num_actions, config.num_bins),
    )


def init_params(networks: MZNetworks, spec: EnvironmentSpec, key: jax.Array) -> MZParams:
    k_repr, k_pred, k_dyn = jax.random.split(key, 3)
    obs = jnp.zeros((1, spec.obs_dim))
    action = jnp.zeros((1,), dtype=jnp.int32)
    repr_params = networks.representation.init(k_repr, obs)
    latent = networks.representation.apply(repr_params, obs)
    return MZParams(
        representation=repr_params,
        prediction=networks.prediction.init(k_pred, latent),
        dynamics=networks.dynamics.init(k_dyn, latent, action),
    )


def recurrent_fn(networks: MZNetworks, support: Support, config: MZConfig | GMZConfig):
    """mctx recurrent_fn of the deterministic model; shared by the MZ and GMZ searches."""

    def fn(params, key, action, latent):
        next_latent, reward_logits = networks.dynamics.apply(params.dynamics, latent, action)
        value_logits, policy_logits = networks.prediction.apply(params.prediction, next_latent)
        reward = support.logits_to_scalar(reward_logits)
        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.full_like(reward, config.discount),
            prior_logits=policy_logits,
            value=support.logits_to_scalar(value_logits),
        )
        return output, next_latent

    return fn


def make_policy(networks: MZNetworks, spec: EnvironmentSpec, config: MZConfig, evaluation: bool) -> SearchPolicy:
    """Batched MuZero (PUCT) search. `evaluation=True` turns off temperature and Dirichlet noise."""
    support = Support.from_config(config)
    exploration = muzero.puct_exploration(config, spec.num_actions, evaluation)
    step_fn = recurrent_fn(networks, support, config)

    def policy(params, obs, key, invalid_actions, learner_steps) -> SearchOutput:
        out = mctx.muzero_policy(
            params,
            key,
            muzero.root_output(networks, support, params, obs),
            step_fn,
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            max_depth=config.max_depth,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            **exploration(learner_steps),
        )
        return SearchOutput(out.action, out.action_weights, out.search_tree.summary().value)

    return policy


def loss(networks: MZNetworks, config: MZConfig | GMZConfig, params: MZParams, batch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    def unroll_step(params, latent, now: Targets, next_: Targets):
        next_latent, reward_logits = networks.dynamics.apply(params.dynamics, latent, now.action)
        value_logits, policy_logits = networks.prediction.apply(params.prediction, next_latent)
        losses = {
            "reward": muzero.masked_mean(muzero.cross_entropy(reward_logits, now.reward_probs), now.in_episode),
            "value": muzero.masked_mean(muzero.cross_entropy(value_logits, next_.value_probs), next_.in_episode),
            "policy": muzero.masked_mean(muzero.cross_entropy(policy_logits, next_.policy_probs), next_.in_episode),
        }
        return next_latent, losses, {}

    return muzero.loss(networks, config, params, batch, unroll_step=unroll_step, loss_weights={"value": config.value_loss_weight})


MZ = Algorithm("mz", MZConfig, make_networks, init_params, make_policy, loss, muzero.make_value_fn)
