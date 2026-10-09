"""MuZero: config, networks, search and loss.

The deterministic model here (representation, dynamics, prediction) and its loss are also used by Gumbel MuZero (gmz.py),
which only replaces the search.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import mctx
from flax import nnx

from mujax.algorithm import Algorithm, SearchPolicy
from mujax.algorithms import muzero
from mujax.algorithms.muzero import Embedding, Head, Prediction, PUCTConfig, Support, SymlogInput, Targets
from mujax.types import EnvironmentSpec, SearchOutput

if TYPE_CHECKING:
    from mujax.algorithms.gmz import GMZConfig


@dataclasses.dataclass
class MZConfig(PUCTConfig):
    """MuZero hyperparameters."""

    max_depth: int | None = None
    dynamics_layer_sizes: tuple[int, ...] = (256, 256, 256)


class Dynamics(nnx.Module):
    def __init__(self, embedding_dim: int, layer_sizes: Sequence[int], num_actions: int, num_bins: int, *, rngs: nnx.Rngs):
        self.num_actions = num_actions
        self.embedding = Embedding(embedding_dim + num_actions, layer_sizes, embedding_dim, rngs=rngs)
        self.reward = Head(embedding_dim, layer_sizes, num_bins, rngs=rngs)

    def __call__(self, latent, action) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(next_latent, reward_logits)."""
        next_latent = self.embedding(muzero.one_hot_concat(latent, action, self.num_actions))
        return next_latent, self.reward(next_latent)


class MZModel(nnx.Module):
    """The deterministic model: representation, prediction and dynamics."""

    def __init__(self, spec: EnvironmentSpec, config: MZConfig | GMZConfig, rngs: nnx.Rngs):
        self.representation = SymlogInput(Embedding(spec.obs_dim, config.representation_layer_sizes, config.embedding_dim, rngs=rngs))
        self.prediction = Prediction(
            config.embedding_dim, config.prediction_layer_sizes, spec.num_actions, config.num_bins, unimix=config.policy_unimix, rngs=rngs
        )
        self.dynamics = Dynamics(config.embedding_dim, config.dynamics_layer_sizes, spec.num_actions, config.num_bins, rngs=rngs)


def recurrent_fn(graphdef: nnx.GraphDef, support: Support, config: MZConfig | GMZConfig):
    """mctx recurrent_fn of the deterministic model; shared by the MZ and GMZ searches."""

    def fn(params, key, action, latent):
        model = nnx.merge(graphdef, params)
        next_latent, reward_logits = model.dynamics(latent, action)
        value_logits, policy_logits = model.prediction(next_latent)
        reward = support.logits_to_scalar(reward_logits)
        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.full_like(reward, config.discount),
            prior_logits=policy_logits,
            value=support.logits_to_scalar(value_logits),
        )
        return output, next_latent

    return fn


def make_policy(graphdef: nnx.GraphDef, spec: EnvironmentSpec, config: MZConfig, evaluation: bool) -> SearchPolicy:
    """Batched MuZero (PUCT) search. `evaluation=True` turns off temperature and Dirichlet noise."""
    support = Support.from_config(config)
    exploration = muzero.puct_exploration(config, spec.num_actions, evaluation)
    step_fn = recurrent_fn(graphdef, support, config)

    def policy(params, obs, key, invalid_actions, learner_steps) -> SearchOutput:
        out = mctx.muzero_policy(
            params,
            key,
            muzero.root_output(nnx.merge(graphdef, params), support, obs),
            step_fn,
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            max_depth=config.max_depth,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            **exploration(learner_steps),
        )
        return SearchOutput(out.action, out.action_weights, out.search_tree.summary().value)

    return policy


def loss(graphdef: nnx.GraphDef, config: MZConfig | GMZConfig, params: nnx.State, batch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    def unroll_step(model: MZModel, latent, now: Targets, next_: Targets):
        next_latent, reward_logits = model.dynamics(latent, now.action)
        value_logits, policy_logits = model.prediction(next_latent)
        losses = {
            "reward": muzero.masked_mean(muzero.cross_entropy(reward_logits, now.reward_probs), now.in_episode),
            "value": muzero.masked_mean(muzero.cross_entropy(value_logits, next_.value_probs), next_.in_episode),
            "policy": muzero.masked_mean(muzero.cross_entropy(policy_logits, next_.policy_probs), next_.in_episode),
        }
        return next_latent, losses, {}

    return muzero.loss(graphdef, config, params, batch, unroll_step=unroll_step, loss_weights={"value": config.value_loss_weight})


MZ = Algorithm("mz", MZConfig, MZModel, make_policy, loss, muzero.make_value_fn)
