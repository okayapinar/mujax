"""Stochastic MuZero: config, networks, search and loss."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Self

import jax
import jax.numpy as jnp
import mctx
from flax import nnx

from mujax.algorithm import Algorithm, SearchPolicy
from mujax.algorithms import muzero
from mujax.algorithms.muzero import Embedding, Head, Prediction, Support, SymlogInput, Targets, cross_entropy, masked_mean
from mujax.config import MuZeroConfig
from mujax.types import EnvironmentSpec, SearchOutput


@dataclasses.dataclass
class SMZConfig(MuZeroConfig):
    """Stochastic MuZero hyperparameters."""

    num_simulations: int = 32
    batch_size: int = 2048
    learning_rate: float = 1e-3

    # PUCT exploration (`mctx.stochastic_muzero_policy`)
    temperature_decay_steps: int = 100_000  # temperature 1.0 -> 0.5 at half -> 0.25 at three quarters
    dirichlet_fraction: float = 0.1
    dirichlet_alpha: float | None = None  # If None, 1/sqrt(num_actions)
    pb_c_init: float = 1.25
    pb_c_base: float = 19652.0

    # Networks
    representation_layer_sizes: tuple[int, ...] = (128, 128)
    prediction_layer_sizes: tuple[int, ...] = (128, 128)
    encoder_layer_sizes: tuple[int, ...] = (128,)
    decision_layer_sizes: tuple[int, ...] = (128, 128)
    chance_layer_sizes: tuple[int, ...] = (128, 128)
    codebook_size: int = 32
    vqvae_beta: float = 0.25  # commitment loss weight

    def with_num_steps(self, num_steps: int) -> Self:
        config = super().with_num_steps(num_steps)
        return dataclasses.replace(config, temperature_decay_steps=num_steps)


class Decision(nnx.Module):
    """(latent, action) -> (afterstate, chance_logits, afterstate_value_logits)."""

    def __init__(
        self,
        embedding_dim: int,
        layer_sizes: Sequence[int],
        codebook_size: int,
        num_actions: int,
        num_bins: int,
        *,
        unimix: float = 0.0,
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.unimix = float(unimix)
        self.afterstate = Embedding(embedding_dim + num_actions, layer_sizes, embedding_dim, rngs=rngs)
        self.chance_logits = Head(embedding_dim, layer_sizes, codebook_size, rngs=rngs)
        self.afterstate_value = Head(embedding_dim, layer_sizes, num_bins, rngs=rngs)

    def __call__(self, latent, action) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        latent_action = muzero.one_hot_concat(latent, action, self.num_actions)
        afterstate = self.afterstate(latent_action)
        chance_logits = self.chance_logits(afterstate)
        chance_logits = muzero.unimix_logits(chance_logits, self.unimix)
        afterstate_value = self.afterstate_value(afterstate)
        return afterstate, chance_logits, afterstate_value


class Chance(nnx.Module):
    """(afterstate, chance_code) -> (next_latent, reward_logits)."""

    def __init__(self, embedding_dim: int, layer_sizes: Sequence[int], codebook_size: int, num_bins: int, *, rngs: nnx.Rngs):
        self.embedding = Embedding(embedding_dim + codebook_size, layer_sizes, embedding_dim, rngs=rngs)
        self.reward = Head(embedding_dim, layer_sizes, num_bins, rngs=rngs)

    def __call__(self, afterstate, chance_code) -> tuple[jnp.ndarray, jnp.ndarray]:
        afterstate_code = jnp.concatenate([afterstate, chance_code], axis=-1)
        next_latent = self.embedding(afterstate_code)
        reward_logits = self.reward(next_latent)
        return next_latent, reward_logits


class SMZModel(nnx.Module):
    def __init__(self, spec: EnvironmentSpec, config: SMZConfig, rngs: nnx.Rngs):
        d, k = config.embedding_dim, config.codebook_size
        encoder = Head(spec.obs_dim, config.encoder_layer_sizes, k, zero_init=False, rngs=rngs)
        representation = Embedding(spec.obs_dim, config.representation_layer_sizes, d, rngs=rngs)
        self.encoder = SymlogInput(encoder)
        self.representation = SymlogInput(representation)
        self.prediction = Prediction(d, config.prediction_layer_sizes, spec.num_actions, config.num_bins, unimix=config.policy_unimix, rngs=rngs)
        self.decision = Decision(d, config.decision_layer_sizes, k, spec.num_actions, config.num_bins, unimix=config.policy_unimix, rngs=rngs)
        self.chance = Chance(d, config.chance_layer_sizes, k, config.num_bins, rngs=rngs)


def make_policy(graphdef: nnx.GraphDef, spec: EnvironmentSpec, config: SMZConfig, evaluation: bool) -> SearchPolicy:
    """Batched Stochastic MuZero search. `evaluation=True` turns off temperature and Dirichlet noise."""
    support = Support.from_config(config)
    dirichlet_alpha = 1.0 / spec.num_actions**0.5 if config.dirichlet_alpha is None else config.dirichlet_alpha
    half = int(0.5 * config.temperature_decay_steps)
    three_quarters = int(0.75 * config.temperature_decay_steps)

    def decision_fn(params, key, action, latent):
        model = nnx.merge(graphdef, params)
        afterstate, chance_logits, afterstate_value_logits = model.decision(latent, action)
        afterstate_value = support.logits_to_scalar(afterstate_value_logits)
        output = mctx.DecisionRecurrentFnOutput(chance_logits=chance_logits, afterstate_value=afterstate_value)
        return output, afterstate

    def chance_fn(params, key, chance_outcome, afterstate):
        model = nnx.merge(graphdef, params)
        chance_code = jax.nn.one_hot(chance_outcome, config.codebook_size)
        next_latent, reward_logits = model.chance(afterstate, chance_code)
        value_logits, policy_logits = model.prediction(next_latent)
        reward = support.logits_to_scalar(reward_logits)
        value = support.logits_to_scalar(value_logits)
        output = mctx.ChanceRecurrentFnOutput(
            action_logits=policy_logits,
            value=value,
            reward=reward,
            discount=jnp.full_like(reward, config.discount),
        )
        return output, next_latent

    def policy(params, obs, key, invalid_actions, learner_steps) -> SearchOutput:
        if evaluation:
            temperature = 0.0
            dirichlet_fraction = 0.0
        else:
            later = jnp.where(learner_steps < three_quarters, 0.5, 0.25)
            temperature = jnp.where(learner_steps < half, 1.0, later)
            dirichlet_fraction = config.dirichlet_fraction
        model = nnx.merge(graphdef, params)
        root = muzero.root_output(model, support, obs)
        out = mctx.stochastic_muzero_policy(
            params,
            key,
            root,
            decision_fn,
            chance_fn,
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            temperature=temperature,
            dirichlet_fraction=dirichlet_fraction,
            dirichlet_alpha=dirichlet_alpha,
            pb_c_init=config.pb_c_init,
            pb_c_base=config.pb_c_base,
        )
        value = out.search_tree.summary().value
        return SearchOutput(out.action, out.action_weights, value)

    return policy


def loss(graphdef: nnx.GraphDef, config: SMZConfig, params: nnx.State, batch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    def unroll_step(model: SMZModel, latent, now: Targets, next_: Targets):
        afterstate, chance_logits, afterstate_value_logits = model.decision(latent, now.action)

        # VQ-VAE: one-hot code of the next observation with straight-through gradients to the encoder.
        code_logits = model.encoder(next_.observation)
        code_index = jnp.argmax(code_logits, axis=-1)
        code_onehot = jax.nn.one_hot(code_index, config.codebook_size)
        straight_through = jax.lax.stop_gradient(code_onehot - code_logits)
        code = code_logits + straight_through

        next_latent, reward_logits = model.chance(afterstate, code)
        value_logits, policy_logits = model.prediction(next_latent)
        target = jax.lax.stop_gradient(code_onehot)
        commitment = jnp.sum(jnp.square(code_logits - target), axis=-1)
        losses = {
            "afterstate_value": cross_entropy(afterstate_value_logits, now.value_probs, now.in_episode),
            "chance": cross_entropy(chance_logits, code_onehot, next_.in_episode),
            "reward": cross_entropy(reward_logits, now.reward_probs, now.in_episode),
            "value": cross_entropy(value_logits, next_.value_probs, next_.in_episode),
            "policy": cross_entropy(policy_logits, next_.policy_probs, next_.in_episode),
            "commitment": masked_mean(commitment, next_.in_episode),
        }
        return next_latent, losses

    loss_weights = {"value": config.value_loss_weight, "afterstate_value": config.value_loss_weight, "commitment": config.vqvae_beta}
    return muzero.loss(graphdef, config, params, batch, unroll_step=unroll_step, loss_weights=loss_weights)


SMZ = Algorithm("smz", SMZConfig, SMZModel, make_policy, loss, muzero.make_value_fn)
