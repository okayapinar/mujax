"""Stochastic MuZero: config, networks, search and loss."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx

from mujax.algorithm import Algorithm, SearchPolicy
from mujax.algorithms import muzero
from mujax.algorithms.muzero import Embedding, Head, Prediction, PUCTConfig, Support, Targets
from mujax.types import EnvironmentSpec, SearchOutput


@dataclasses.dataclass
class SMZConfig(PUCTConfig):
    """Stochastic MuZero hyperparameters."""

    num_simulations: int = 32
    batch_size: int = 2048
    learning_rate: float = 1e-3
    lr_end_value: float = 3e-4
    lr_warmup_steps: int = 3000
    lr_decay_steps: int = 97_000

    temperature_decay_steps: int = 100_000

    # Networks
    representation_layer_sizes: tuple[int, ...] = (128, 128)
    prediction_layer_sizes: tuple[int, ...] = (128, 128)
    encoder_layer_sizes: tuple[int, ...] = (128,)
    decision_layer_sizes: tuple[int, ...] = (128, 128)
    chance_layer_sizes: tuple[int, ...] = (128, 128)
    codebook_size: int = 32
    vqvae_beta: float = 0.25  # commitment loss weight


class Decision(nn.Module):
    """(latent, action) -> afterstate; chance logits and afterstate value from the afterstate."""

    layer_sizes: Sequence[int]
    embedding_dim: int
    codebook_size: int
    num_actions: int
    num_bins: int

    @nn.compact
    def __call__(self, latent, action) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        afterstate = Embedding(self.layer_sizes, self.embedding_dim)(muzero.one_hot_concat(latent, action, self.num_actions))
        chance_logits = Head(self.layer_sizes, self.codebook_size)(afterstate)
        afterstate_value_logits = Head(self.layer_sizes, self.num_bins)(afterstate)
        return afterstate, chance_logits, afterstate_value_logits


class Chance(nn.Module):
    """(afterstate, chance_code) -> (next_latent, reward_logits). The code is one-hot (search) or straight-through (learner)."""

    layer_sizes: Sequence[int]
    embedding_dim: int
    num_bins: int

    @nn.compact
    def __call__(self, afterstate, chance_code) -> tuple[jnp.ndarray, jnp.ndarray]:
        next_latent = Embedding(self.layer_sizes, self.embedding_dim)(jnp.concatenate([afterstate, chance_code], axis=-1))
        return next_latent, Head(self.layer_sizes, self.num_bins)(next_latent)


class SMZNetworks(NamedTuple):
    encoder: Head  # obs -> codebook logits (VQ-VAE encoder)
    representation: Embedding
    prediction: Prediction
    decision: Decision
    chance: Chance


class SMZParams(NamedTuple):
    encoder: dict
    representation: dict
    prediction: dict
    decision: dict
    chance: dict


def make_networks(spec: EnvironmentSpec, config: SMZConfig) -> SMZNetworks:
    return SMZNetworks(
        encoder=Head(config.encoder_layer_sizes, config.codebook_size),
        representation=Embedding(config.representation_layer_sizes, config.embedding_dim),
        prediction=Prediction(config.prediction_layer_sizes, spec.num_actions, config.num_bins),
        decision=Decision(config.decision_layer_sizes, config.embedding_dim, config.codebook_size, spec.num_actions, config.num_bins),
        chance=Chance(config.chance_layer_sizes, config.embedding_dim, config.num_bins),
    )


def init_params(networks: SMZNetworks, spec: EnvironmentSpec, key: jax.Array) -> SMZParams:
    k_enc, k_repr, k_pred, k_dec, k_chance = jax.random.split(key, 5)
    obs = jnp.zeros((1, spec.obs_dim))
    action = jnp.zeros((1,), dtype=jnp.int32)
    code = jnp.zeros((1, networks.encoder.output_dim))
    repr_params = networks.representation.init(k_repr, obs)
    latent = networks.representation.apply(repr_params, obs)
    decision_params = networks.decision.init(k_dec, latent, action)
    afterstate, _, _ = networks.decision.apply(decision_params, latent, action)
    return SMZParams(
        encoder=networks.encoder.init(k_enc, obs),
        representation=repr_params,
        prediction=networks.prediction.init(k_pred, latent),
        decision=decision_params,
        chance=networks.chance.init(k_chance, afterstate, code),
    )


def make_policy(networks: SMZNetworks, spec: EnvironmentSpec, config: SMZConfig, evaluation: bool) -> SearchPolicy:
    """Batched Stochastic MuZero search. `evaluation=True` turns off temperature and Dirichlet noise."""
    support = Support.from_config(config)
    codebook_size = networks.encoder.output_dim
    exploration = muzero.puct_exploration(config, spec.num_actions, evaluation)

    def decision_fn(params, key, action, latent):
        afterstate, chance_logits, afterstate_value_logits = networks.decision.apply(params.decision, latent, action)
        output = mctx.DecisionRecurrentFnOutput(chance_logits=chance_logits, afterstate_value=support.logits_to_scalar(afterstate_value_logits))
        return output, afterstate

    def chance_fn(params, key, chance_outcome, afterstate):
        next_latent, reward_logits = networks.chance.apply(params.chance, afterstate, jax.nn.one_hot(chance_outcome, codebook_size))
        value_logits, policy_logits = networks.prediction.apply(params.prediction, next_latent)
        reward = support.logits_to_scalar(reward_logits)
        output = mctx.ChanceRecurrentFnOutput(
            action_logits=policy_logits,
            value=support.logits_to_scalar(value_logits),
            reward=reward,
            discount=jnp.full_like(reward, config.discount),
        )
        return output, next_latent

    def policy(params, obs, key, invalid_actions, learner_steps) -> SearchOutput:
        out = mctx.stochastic_muzero_policy(
            params,
            key,
            muzero.root_output(networks, support, params, obs),
            decision_fn,
            chance_fn,
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            **exploration(learner_steps),
        )
        return SearchOutput(out.action, out.action_weights, out.search_tree.summary().value)

    return policy


def loss(networks: SMZNetworks, config: SMZConfig, params: SMZParams, batch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    def unroll_step(params, latent, now: Targets, next_: Targets):
        afterstate, chance_logits, afterstate_value_logits = networks.decision.apply(params.decision, latent, now.action)

        # VQ-VAE: code of the next observation; the gradient flows to the encoder via straight-through.
        code_logits = networks.encoder.apply(params.encoder, next_.observation)
        code_onehot = jax.nn.one_hot(jnp.argmax(code_logits, axis=-1), code_logits.shape[-1])
        code = code_logits + jax.lax.stop_gradient(code_onehot - code_logits)

        next_latent, reward_logits = networks.chance.apply(params.chance, afterstate, code)
        value_logits, policy_logits = networks.prediction.apply(params.prediction, next_latent)
        losses = {
            "afterstate_value": muzero.masked_mean(muzero.cross_entropy(afterstate_value_logits, now.value_probs), now.in_episode),
            "chance": muzero.masked_mean(muzero.cross_entropy(chance_logits, code), next_.in_episode),
            "reward": muzero.masked_mean(muzero.cross_entropy(reward_logits, now.reward_probs), now.in_episode),
            "value": muzero.masked_mean(muzero.cross_entropy(value_logits, next_.value_probs), next_.in_episode),
            "policy": muzero.masked_mean(muzero.cross_entropy(policy_logits, next_.policy_probs), next_.in_episode),
            # Paper: commitment loss β‖cᵉ − sg(c)‖² (squared L2 norm, summed over dimensions).
            "commitment": muzero.masked_mean(jnp.sum(jnp.square(code_logits - jax.lax.stop_gradient(code_onehot)), axis=-1), next_.in_episode),
        }
        return next_latent, losses, {"code_usage": jnp.mean(code_onehot, axis=0)}

    loss_weights = {"value": config.value_loss_weight, "afterstate_value": config.value_loss_weight, "commitment": config.vqvae_beta}
    total_loss, metrics = muzero.loss(networks, config, params, batch, unroll_step=unroll_step, loss_weights=loss_weights)
    code_usage = jnp.mean(metrics.pop("code_usage"), axis=0)  # (K, codebook) -> (codebook,)
    metrics["chance_code_perplexity"] = jnp.exp(distrax.Categorical(probs=code_usage).entropy())
    return total_loss, metrics


SMZ = Algorithm("smz", SMZConfig, make_networks, init_params, make_policy, loss, muzero.make_value_fn)
