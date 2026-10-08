"""Stochastic MuZero: config, networks, search and loss."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
import distrax
import jax
import jax.numpy as jnp
import mctx
from flax import nnx

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


class Decision(nnx.Module):
    """(latent, action) -> afterstate; chance logits and afterstate value from the afterstate."""

    def __init__(self, embedding_dim: int, layer_sizes: Sequence[int], codebook_size: int, num_actions: int, num_bins: int, *, rngs: nnx.Rngs):
        self.num_actions = num_actions
        self.afterstate = Embedding(embedding_dim + num_actions, layer_sizes, embedding_dim, rngs=rngs)
        self.chance_logits = Head(embedding_dim, layer_sizes, codebook_size, rngs=rngs)
        self.afterstate_value = Head(embedding_dim, layer_sizes, num_bins, rngs=rngs)

    def __call__(self, latent, action) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        afterstate = self.afterstate(muzero.one_hot_concat(latent, action, self.num_actions))
        return afterstate, self.chance_logits(afterstate), self.afterstate_value(afterstate)


class Chance(nnx.Module):
    """(afterstate, chance_code) -> (next_latent, reward_logits). The code is one-hot (search) or straight-through (learner)."""

    def __init__(self, embedding_dim: int, layer_sizes: Sequence[int], codebook_size: int, num_bins: int, *, rngs: nnx.Rngs):
        self.embedding = Embedding(embedding_dim + codebook_size, layer_sizes, embedding_dim, rngs=rngs)
        self.reward = Head(embedding_dim, layer_sizes, num_bins, rngs=rngs)

    def __call__(self, afterstate, chance_code) -> tuple[jnp.ndarray, jnp.ndarray]:
        next_latent = self.embedding(jnp.concatenate([afterstate, chance_code], axis=-1))
        return next_latent, self.reward(next_latent)


class SMZModel(nnx.Module):
    def __init__(self, spec: EnvironmentSpec, config: SMZConfig, rngs: nnx.Rngs):
        d, k = config.embedding_dim, config.codebook_size
        self.encoder = Head(spec.obs_dim, config.encoder_layer_sizes, k, rngs=rngs)  # obs -> codebook logits (VQ-VAE encoder)
        self.representation = Embedding(spec.obs_dim, config.representation_layer_sizes, d, rngs=rngs)
        self.prediction = Prediction(d, config.prediction_layer_sizes, spec.num_actions, config.num_bins, rngs=rngs)
        self.decision = Decision(d, config.decision_layer_sizes, k, spec.num_actions, config.num_bins, rngs=rngs)
        self.chance = Chance(d, config.chance_layer_sizes, k, config.num_bins, rngs=rngs)


def make_policy(graphdef: nnx.GraphDef, spec: EnvironmentSpec, config: SMZConfig, evaluation: bool) -> SearchPolicy:
    """Batched Stochastic MuZero search. `evaluation=True` turns off temperature and Dirichlet noise."""
    support = Support.from_config(config)
    codebook_size = config.codebook_size
    exploration = muzero.puct_exploration(config, spec.num_actions, evaluation)

    def decision_fn(params, key, action, latent):
        afterstate, chance_logits, afterstate_value_logits = nnx.merge(graphdef, params).decision(latent, action)
        output = mctx.DecisionRecurrentFnOutput(chance_logits=chance_logits, afterstate_value=support.logits_to_scalar(afterstate_value_logits))
        return output, afterstate

    def chance_fn(params, key, chance_outcome, afterstate):
        model = nnx.merge(graphdef, params)
        next_latent, reward_logits = model.chance(afterstate, jax.nn.one_hot(chance_outcome, codebook_size))
        value_logits, policy_logits = model.prediction(next_latent)
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
            muzero.root_output(nnx.merge(graphdef, params), support, obs),
            decision_fn,
            chance_fn,
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            **exploration(learner_steps),
        )
        return SearchOutput(out.action, out.action_weights, out.search_tree.summary().value)

    return policy


def loss(graphdef: nnx.GraphDef, config: SMZConfig, params: nnx.State, batch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    def unroll_step(model: SMZModel, latent, now: Targets, next_: Targets):
        afterstate, chance_logits, afterstate_value_logits = model.decision(latent, now.action)

        # VQ-VAE: code of the next observation; the gradient flows to the encoder via straight-through.
        code_logits = model.encoder(next_.observation)
        code_onehot = jax.nn.one_hot(jnp.argmax(code_logits, axis=-1), code_logits.shape[-1])
        code = code_logits + jax.lax.stop_gradient(code_onehot - code_logits)

        next_latent, reward_logits = model.chance(afterstate, code)
        value_logits, policy_logits = model.prediction(next_latent)
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
    total_loss, metrics = muzero.loss(graphdef, config, params, batch, unroll_step=unroll_step, loss_weights=loss_weights)
    code_usage = jnp.mean(metrics.pop("code_usage"), axis=0)  # (K, codebook) -> (codebook,)
    metrics["chance_code_perplexity"] = jnp.exp(distrax.Categorical(probs=code_usage).entropy())
    return total_loss, metrics


SMZ = Algorithm("smz", SMZConfig, SMZModel, make_policy, loss, muzero.make_value_fn)
