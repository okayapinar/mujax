"""Shared pieces of the MuZero family: value support, network blocks and the loss."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import mctx
import optax
import rlax
from flax import nnx

from mujax.algorithm import ValueFn
from mujax.config import MuZeroConfig


class Support(NamedTuple):
    """Splits `[min, max]` into `num_bins` evenly spaced bins in symlog space (HL-Gauss targets)."""

    min: float
    max: float
    num_bins: int
    sigma_scale: float = 0.75

    @classmethod
    def from_config(cls, config: MuZeroConfig) -> Support:
        return cls(float(config.support_min), float(config.support_max), int(config.num_bins), float(config.hl_gauss_sigma_scale))

    @property
    def bin_width(self) -> float:
        return (self.max - self.min) / (self.num_bins - 1)

    @property
    def bin_centers(self) -> jnp.ndarray:
        return jnp.linspace(self.min, self.max, self.num_bins)

    def to_probs(self, x: jnp.ndarray) -> jnp.ndarray:
        """Scalar -> probabilities over the bins."""
        symlog = rlax.signed_logp1(x)
        center = jnp.clip(symlog, self.min, self.max)[..., None]
        scale = self.sigma_scale * self.bin_width
        half = self.bin_width / 2
        cdf = jax.scipy.stats.norm.cdf
        upper = cdf(self.bin_centers + half, loc=center, scale=scale)
        lower = cdf(self.bin_centers - half, loc=center, scale=scale)
        probs = upper - lower
        total = jnp.sum(probs, axis=-1, keepdims=True)
        return probs / jnp.maximum(total, 1e-8)

    def to_scalar(self, probs: jnp.ndarray) -> jnp.ndarray:
        """Inverse of `to_probs`: expectation in symlog space, then symexp."""
        symlog = rlax.transform_from_2hot(probs, self.min, self.max, self.num_bins)
        return rlax.signed_expm1(symlog)

    def logits_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        probs = jax.nn.softmax(logits)
        return self.to_scalar(probs)


class SymlogInput(nnx.Module):
    """Applies `signed_logp1` to raw observations before passing them to `net`."""

    def __init__(self, net: nnx.Module):
        self.net = net

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        obs = rlax.signed_logp1(obs)
        return self.net(obs)


class Head(nnx.Module):
    """MLP (Linear -> LayerNorm -> SiLU) + linear output (zero-initialized by default)."""

    def __init__(self, in_features: int, layer_sizes: Sequence[int], output_dim: int, *, zero_init: bool = True, rngs: nnx.Rngs):
        layers = []
        for out_features in layer_sizes:
            layers.extend([nnx.Linear(in_features, out_features, rngs=rngs), nnx.LayerNorm(out_features, rngs=rngs), nnx.silu])
            in_features = out_features
        kernel_init = nnx.initializers.zeros_init() if zero_init else nnx.initializers.lecun_normal()
        layers.append(nnx.Linear(in_features, output_dim, kernel_init=kernel_init, rngs=rngs))
        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layers(x)


class Embedding(nnx.Module):
    """Head + min-max normalization to [0, 1] for latent states."""

    def __init__(self, in_features: int, layer_sizes: Sequence[int], embedding_dim: int, *, rngs: nnx.Rngs):
        self.head = Head(in_features, layer_sizes, embedding_dim, zero_init=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.head(x)
        return min_max_normalize(x)


class Prediction(nnx.Module):
    """Latent -> (value_logits, policy_logits) with unimix on the policy logits."""

    def __init__(self, embedding_dim: int, layer_sizes: Sequence[int], num_actions: int, num_bins: int, *, unimix: float = 0.0, rngs: nnx.Rngs):
        self.value = Head(embedding_dim, layer_sizes, num_bins, rngs=rngs)
        self.policy = Head(embedding_dim, layer_sizes, num_actions, rngs=rngs)
        self.unimix = float(unimix)

    def __call__(self, latent: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        value_logits = self.value(latent)
        policy_logits = self.policy(latent)
        return value_logits, unimix_logits(policy_logits, self.unimix)


def unimix_logits(logits: jnp.ndarray, ratio: float) -> jnp.ndarray:
    """DreamerV3 unimix: log((1 - ratio) * softmax(logits) + ratio / A). Identity for ratio <= 0."""
    if ratio <= 0.0:
        return logits
    probs = jax.nn.softmax(logits)
    return jnp.log((1.0 - ratio) * probs + ratio / logits.shape[-1])


def min_max_normalize(x: jnp.ndarray) -> jnp.ndarray:
    """Rescales the last axis to [0, 1]."""
    x_min = x.min(axis=-1, keepdims=True)
    x_max = x.max(axis=-1, keepdims=True)
    return (x - x_min) / jnp.maximum(x_max - x_min, 1e-5)


def scale_gradient(x: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Identity on the forward pass; scales the gradient by `scale` on the backward pass."""
    return x * scale + jax.lax.stop_gradient(x) * (1.0 - scale)


def one_hot_concat(latent: jnp.ndarray, index: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Appends a one-hot action/code to `latent`."""
    return jnp.concatenate([latent, jax.nn.one_hot(index, num_classes)], axis=-1)


def root_output(model: Any, support: Support, obs: jnp.ndarray) -> mctx.RootFnOutput:
    """representation -> prediction; root input of the mctx search."""
    embedding = model.representation(obs)
    value_logits, policy_logits = model.prediction(embedding)
    value = support.logits_to_scalar(value_logits)
    return mctx.RootFnOutput(prior_logits=policy_logits, value=value, embedding=embedding)


def make_value_fn(graphdef: nnx.GraphDef, config: MuZeroConfig) -> ValueFn:
    """(params, obs) -> search-free root value (N,); used for the bootstrap tail in reanalyze."""
    support = Support.from_config(config)

    def value_fn(params, obs):
        model = nnx.merge(graphdef, params)
        return root_output(model, support, obs).value

    return value_fn


class Targets(NamedTuple):
    """Loss targets built by `make_targets`."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward_probs: jnp.ndarray
    value_probs: jnp.ndarray
    policy_probs: jnp.ndarray
    in_episode: jnp.ndarray  # 1 inside the episode, 0 after it ends


UnrollStep = Callable[[Any, jnp.ndarray, Targets, Targets], tuple[jnp.ndarray, dict[str, jnp.ndarray]]]


def masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean over the unmasked entries."""
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1e-6)


def cross_entropy(logits: jnp.ndarray, target_probs: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    targets = jax.lax.stop_gradient(target_probs)
    losses = optax.softmax_cross_entropy(logits, targets)
    if mask is None:
        return jnp.mean(losses)
    return masked_mean(losses, mask)


def value_targets(config: MuZeroConfig, experience: Any) -> jnp.ndarray:
    """n-step TD(lambda) returns; truncated steps bootstrap from `bootstrap_value` with lambda=0."""
    truncated = experience.truncated[..., :-1]
    bootstrap = jnp.where(truncated > 0, experience.bootstrap_value[..., :-1], experience.value[..., 1:])
    lambdas = config.bootstrapping_lambda * (1.0 - truncated)
    discounts = experience.discount[..., :-1] * config.discount
    rewards = experience.reward[..., :-1]

    def n_step(reward, discount, value, lam):
        return rlax.n_step_bootstrapped_returns(reward, discount, value, config.num_bootstrapping, lam, stop_target_gradients=True)

    return jax.vmap(n_step)(rewards, discounts, bootstrap, lambdas)


def make_targets(config: MuZeroConfig, support: Support, experience: Any) -> tuple[Targets, jnp.ndarray]:
    """Builds loss targets from replay transitions (B, T, ...)."""
    value_target = value_targets(config, experience)
    alive = experience.discount * (1.0 - experience.truncated)
    in_episode = jnp.concatenate([jnp.ones_like(alive[:, :1]), jnp.cumprod(alive[:, :-1], axis=-1)], axis=-1)
    targets = Targets(
        observation=experience.observation,
        action=experience.action,
        reward_probs=support.to_probs(experience.reward),
        value_probs=support.to_probs(value_target),
        policy_probs=experience.policy_probs,
        in_episode=in_episode,
    )
    return targets, value_target


def loss(
    graphdef: nnx.GraphDef,
    config: MuZeroConfig,
    params: Any,
    batch: Any,
    *,
    unroll_step: UnrollStep,
    loss_weights: dict[str, float],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Root loss plus the weighted K-step unroll loss."""
    model = nnx.merge(graphdef, params)
    support = Support.from_config(config)
    experience = batch.experience
    targets, value_target = make_targets(config, support, experience)

    # Root step (t = 0)
    latent = model.representation(targets.observation[:, 0])
    value_logits, policy_logits = model.prediction(latent)
    root_value_loss = cross_entropy(value_logits, targets.value_probs[:, 0])
    root_policy_loss = cross_entropy(policy_logits, targets.policy_probs[:, 0])
    root_loss = config.value_loss_weight * root_value_loss + root_policy_loss

    # Unroll K steps (t = 0 .. K-1)
    k = config.num_unroll_steps
    by_time = jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), targets)
    now = jax.tree.map(lambda x: x[:k], by_time)
    next_ = jax.tree.map(lambda x: x[1 : k + 1], by_time)

    def scan_step(latent, step):
        now_t, next_t = step
        next_latent, step_losses = unroll_step(model, latent, now_t, next_t)
        return scale_gradient(next_latent, config.latent_gradient_scale), step_losses

    _, unroll_losses = jax.lax.scan(scan_step, latent, (now, next_))
    unroll_losses = {name: jnp.mean(values) for name, values in unroll_losses.items()}
    unroll_loss = sum(loss_weights.get(name, 1.0) * value for name, value in unroll_losses.items())
    total_loss = root_loss + unroll_loss

    policy = distrax.Categorical(logits=policy_logits)
    metrics = {
        "loss": total_loss,
        "root_value_loss": root_value_loss,
        "root_policy_loss": root_policy_loss,
        "root_policy_entropy": jnp.mean(policy.entropy()),
        "root_value_pred_mean": jnp.mean(support.logits_to_scalar(value_logits)),
        "unroll_loss": unroll_loss,
        **{f"unroll_{name}_loss": value for name, value in unroll_losses.items()},
        "target_reward_mean": jnp.mean(experience.reward),
        "target_reward_std": jnp.std(experience.reward),
        "target_reward_abs_mean": jnp.mean(jnp.abs(experience.reward)),
        "target_value_mean": jnp.mean(value_target),
        "target_value_std": jnp.std(value_target),
        "mcts_value_mean": jnp.mean(experience.value),
        "mcts_value_std": jnp.std(experience.value),
    }
    return total_loss, metrics
