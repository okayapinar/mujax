"""Components shared by the MuZero-family algorithms (SMZ, GMZ).

The module has three sections:
    1. `Support`: categorical (HL-Gauss) representation of value/reward; scalar <-> probability conversions.
    2. Network blocks: LayerNorm MLP, Head, Embedding, Prediction and small helpers.
    3. Loss skeleton: target construction, root loss, unroll loss over `num_unroll_steps`, and metrics.
    4. PUCT exploration: temperature and Dirichlet noise config shared by MZ and SMZ.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Self

import distrax
import jax
import jax.numpy as jnp
import mctx
import optax
import rlax
from flax import nnx
from jaxtyping import Array, Float, Int

from mujax.algorithm import ValueFn
from mujax.config import MuZeroConfig

# ---------------------------------------------------------------------------
# 1. Categorical value/reward: HL-Gauss targets in symlog space
# ---------------------------------------------------------------------------


class Support(NamedTuple):
    """Splits `[min, max]` into `num_bins` evenly spaced bins; scalars are represented in symlog space.

    Target distribution (HL-Gauss): the mass that a Gaussian centered at symlog(x) with standard
    deviation `sigma_scale * bin_width` places on each bin. Unlike two-hot targets, it also spreads
    mass onto neighboring bins.
    """

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
    def bin_centers(self) -> Float[Array, " bins"]:
        return jnp.linspace(self.min, self.max, self.num_bins)

    def to_probs(self, x: Float[Array, "*shape"]) -> Float[Array, "*shape bins"]:
        """Scalar -> probabilities over the bins."""
        center = jnp.clip(rlax.signed_logp1(x), self.min, self.max)[..., None]
        gauss_cdf = functools.partial(jax.scipy.stats.norm.cdf, loc=center, scale=self.sigma_scale * self.bin_width)
        half = self.bin_width / 2
        probs = gauss_cdf(self.bin_centers + half) - gauss_cdf(self.bin_centers - half)
        return probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1e-8)

    def to_scalar(self, probs: Float[Array, "*shape bins"]) -> Float[Array, "*shape"]:
        """Inverse of `to_probs`: expectation in symlog space, then symexp."""
        return rlax.signed_expm1(rlax.transform_from_2hot(probs, self.min, self.max, self.num_bins))

    def logits_to_scalar(self, logits: Float[Array, "*shape bins"]) -> Float[Array, "*shape"]:
        return self.to_scalar(jax.nn.softmax(logits))


# ---------------------------------------------------------------------------
# 2. Network blocks
# ---------------------------------------------------------------------------


class LayerNormMLP(nnx.Module):
    """MLP whose layers are Linear -> LayerNorm -> ReLU."""

    def __init__(self, in_features: int, layer_sizes: Sequence[int], *, rngs: nnx.Rngs):
        sizes = [in_features, *layer_sizes]
        self.linears = nnx.List([nnx.Linear(i, o, rngs=rngs) for i, o in zip(sizes[:-1], sizes[1:])])
        self.norms = nnx.List([nnx.LayerNorm(o, rngs=rngs) for o in layer_sizes])
        self.out_features = sizes[-1]

    def __call__(self, x: Float[Array, "*batch I"]) -> Float[Array, "*batch O"]:
        for linear, norm in zip(self.linears, self.norms):
            x = nnx.relu(norm(linear(x)))
        return x


class Head(nnx.Module):
    """LayerNormMLP + linear output; used for policy/value/reward logits."""

    def __init__(self, in_features: int, layer_sizes: Sequence[int], output_dim: int, *, rngs: nnx.Rngs):
        self.mlp = LayerNormMLP(in_features, layer_sizes, rngs=rngs)
        self.out = nnx.Linear(self.mlp.out_features, output_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "*batch I"]) -> Float[Array, "*batch O"]:
        return self.out(self.mlp(x))


class Embedding(nnx.Module):
    """Head + min-max normalization; used by networks that produce latent states (representation, dynamics, afterstate)."""

    def __init__(self, in_features: int, layer_sizes: Sequence[int], embedding_dim: int, *, rngs: nnx.Rngs):
        self.head = Head(in_features, layer_sizes, embedding_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "*batch I"]) -> Float[Array, "*batch D"]:
        return min_max_normalize(self.head(x))


class Prediction(nnx.Module):
    """Latent -> (value_logits, policy_logits)."""

    def __init__(self, embedding_dim: int, layer_sizes: Sequence[int], num_actions: int, num_bins: int, *, rngs: nnx.Rngs):
        self.value = Head(embedding_dim, layer_sizes, num_bins, rngs=rngs)
        self.policy = Head(embedding_dim, layer_sizes, num_actions, rngs=rngs)

    def __call__(self, latent: Float[Array, "B D"]) -> tuple[Float[Array, "B bins"], Float[Array, "B A"]]:
        return self.value(latent), self.policy(latent)


def min_max_normalize(x: Float[Array, "*batch D"]) -> Float[Array, "*batch D"]:
    """Rescales the last axis to [0, 1] (the latent normalization from the MuZero paper)."""
    x_min = x.min(axis=-1, keepdims=True)
    x_max = x.max(axis=-1, keepdims=True)
    return (x - x_min) / jnp.maximum(x_max - x_min, 1e-5)


def scale_gradient(x: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Identity on the forward pass; multiplies the gradient by `scale` on the backward pass."""
    return x * scale + jax.lax.stop_gradient(x) * (1.0 - scale)


def one_hot_concat(latent: Float[Array, "B D"], index: Int[Array, " B"], num_classes: int) -> Float[Array, "B D+C"]:
    """Appends a one-hot action/code to the latent; the input of the dynamics and decision networks."""
    return jnp.concatenate([latent, jax.nn.one_hot(index, num_classes)], axis=-1)


def root_output(model: Any, support: Support, obs: Float[Array, "B O"]) -> mctx.RootFnOutput:
    """representation -> prediction; the root input of the mctx search."""
    embedding = model.representation(obs)
    value_logits, policy_logits = model.prediction(embedding)
    return mctx.RootFnOutput(prior_logits=policy_logits, value=support.logits_to_scalar(value_logits), embedding=embedding)


def make_value_fn(graphdef: nnx.GraphDef, config: MuZeroConfig) -> ValueFn:
    """(params, obs) -> search-free root value (N,); used for the bootstrap tail in reanalyze."""
    support = Support.from_config(config)
    return lambda params, obs: root_output(nnx.merge(graphdef, params), support, obs).value


# ---------------------------------------------------------------------------
# 3. Loss skeleton
# ---------------------------------------------------------------------------


class Targets(NamedTuple):
    """Loss targets, built by `make_targets`.

    Fields have shape (B, T, ...), except value_probs (B, T-1, ...): the last step has no bootstrap target.
    The scan inside `loss` slices them along the time axis, so `UnrollStep` sees (B, ...) for a single t.
    """

    observation: jnp.ndarray
    action: jnp.ndarray
    reward_probs: jnp.ndarray  # support.to_probs(reward)
    value_probs: jnp.ndarray  # support.to_probs(n-step bootstrapped return)
    policy_probs: jnp.ndarray  # MCTS visit distribution (possibly reanalyzed)
    in_episode: jnp.ndarray  # 1 inside the episode, 0 after it ends (loss mask)


# A single unroll step of an algorithm: (model, latent, now, next) -> (next_latent, losses, extras).
#   now / next: targets at times t and t+1, (B, ...).
#   losses: name -> scalar; averaged over steps and weighted by `loss_weights`.
#   extras: name -> array; stacked along time as (K, ...) and added to the metrics.
UnrollStep = Callable[[Any, jnp.ndarray, Targets, Targets], tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]]


def cross_entropy(logits: jnp.ndarray, target_probs: jnp.ndarray) -> jnp.ndarray:
    return optax.softmax_cross_entropy(logits, jax.lax.stop_gradient(target_probs))


def masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Divides by the mask sum, so masked-out samples do not change the scale."""
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1e-6)


def in_episode_mask(discount: Float[Array, "B T"], truncated: Float[Array, "B T"]) -> Float[Array, "B T"]:
    """1 at t=0, then 0 after the episode ends (terminated: discount 0, or truncated)."""
    alive = discount * (1.0 - truncated)
    return jnp.concatenate([jnp.ones_like(alive[:, :1]), jnp.cumprod(alive[:, :-1], axis=-1)], axis=-1)


def value_targets(config: MuZeroConfig, batch: Any) -> Float[Array, "B T-1"]:
    """n-step bootstrapped returns; the bootstrap value is the (possibly reanalyzed) search value from replay.

    At a truncated step the next replay entry already belongs to a new episode, so that step bootstraps
    from V(final_obs) (`bootstrap_value`) and lambda 0 stops the return from reaching past it.
    """
    truncated = batch.truncated[..., :-1]
    bootstrap = jnp.where(truncated > 0, batch.bootstrap_value[..., :-1], batch.value[..., 1:])
    lambdas = config.bootstrapping_lambda * (1.0 - truncated)

    def n_step(reward, discount, value, lambda_):
        return rlax.n_step_bootstrapped_returns(reward, discount, value, n=config.num_bootstrapping, lambda_t=lambda_, stop_target_gradients=True)

    return jax.vmap(n_step)(batch.reward[..., :-1], batch.discount[..., :-1] * config.discount, bootstrap, lambdas)


def make_targets(config: MuZeroConfig, support: Support, experience: Any) -> tuple[Targets, jnp.ndarray]:
    """Builds loss targets from replay transitions (B, T, ...); also returns the raw value target for metrics."""
    value_target = value_targets(config, experience)
    targets = Targets(
        observation=experience.observation,
        action=experience.action,
        reward_probs=support.to_probs(experience.reward),
        value_probs=support.to_probs(value_target),
        policy_probs=experience.policy_probs,
        in_episode=in_episode_mask(experience.discount, experience.truncated),
    )
    return targets, value_target


def _root_loss(model: Any, config: MuZeroConfig, targets: Targets) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """representation + prediction on the t=0 observation; returns (root_loss, latent, metrics)."""
    latent = model.representation(targets.observation[:, 0])
    value_logits, policy_logits = model.prediction(latent)
    value_loss = jnp.mean(cross_entropy(value_logits, targets.value_probs[:, 0]))
    policy_loss = jnp.mean(cross_entropy(policy_logits, targets.policy_probs[:, 0]))
    root_loss = config.value_loss_weight * value_loss + policy_loss
    metrics = {
        "root_value_loss": value_loss,
        "root_policy_loss": policy_loss,
        "root_policy_entropy": jnp.mean(distrax.Categorical(logits=policy_logits).entropy()),
        "root_value_pred_mean": jnp.mean(Support.from_config(config).logits_to_scalar(value_logits)),
    }
    return root_loss, latent, metrics


def _unroll_loss(
    config: MuZeroConfig, model: Any, latent: jnp.ndarray, targets: Targets, unroll_step: UnrollStep, loss_weights: dict[str, float]
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Runs `unroll_step` for `num_unroll_steps` steps with a scan.

    Returns the weighted total, the unweighted losses (name -> scalar), and the extras stacked along time.
    """
    k = config.num_unroll_steps

    def scan_step(latent, step):
        now, next_ = step
        latent, losses, extras = unroll_step(model, latent, now, next_)
        return scale_gradient(latent, config.latent_gradient_scale), (losses, extras)

    time_major = jax.tree.map(lambda a: jnp.moveaxis(a, 1, 0), targets)  # (T, B, ...)
    now_steps = jax.tree.map(lambda a: a[:k], time_major)
    next_steps = jax.tree.map(lambda a: a[1 : k + 1], time_major)
    _, (losses, extras) = jax.lax.scan(scan_step, latent, (now_steps, next_steps))

    losses = {name: jnp.sum(values) / k for name, values in losses.items()}
    weighted_total = sum(loss_weights.get(name, 1.0) * value for name, value in losses.items())
    return weighted_total, losses, extras


def _target_metrics(experience: Any, value_target: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Summary statistics for monitoring the target distributions."""
    return {
        "target_reward_mean": jnp.mean(experience.reward),
        "target_reward_std": jnp.std(experience.reward),
        "target_reward_abs_mean": jnp.mean(jnp.abs(experience.reward)),
        "target_value_mean": jnp.mean(value_target),
        "target_value_std": jnp.std(value_target),
        "mcts_value_mean": jnp.mean(experience.value),
        "mcts_value_std": jnp.std(experience.value),
    }


def loss(
    graphdef: nnx.GraphDef,
    config: MuZeroConfig,
    params: Any,
    batch: Any,
    *,
    unroll_step: UnrollStep,
    loss_weights: dict[str, float],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Root loss plus the weighted sum of `unroll_step` losses over `num_unroll_steps`.

    Losses missing from `loss_weights` get weight 1. Metrics report the losses unweighted.
    """
    model = nnx.merge(graphdef, params)
    support = Support.from_config(config)
    experience = batch.experience
    targets, value_target = make_targets(config, support, experience)

    root_loss, latent, root_metrics = _root_loss(model, config, targets)
    unroll_loss, unroll_losses, extras = _unroll_loss(config, model, latent, targets, unroll_step, loss_weights)
    total_loss = root_loss + unroll_loss

    metrics = {
        "loss": total_loss,
        **root_metrics,
        "unroll_loss": unroll_loss,
        **{f"unroll_{name}_loss": value for name, value in unroll_losses.items()},
        **_target_metrics(experience, value_target),
        **extras,
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# 4. PUCT exploration (MZ, SMZ)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PUCTConfig(MuZeroConfig):
    """MuZeroConfig + exploration of the PUCT search (`mctx.muzero_policy`, `mctx.stochastic_muzero_policy`)."""

    temperature_decay_steps: int = 400_000
    dirichlet_fraction: float = 0.1
    dirichlet_alpha: float | None = None  # If None, 1/sqrt(num_actions)
    pb_c_init: float = 1.25
    pb_c_base: float = 19652.0

    def with_num_steps(self, num_steps: int) -> Self:
        return dataclasses.replace(super().with_num_steps(num_steps), temperature_decay_steps=num_steps)


def puct_exploration(config: PUCTConfig, num_actions: int, evaluation: bool) -> Callable[[Any], dict[str, Any]]:
    """learner_steps -> exploration kwargs of the mctx PUCT policies; `evaluation=True` turns off temperature and noise."""
    dirichlet_alpha = config.dirichlet_alpha if config.dirichlet_alpha is not None else 1.0 / num_actions**0.5
    steps = config.temperature_decay_steps
    temperature_schedule = optax.piecewise_constant_schedule(1.0, {int(0.5 * steps): 0.5, int(0.75 * steps): 0.5})

    def kwargs(learner_steps):
        return {
            "temperature": 0.0 if evaluation else temperature_schedule(learner_steps),
            "dirichlet_fraction": 0.0 if evaluation else config.dirichlet_fraction,
            "dirichlet_alpha": dirichlet_alpha,
            "pb_c_init": config.pb_c_init,
            "pb_c_base": config.pb_c_base,
        }

    return kwargs
