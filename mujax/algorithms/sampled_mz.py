"""Sampled MuZero: config and search; the model and loss are MuZero's (mz.py).

Hubert et al., *Learning and Planning in Complex Action Spaces* (2021). Whenever a node is expanded (root and
interior), K actions are sampled with replacement from the proposal distribution β = π^(1/τ) and the search is
restricted to them. With β̂(a) = count(a) / K the empirical distribution of the samples, the PUCT prior is
π̂_β ∝ (β̂ / β) · π (Section 5.1); for τ = 1 this is simply β̂. At the root the Dirichlet exploration noise is
added to π before sampling, so that β and π both carry the noise (Section 5.5).

The visit-count distribution over the sampled actions is the sample-based improved policy Iπ̂_β (Section 5.2);
training π with the cross-entropy against it (the MZ policy loss, with π normalized over the whole action space)
minimizes KL(Iπ̂_β ‖ π), so the loss, value and reward targets are unchanged from MuZero.

Appendix A's Go-specific trick (evaluating all sampled root actions before the search to initialize Q) is not
implemented: mctx exposes no hook for it.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import mctx
from flax import nnx

from mujax.algorithm import Algorithm, SearchPolicy
from mujax.algorithms import muzero, mz
from mujax.algorithms.muzero import Support
from mujax.types import EnvironmentSpec, SearchOutput


@dataclasses.dataclass
class SampledMZConfig(mz.MZConfig):
    """MuZero hyperparameters + the sampling of Sampled MuZero."""

    num_sampled_actions: int = 16  # K; sampled with replacement, so it may exceed the number of actions
    sample_temperature: float = 1.0  # β = π^(1/τ); 1 gives β = π as in the paper's experiments


def sampled_prior_logits(
    key: jax.Array,
    policy_logits: jnp.ndarray,
    invalid_actions: jnp.ndarray | None,
    num_samples: int,
    temperature: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Samples K actions per row from β = softmax(logits / τ) and returns (prior_logits, sampled).

    prior_logits: log((β̂ / β) · π) on the sampled actions, the dtype's minimum elsewhere (zero mass under
    softmax, as in mctx's own masking). Invalid actions (mask value 1) are never sampled.
    sampled: True for actions drawn at least once.
    """
    min_logit = jnp.finfo(policy_logits.dtype).min
    num_actions = policy_logits.shape[-1]

    def masked(logits):
        return logits if invalid_actions is None else jnp.where(invalid_actions > 0, min_logit, logits)

    log_pi = jax.nn.log_softmax(masked(policy_logits))
    log_beta = jax.nn.log_softmax(masked(policy_logits / temperature))
    samples = jax.random.categorical(key, log_beta, shape=(num_samples, *policy_logits.shape[:-1]))  # (K, B)
    counts = jnp.sum(jax.nn.one_hot(samples, num_actions, dtype=policy_logits.dtype), axis=0)  # (B, A)
    sampled = counts > 0
    log_beta_hat = jnp.log(jnp.maximum(counts, 1.0) / num_samples)  # counts >= 1 where sampled; the rest is masked
    return jnp.where(sampled, log_beta_hat - log_beta + log_pi, min_logit), sampled


def noisy_logits(key: jax.Array, policy_logits: jnp.ndarray, dirichlet_fraction: float, dirichlet_alpha: float) -> jnp.ndarray:
    """log((1 - f) · softmax(logits) + f · Dirichlet(α)); the root exploration noise of MuZero, applied before sampling."""
    batch_size, num_actions = policy_logits.shape
    noise = jax.random.dirichlet(key, jnp.full((num_actions,), dirichlet_alpha), shape=(batch_size,))
    probs = (1.0 - dirichlet_fraction) * jax.nn.softmax(policy_logits) + dirichlet_fraction * noise
    return jnp.log(jnp.maximum(probs, jnp.finfo(probs.dtype).tiny))


def make_policy(graphdef: nnx.GraphDef, spec: EnvironmentSpec, config: SampledMZConfig, evaluation: bool) -> SearchPolicy:
    """Batched Sampled MuZero (PUCT over K sampled actions). `evaluation=True` turns off temperature and Dirichlet noise.

    Unsampled actions get zero prior mass, so their PUCT score is only the completed Q-value, which
    `qtransform_by_parent_and_siblings` sets to 0 for unvisited actions; every sampled action scores strictly
    higher, so interior nodes never leave the sampled set. At the root they are also masked as invalid.
    """
    support = Support.from_config(config)
    exploration = muzero.puct_exploration(config, spec.num_actions, evaluation)
    model_step = mz.recurrent_fn(graphdef, support, config)
    num_samples, temperature = int(config.num_sampled_actions), float(config.sample_temperature)

    def recurrent_fn(params, key, action, latent):
        output, next_latent = model_step(params, key, action, latent)
        prior_logits, _ = sampled_prior_logits(key, output.prior_logits, None, num_samples, temperature)
        return output.replace(prior_logits=prior_logits), next_latent

    def policy(params, obs, key, invalid_actions, learner_steps) -> SearchOutput:
        kwargs = exploration(learner_steps)
        key, noise_key, sample_key = jax.random.split(key, 3)
        root = muzero.root_output(nnx.merge(graphdef, params), support, obs)
        logits = noisy_logits(noise_key, root.prior_logits, kwargs["dirichlet_fraction"], kwargs["dirichlet_alpha"])
        prior_logits, sampled = sampled_prior_logits(sample_key, logits, invalid_actions, num_samples, temperature)
        out = mctx.muzero_policy(
            params,
            key,
            root.replace(prior_logits=prior_logits),
            recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=jnp.maximum(invalid_actions, (~sampled).astype(invalid_actions.dtype)),
            max_depth=config.max_depth,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            temperature=kwargs["temperature"],
            dirichlet_fraction=0.0,  # already in `logits`
            dirichlet_alpha=kwargs["dirichlet_alpha"],
            pb_c_init=kwargs["pb_c_init"],
            pb_c_base=kwargs["pb_c_base"],
        )
        return SearchOutput(out.action, out.action_weights, out.search_tree.summary().value)

    return policy


SampledMZ = Algorithm("sampled_mz", SampledMZConfig, mz.MZModel, make_policy, mz.loss, muzero.make_value_fn)
