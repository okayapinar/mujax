"""Sampled MuZero: the sampled prior and the search restricted to the sampled actions."""

import jax
import jax.numpy as jnp
import numpy as np
from conftest import tiny_sampled_mz

from mujax.algorithms import SampledMZ, algorithm_for
from mujax.algorithms.mz import MZ
from mujax.algorithms.sampled_mz import sampled_prior_logits
from mujax.types import EnvironmentSpec


def test_algorithm_for_prefers_exact_class():
    assert algorithm_for(tiny_sampled_mz()) is SampledMZ
    from mujax.algorithms import MZConfig

    assert algorithm_for(MZConfig()) is MZ


def test_sampled_prior_is_empirical_distribution_at_unit_temperature():
    logits = jnp.log(jnp.array([[0.7, 0.2, 0.1, 0.0 + 1e-9]] * 64))
    prior_logits, sampled = sampled_prior_logits(jax.random.PRNGKey(0), logits, None, num_samples=5, temperature=1.0)
    probs = jax.nn.softmax(prior_logits)
    # Only sampled actions carry mass, and the (near-)zero-probability action is never drawn.
    assert bool(jnp.all((probs > 0) == sampled))
    assert not bool(jnp.any(sampled[:, 3]))
    # With beta = pi the prior is beta_hat: multiples of 1/K that sum to 1.
    counts = probs * 5
    np.testing.assert_allclose(counts, jnp.round(counts), atol=1e-4)
    np.testing.assert_allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)


def test_sampled_prior_respects_invalid_actions_and_temperature():
    logits = jnp.zeros((32, 4))
    invalid = jnp.array([[1.0, 0.0, 1.0, 0.0]] * 32)
    prior_logits, sampled = sampled_prior_logits(jax.random.PRNGKey(1), logits, invalid, num_samples=8, temperature=0.5)
    assert not bool(jnp.any(sampled[:, [0, 2]]))
    assert bool(jnp.all(jnp.isfinite(jax.nn.softmax(prior_logits))))
    # Uniform logits: pi = beta regardless of temperature, so the prior is again beta_hat.
    counts = jax.nn.softmax(prior_logits) * 8
    np.testing.assert_allclose(counts, jnp.round(counts), atol=1e-4)


def test_search_visits_only_sampled_actions():
    config = tiny_sampled_mz()
    spec = EnvironmentSpec(obs_dim=4, num_actions=6)
    networks = SampledMZ.make_networks(spec, config)
    params = SampledMZ.init_params(networks, spec, jax.random.PRNGKey(0))
    policy = jax.jit(SampledMZ.make_policy(networks, spec, config, evaluation=False))

    obs = jnp.zeros((16, 4))
    invalid = jnp.zeros((16, 6)).at[:, 5].set(1.0)
    out = policy(params, obs, jax.random.PRNGKey(2), invalid, 0)

    assert out.action.shape == (16,) and out.policy_probs.shape == (16, 6) and out.value.shape == (16,)
    # K=3 samples: at most 3 distinct root actions get visits; the invalid action never does.
    assert bool(jnp.all(jnp.sum(out.policy_probs > 0, axis=-1) <= 3))
    assert bool(jnp.all(out.policy_probs[:, 5] == 0))
    assert bool(jnp.all(out.action != 5))
    np.testing.assert_allclose(jnp.sum(out.policy_probs, axis=-1), 1.0, atol=1e-5)
